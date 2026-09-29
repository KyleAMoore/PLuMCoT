"""plumcot/scorers.py: scorers, message builders, NaN filling and composition."""
import json
import math
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import ArrayScorer, cont, dist_from_probs
from fake_backend import EOS, FakeBackend, FakeTokenizer
from plumcot import (Branch, CompositeScorer, LogLikelihoodScorer, ScoreContext,
                     SelfCertaintyScorer, TokenDist, VerdictJudgeScorer, fill_nan,
                     guard_messages, llama_guard_scorer, policy_judge_scorer, policy_messages,
                     qwen3guard_scorer)
from plumcot.base import ValidationContext
from plumcot.config import PLuMCoTConfig

GEN_TOK = FakeTokenizer()


def ctx(user="hi", response=(20, 21)):
    return ScoreContext(user, [1, 2], list(response), GEN_TOK)


# ===========================================================================
# fill_nan
# ===========================================================================

def test_fill_logprob_kind_uses_log_mean_probability():
    x, all_nan = fill_nan(np.array([math.log(0.9), math.log(0.1), math.nan]), "logprob")
    assert not all_nan and x[2] == pytest.approx(math.log(0.5))


def test_fill_utility_kind_uses_arithmetic_mean():
    x, _ = fill_nan(np.array([1.0, 3.0, math.nan]), "utility")
    assert x[2] == 2.0


def test_fill_keeps_infinities_and_ignores_them_for_the_fill_value():
    x, _ = fill_nan(np.array([-math.inf, 0.0, 2.0, math.nan]), "utility")
    assert x[0] == -math.inf and x[3] == 1.0
    x, _ = fill_nan(np.array([math.inf, 0.0]), "utility")
    assert x[0] == math.inf


def test_fill_all_nan_is_flagged():
    x, all_nan = fill_nan(np.array([math.nan, math.nan]), "logprob")
    assert all_nan and list(x) == [0.0, 0.0]


def test_fill_nan_next_to_only_infinities_raises():
    with pytest.raises(ValueError, match="no neutral value"):
        fill_nan(np.array([math.nan, -math.inf]), "logprob")


def test_fill_does_not_modify_its_input_and_rejects_unknown_kind():
    a = np.array([1.0, math.nan])
    fill_nan(a, "utility")
    assert math.isnan(a[1])
    with pytest.raises(ValueError, match="kind"):
        fill_nan(a, "probability")


# ===========================================================================
# LogLikelihoodScorer
# ===========================================================================

def test_loglik_mean_and_sum():
    b = [Branch(5, -1.0, cont([6, 7, 8], [-0.5, -1.0, -1.5]))]
    assert LogLikelihoodScorer("mean").score(ctx(), b)[0] == pytest.approx(-1.0)
    assert LogLikelihoodScorer("sum").score(ctx(), b)[0] == pytest.approx(-3.0)


def test_loglik_empty_lookahead_is_nan():
    b = [Branch(EOS, -1.0, cont([], []))]
    assert math.isnan(LogLikelihoodScorer().score(ctx(), b)[0])


@pytest.mark.parametrize("c, match", [
    (cont([6, 7], None), "one logprob per"),
    (cont([6, 7], [-0.5]), "one logprob per"),
    (cont([6, 7], [-0.5, math.nan]), "finite"),
    (cont([6, 7], [-0.5, -math.inf]), "finite"),
])
def test_loglik_rejects_missing_or_bad_logprobs(c, match):
    with pytest.raises(RuntimeError, match=match):
        LogLikelihoodScorer().score(ctx(), [Branch(5, -1.0, c)])


def test_loglik_declarations_and_settings():
    s = LogLikelihoodScorer(scale=2.0)
    assert (s.needs_logprobs, s.kind, s.scale, s.default_aggregate) == (
        "chosen", "utility", 2.0, "mean")
    with pytest.raises(ValueError, match="reduce"):
        LogLikelihoodScorer(reduce="avg")
    with pytest.raises(ValueError, match="scale"):
        LogLikelihoodScorer(scale=math.nan)


# ===========================================================================
# SelfCertaintyScorer
# ===========================================================================

@pytest.fixture
def certainty_branch():
    d1 = dist_from_probs([0.5, 0.25])            # tail 0.25
    d2 = dist_from_probs([0.9, 0.05])            # tail 0.05
    return [Branch(5, -1.0, cont([6, 7], [-0.7, -0.1], [d1, d2]))], (d1, d2)


def test_certainty_metrics_match_hand_computation(certainty_branch):
    b, (d1, d2) = certainty_branch
    assert SelfCertaintyScorer("neg_entropy").score(ctx(), b)[0] == pytest.approx(
        -(d1.entropy() + d2.entropy()) / 2)
    assert SelfCertaintyScorer("neg_entropy", tail="uniform").score(ctx(), b)[0] == pytest.approx(
        -(d1.entropy("uniform") + d2.entropy("uniform")) / 2)
    assert SelfCertaintyScorer("max_logprob").score(ctx(), b)[0] == pytest.approx(
        (math.log(0.5) + math.log(0.9)) / 2)
    assert SelfCertaintyScorer("self_certainty").score(ctx(), b)[0] == pytest.approx(
        -(math.log(0.25) + math.log(0.05)) / 2)


def test_certainty_empty_is_nan_and_missing_dists_raise():
    assert math.isnan(SelfCertaintyScorer().score(ctx(), [Branch(EOS, 0.0, cont([], [], []))])[0])
    with pytest.raises(RuntimeError, match="distribution per lookahead token"):
        SelfCertaintyScorer().score(ctx(), [Branch(5, 0.0, cont([6], [-0.1], None))])


def test_certainty_declarations_and_settings():
    assert SelfCertaintyScorer().needs_logprobs == "dist"
    with pytest.raises(ValueError, match="metric"):
        SelfCertaintyScorer(metric="entropy")
    with pytest.raises(ValueError, match="tail"):
        SelfCertaintyScorer(tail="lump")


# ===========================================================================
# VerdictJudgeScorer
# ===========================================================================

class JudgeTokenizer:
    VOCAB = {0: "<eos>", 1: "safe", 2: "unsafe", 3: "Yes", 4: "No", 5: "Safety", 6: ":",
             7: "\n\n", 8: " Uns", 9: "s", 10: "Controversial", 11: " SAFE ", 12: "un",
             13: "saf"}

    def __init__(self):
        self.decoded = []
        self.templates = []

    def decode(self, ids, skip_special_tokens=False):
        self.decoded += list(ids)
        return "".join(self.VOCAB.get(int(i), f" w{int(i)}") for i in ids)

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, **kw):
        self.templates.append((messages, dict(kw, add_generation_prompt=add_generation_prompt)))
        return [100 + len(self.templates)]


class ScriptedJudge:
    """Answers each prompt with the next queued verdict continuation."""

    def __init__(self, verdicts=(), num_logprobs=20):
        self.tokenizer = JudgeTokenizer()
        self.num_logprobs = num_logprobs
        self.verdicts = list(verdicts)
        self.calls = []
        self.cap_checks = 0

    def check_logprob_cap(self):
        self.cap_checks += 1

    def generate(self, contexts, **kw):
        self.calls.append(dict(contexts=list(contexts), **kw))
        out = [[self.verdicts.pop(0)] for _ in contexts]
        return out


def verdict(tokens, probs):
    """tokens: the judge's greedy output; probs: one {token_id: prob} per position."""
    return cont(tokens, None, [TokenDist.from_items([(t, math.log(p)) for t, p in d.items()], 40)
                               for d in probs])


def judge_scorer(verdicts, **kw):
    j = ScriptedJudge(verdicts)
    kw.setdefault("positive", ("safe",))
    kw.setdefault("negative", ("unsafe",))
    return VerdictJudgeScorer(j, guard_messages(), **kw), j


def one_branch():
    return [Branch(22, -1.0, cont([23]))]


def test_verdict_ratio_at_first_label_position():
    v = verdict([7, 1, 2], [{7: 1.0}, {1: 0.6, 2: 0.3, 5: 0.1}, {2: 0.99}])
    sc, _ = judge_scorer([v], transform="prob")
    assert sc.score(ctx(), one_branch())[0] == pytest.approx(0.6 / 0.9)


def test_verdict_sums_all_tokens_mapping_to_a_label():
    # " SAFE " and "saf" both map to "safe"; " Uns" and "un" to "unsafe"
    v = verdict([1], [{1: 0.3, 11: 0.1, 13: 0.1, 8: 0.2, 12: 0.1, 5: 0.2}])
    sc, _ = judge_scorer([v], transform="prob")
    assert sc.score(ctx(), one_branch())[0] == pytest.approx(0.5 / 0.8)


def test_label_matching_rules():
    sc, _ = judge_scorer([])
    assert sc._label_of(1) == "safe" and sc._label_of(11) == "safe"   # case/whitespace
    assert sc._label_of(8) == "unsafe" and sc._label_of(12) == "unsafe"  # unique prefixes
    assert sc._label_of(9) is None                                      # 1 character
    assert sc._label_of(5) is None                                      # not a label
    amb, _ = judge_scorer([], positive=("safe",), negative=("safety",))
    assert amb._label_of(13) is None                                    # "saf": ambiguous
    assert amb._label_of(1) == "safe"                                   # exact wins


def test_label_lookups_are_cached():
    sc, j = judge_scorer([])
    sc._label_of(1), sc._label_of(1), sc._label_of(2)
    assert j.tokenizer.decoded == [1, 2]


@pytest.mark.parametrize("mode, expect", [("negative", 0.2), ("positive", 0.8),
                                          ("neutral", 0.5)])
def test_qwen3guard_controversial_modes(mode, expect):
    v = verdict([5, 6, 10], [{5: 1.0}, {6: 1.0}, {1: 0.2, 2: 0.2, 10: 0.6}])
    j = ScriptedJudge([v])
    sc = qwen3guard_scorer(j, controversial=mode, transform="prob")
    assert sc.score(ctx(), one_branch())[0] == pytest.approx(expect)


def test_neutral_label_ends_the_scan():
    """A neutral label at position 0 is where the verdict is read, even
    though a positive label follows."""
    v = verdict([10, 1], [{10: 0.5, 1: 0.1, 2: 0.4}, {1: 1.0}])
    sc = qwen3guard_scorer(ScriptedJudge([v]), controversial="neutral", transform="prob")
    assert sc.score(ctx(), one_branch())[0] == pytest.approx(0.2)


def test_unparsed_verdicts():
    no_label = verdict([7, 5], [{7: 1.0}, {5: 1.0}])
    no_mass = verdict([10], [{10: 1.0}])                       # neutral label only
    sc = qwen3guard_scorer(ScriptedJudge([no_label, no_mass]), controversial="neutral")
    out = sc.score(ctx(), one_branch() * 2)
    assert np.isnan(out).all() and sc.n_failures == 2 and sc.n_calls == 2
    strict, _ = judge_scorer([verdict([7], [{7: 1.0}])], on_unparsed="raise")
    with pytest.raises(RuntimeError, match="could not parse"):
        strict.score(ctx(), one_branch())


@pytest.mark.parametrize("transform, p, expect, kind", [
    ("prob", 1.0, 1 - 1e-6, "utility"),
    ("prob", 0.25, 0.25, "utility"),
    ("logprob", 0.25, math.log(0.25), "logprob"),
    ("logprob", 1.0, math.log(1 - 1e-6), "logprob"),
    ("logit", 0.25, math.log(0.25 / 0.75), "utility"),
])
def test_transforms_and_clipping(transform, p, expect, kind):
    probs = {1: p} if p == 1.0 else {1: p, 2: 1 - p}
    sc, _ = judge_scorer([verdict([1], [probs])], transform=transform)
    assert sc.kind == kind
    assert sc.score(ctx(), one_branch())[0] == pytest.approx(expect)


def test_judge_call_and_prompt_construction():
    sc, j = judge_scorer([verdict([1], [{1: 0.9, 2: 0.1}])] * 2,
                         chat_template_kwargs={"enable_thinking": False}, scan_tokens=5)
    branches = [Branch(22, -1.0, cont([23, EOS])), Branch(EOS, -1.0, cont([]))]
    sc.score(ctx("USER", response=[20, 21]), branches)
    call = j.calls[0]
    assert len(call["contexts"]) == 2
    assert (call["n"], call["max_tokens"], call["temperature"], call["logprobs"]) == (
        1, 5, 0.0, "dist")
    (m0, kw0), (m1, _) = j.tokenizer.templates
    assert kw0 == {"enable_thinking": False, "add_generation_prompt": True}
    assert m0 == [{"role": "user", "content": "USER"},
                  {"role": "assistant", "content": " w20 w21 w22 w23"}]   # EOS skipped
    # a stop-token candidate is judged on the response as it stands
    assert m1[1]["content"] == " w20 w21"


def test_judge_backend_contract_errors():
    sc, j = judge_scorer([])
    j.generate = lambda contexts, **kw: []
    with pytest.raises(RuntimeError, match="wrong number of outputs"):
        sc.score(ctx(), one_branch())
    sc2, _ = judge_scorer([cont([1], None, None)])
    with pytest.raises(RuntimeError, match="distribution per token"):
        sc2.score(ctx(), one_branch())


@pytest.mark.parametrize("kw, match", [
    (dict(transform="logits"), "transform"),
    (dict(on_unparsed="skip"), "on_unparsed"),
    (dict(scan_tokens=0), "scan_tokens"),
    (dict(positive=()), "at least one"),
    (dict(negative=("safe",)), "more than one group"),
    (dict(neutral=("unsafe",)), "more than one group"),
])
def test_judge_rejects_bad_settings(kw, match):
    with pytest.raises(ValueError, match=match):
        judge_scorer([], **kw)


def test_judge_validation():
    gen = FakeBackend()
    vc = ValidationContext(PLuMCoTConfig(), 20, gen)
    sc, j = judge_scorer([])
    sc.validate(vc)
    assert j.cap_checks == 1
    j.num_logprobs = "auto"
    with pytest.raises(ValueError, match="Give the judge an int"):
        sc.validate(vc)
    # the generator's own backend is resolved and checked by the engine
    self_judge = policy_judge_scorer(gen)
    gen.num_logprobs = "auto"
    self_judge.validate(vc)


def test_judge_to_dict():
    sc, _ = judge_scorer([], neutral=("controversial",), name="g")
    d = json.loads(json.dumps(sc.to_dict()))
    assert d["name"] == "g" and d["neutral"] == ["controversial"] and d["kind"] == "logprob"


# -- message builders and presets ---------------------------------------------------------

def test_guard_messages_formats():
    s = guard_messages("string")(ctx("Q"), "A")
    assert s == [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]
    lst = guard_messages("list")(ctx("Q"), "A")
    assert lst[1]["content"] == [{"type": "text", "text": "A"}]
    with pytest.raises(ValueError, match="content_format"):
        guard_messages("dict")


def test_guard_messages_need_the_user_message():
    with pytest.raises(ValueError, match="user_message"):
        guard_messages()(ctx(None), "A")


def test_policy_messages():
    m = policy_messages(policy="POLICY_X", question="QUESTION_Y")(ctx("USER_Z"), "RESP_W")
    text = m[0]["content"]
    assert all(s in text for s in ("POLICY_X", "QUESTION_Y", "USER_Z", "RESP_W"))
    assert "(none)" in policy_messages()(ctx(None), "R")[0]["content"]
    assert "N/A" in policy_messages(missing_user="N/A")(ctx(None), "R")[0]["content"]


def test_presets():
    j = ScriptedJudge()
    lg = llama_guard_scorer(j, content_format="list")
    assert (lg.name, lg.positive, lg.negative) == ("llama_guard", {"safe"}, {"unsafe"})
    q = qwen3guard_scorer(j)
    assert "controversial" in q.negative and q.name == "qwen3guard"
    with pytest.raises(ValueError, match="controversial"):
        qwen3guard_scorer(j, controversial="unsafe")
    p = policy_judge_scorer(j, transform="prob")
    assert (p.positive, p.negative, p.kind) == ({"yes"}, {"no"}, "utility")


# ===========================================================================
# CompositeScorer
# ===========================================================================

def test_composite_declarations():
    comp = CompositeScorer([(LogLikelihoodScorer(), 1.0), (SelfCertaintyScorer(), 0.5)])
    assert comp.needs_logprobs == "dist" and comp.kind == "utility"
    comp = CompositeScorer([(ArrayScorer.make([0]), 1.0), (LogLikelihoodScorer(), 1.0)])
    assert comp.needs_logprobs == "chosen"


def test_composite_weights_scales_and_per_kind_fill():
    lp = ArrayScorer.make([math.log(0.9), math.log(0.1), math.nan], kind="logprob", scale=2.0)
    ut = ArrayScorer.make([1.0, math.nan, 3.0], kind="utility")
    out = CompositeScorer([(lp, 0.5), (ut, 3.0)]).score(ctx(), [None] * 3)
    lp_filled = np.array([math.log(0.9), math.log(0.1), math.log(0.5)])
    ut_filled = np.array([1.0, 2.0, 3.0])
    assert out == pytest.approx(0.5 * 2.0 * lp_filled + 3.0 * ut_filled)


def test_composite_all_nan_component_policies():
    nan = ArrayScorer.make([math.nan, math.nan], name="dead")
    ok = ArrayScorer.make([1.0, 2.0])
    with pytest.warns(UserWarning, match="'dead'"):
        out = CompositeScorer([(nan, 1.0), (ok, 1.0)]).score(ctx(), [None] * 2)
    assert list(out) == [1.0, 2.0]
    with pytest.raises(RuntimeError, match="'dead'"):
        CompositeScorer([(nan, 1.0)], on_all_nan="raise").score(ctx(), [None] * 2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        CompositeScorer([(nan, 1.0)], on_all_nan="ignore").score(ctx(), [None] * 2)


def test_composite_rejects_bad_components():
    with pytest.raises(ValueError, match="at least one"):
        CompositeScorer([])
    with pytest.raises(TypeError, match="not a Scorer"):
        CompositeScorer([(object(), 1.0)])
    with pytest.raises(ValueError, match="weight"):
        CompositeScorer([(LogLikelihoodScorer(), math.inf)])
    with pytest.raises(ValueError, match="on_all_nan"):
        CompositeScorer([(LogLikelihoodScorer(), 1.0)], on_all_nan="skip")


def test_composite_validates_and_describes_components():
    j = ScriptedJudge()
    j.num_logprobs = "auto"
    comp = CompositeScorer([(llama_guard_scorer(j), 1.0), (LogLikelihoodScorer(), 0.5)])
    with pytest.raises(ValueError, match="Give the judge an int"):
        comp.validate(ValidationContext(PLuMCoTConfig(), 20, FakeBackend()))
    d = json.loads(json.dumps(comp.to_dict()))
    assert [c["weight"] for c in d["components"]] == [1.0, 0.5]
    assert d["components"][1]["scorer"]["type"] == "LogLikelihoodScorer"


# ===========================================================================
# Scorer base
# ===========================================================================

def test_base_validate_catches_bad_declarations():
    vc = ValidationContext(PLuMCoTConfig(), 20, FakeBackend())
    for attr, val in (("kind", "probability"), ("needs_logprobs", "all"), ("scale", math.nan)):
        s = ArrayScorer.make()
        setattr(s, attr, val)
        with pytest.raises(ValueError):
            s.validate(vc)


@pytest.mark.parametrize("scorer", [LogLikelihoodScorer(reduce="sum"),
                                    SelfCertaintyScorer(metric="max_logprob"),
                                    policy_judge_scorer(SimpleNamespace(tokenizer=None))])
def test_scorer_to_dict_is_json_serializable(scorer):
    d = json.loads(json.dumps(scorer.to_dict()))
    assert d["type"] == type(scorer).__name__ and "scale" in d