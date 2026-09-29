"""plumcot/engine.py: how PLuMCoT sequences the stages.

Tests taking `make_backend` also run on a real model when PLUMCOT_TEST_MODEL
is set (see conftest.py)."""
import json

import numpy as np
import pytest

from conftest import (ArrayScorer, RecordingBackend, ScriptedBackend, cont, dist_from_probs,
                      first_derived_seed, loop_config, prompt_for)
from fake_backend import EOS, HARM, FakeBackend, FakeGuard, FakeTokenizer
from plumcot import (AlwaysTrigger, BlendSelect, Candidates, Configure,
                     EntropyTrigger, Filter, FixedConfigure, Lookahead, LogLikelihoodScorer,
                     NeverTrigger, PLuMCoT, PLuMCoTConfig, RolloutLookahead, SampleAggregator,
                     Select, SelfCertaintyScorer, StageBundle, TopKTopPFilter, Trigger,
                     VerdictJudgeScorer, base_policy_prior, guard_messages)


# ---------------------------------------------------------------------------
# small custom stages
# ---------------------------------------------------------------------------

class FixedFilter(Filter):
    """Always offers the same candidates."""

    def __init__(self, ids, probs=None):
        self.ids = list(ids)
        self.probs = probs or [1.0 / len(ids)] * len(ids)

    def __call__(self, dist, decoding):
        lps = np.log(np.asarray(self.probs, dtype=float))
        return Candidates(np.array(self.ids), lps, base_policy_prior(lps, 1.0))


class PickToken(Select):
    """Chooses `token` if offered, else the first candidate."""

    def __init__(self, token):
        self.token = token

    def __call__(self, candidates, scores, rng):
        ids = list(candidates.token_ids)
        idx = ids.index(self.token) if self.token in ids else 0
        return idx, np.eye(len(ids))[idx]


class ReturnIndex(Select):
    def __init__(self, idx):
        self.idx = idx

    def __call__(self, candidates, scores, rng):
        return self.idx, np.ones(len(candidates.token_ids)) / len(candidates.token_ids)


class FireAt(Trigger):
    """Fires when the response so far has exactly k tokens."""

    def __init__(self, k):
        self.k = k

    def __call__(self, dist, prompt_ids, response_ids):
        return len(response_ids) == self.k, {"k": self.k}


class Choose(Configure):
    """Picks options by a function of the step, recording every call."""

    def __init__(self, options, fn=lambda step, n: 0):
        super().__init__(options)
        self.fn = fn
        self.steps = []

    def choose(self, step):
        self.steps.append(step)
        return self.fn(step, len(self.steps))


def engine(backend=None, *, trigger=None, config=None, stages=None, **bundle):
    if stages is None:
        bundle.setdefault("scorer", LogLikelihoodScorer())
        stages = StageBundle(**bundle)
    return PLuMCoT(backend or FakeBackend(), trigger=trigger or EntropyTrigger(0.3),
                   stages=stages, config=config or loop_config())


def small_real_config(**kw):
    """Cheap enough for a real model: few candidates, short everything."""
    kw.setdefault("max_new_tokens", 12)
    kw.setdefault("chunk_size", 4)
    return loop_config(top_k=4, top_p=0.95, **kw)


def cheap_bundle(**kw):
    kw.setdefault("scorer", LogLikelihoodScorer())
    kw.setdefault("lookahead", RolloutLookahead(max_tokens=4))
    return StageBundle(**kw)


def chunk(ids, finish="length", probs=(0.5, 0.5)):
    """A scripted base-decoding chunk with a distribution at every position."""
    return [[cont(ids, [-0.7] * len(ids), [dist_from_probs(probs) for _ in ids], finish)]]


# ===========================================================================
# construction
# ===========================================================================

def test_constructor_type_checks():
    with pytest.raises(TypeError, match="Trigger"):
        PLuMCoT(FakeBackend(), trigger=object(), stages=StageBundle(LogLikelihoodScorer()))
    with pytest.raises(TypeError, match="StageBundle or a Configure"):
        PLuMCoT(FakeBackend(), trigger=NeverTrigger(), stages=LogLikelihoodScorer())


def test_config_none_uses_documented_defaults():
    e = PLuMCoT(FakeBackend(num_logprobs="auto"), trigger=NeverTrigger(),
                stages=StageBundle(LogLikelihoodScorer()))
    assert e.config == PLuMCoTConfig()
    assert e.backend.num_logprobs == 50


def test_invalid_config_rejected_at_construction():
    with pytest.raises(ValueError, match="chunk_size"):
        engine(config=loop_config(chunk_size=0))


def test_auto_num_logprobs_resolved_in_place_explicit_kept():
    be = FakeBackend(num_logprobs="auto")
    engine(be, config=loop_config(top_k=25))
    assert be.num_logprobs == 25
    be = FakeBackend(num_logprobs=30)
    engine(be)                                       # top_k 10
    assert be.num_logprobs == 30


def test_capacity_errors_and_warnings_at_construction():
    with pytest.raises(ValueError, match="needs 30 logprobs"):
        engine(FakeBackend(num_logprobs=20), config=loop_config(top_k=30))
    with pytest.warns(UserWarning, match="less reliable"):
        engine(FakeBackend(num_logprobs="auto"), config=loop_config(top_k=5))


def test_construction_checks_cap_and_validates_every_stage():
    seen = []

    def spy(obj):
        orig = obj.validate
        obj.validate = lambda ctx: (seen.append(type(obj).__name__), orig(ctx))[1]
        return obj

    b1 = StageBundle(spy(LogLikelihoodScorer()), filter=spy(TopKTopPFilter()),
                     lookahead=spy(RolloutLookahead()), aggregate=spy(SampleAggregator()),
                     select=spy(BlendSelect()))
    b2 = StageBundle(spy(SelfCertaintyScorer()))
    rec = RecordingBackend(FakeBackend())
    PLuMCoT(rec, trigger=spy(EntropyTrigger()), stages=spy(Choose([b1, b2])),
            config=loop_config())
    assert rec.cap_checks == 1
    assert sorted(seen) == sorted(["EntropyTrigger", "Choose", "LogLikelihoodScorer",
                                   "TopKTopPFilter", "RolloutLookahead", "SampleAggregator",
                                   "BlendSelect", "SelfCertaintyScorer"])


def test_temperature_zero_guard_applies_at_construction():
    with pytest.raises(ValueError, match="decoding.temperature=0"):
        engine(config=loop_config(temperature=0.0))
    engine(config=loop_config(temperature=0.0), select=BlendSelect(1.0))


def test_judge_backend_validated_at_construction():
    judge = FakeGuard()
    judge.num_logprobs = "auto"
    with pytest.raises(ValueError, match="Give the judge an int"):
        engine(scorer=VerdictJudgeScorer(judge, guard_messages(), ["safe"], ["unsafe"]))


def test_describe_is_json_serializable():
    e = engine(stages=Choose([StageBundle(LogLikelihoodScorer()),
                              StageBundle(SelfCertaintyScorer(), filter=TopKTopPFilter(top_k=3))]))
    d = json.loads(json.dumps(e.describe()))
    assert d["loop"]["decoding"]["top_k"] == 10
    assert d["trigger"] == {"type": "EntropyTrigger", "threshold": 0.3, "tail": "lumped"}
    assert d["configure"]["type"] == "Choose"
    assert [o["filter"]["top_k"] for o in d["configure"]["options"]] == ["decoding", 3]


# ===========================================================================
# prompt construction
# ===========================================================================

def test_build_prompt_applies_chat_template():
    e = engine()
    ids = e.build_prompt("hello")
    assert ids[-1] == 9                                # fake generation prompt marker
    assert e.build_prompt("hello", chat=False) == FakeTokenizer().encode("hello")


def test_build_prompt_system_message_and_batch_encoding():
    class Tok(FakeTokenizer):
        seen = []

        def apply_chat_template(self, messages, **kw):
            Tok.seen.append(messages)
            return {"input_ids": [4, 5, 6]}            # BatchEncoding-like

    be = FakeBackend()
    be.tokenizer = Tok()
    assert engine(be).build_prompt("hi", system="SYS") == [4, 5, 6]
    assert Tok.seen[-1] == [{"role": "system", "content": "SYS"},
                            {"role": "user", "content": "hi"}]


def test_build_prompt_errors():
    with pytest.raises(ValueError, match="needs chat=True"):
        engine().build_prompt("hi", chat=False, system="SYS")

    class NoTemplate(FakeTokenizer):
        chat_template = None

    be = FakeBackend()
    be.tokenizer = NoTemplate()
    with pytest.raises(ValueError, match="no chat template"):
        engine(be).build_prompt("hi")


# ===========================================================================
# generate(): inputs
# ===========================================================================

def test_generate_needs_a_prompt():
    with pytest.raises(ValueError, match="user_message or prompt_ids"):
        engine().generate()


def test_prompt_ids_path_matches_user_message_path():
    e = engine()
    assert e.generate("hello").response_ids == e.generate(
        prompt_ids=e.build_prompt("hello")).response_ids


def test_user_message_and_context_reach_the_scorer():
    sc = ArrayScorer.make(fn=lambda b: 0.0)
    e = engine(scorer=sc, trigger=FireAt(2), filter=FixedFilter([5, 6]),
               config=loop_config(max_new_tokens=5))
    e.generate("USER", prompt_ids=[1, 2, 3])
    ctx, branches = sc.calls[0]
    assert ctx.user_message == "USER" and ctx.prompt_ids == [1, 2, 3]
    assert len(ctx.response_ids) == 2 and ctx.tokenizer is e.backend.tokenizer


# ===========================================================================
# exactness of chunked decoding
# ===========================================================================

def test_output_is_independent_of_chunk_size():
    """Fully deterministic setup (greedy base decoding, greedy lookahead,
    argmax on score only): chunking must not change anything. Fake backend
    only; on a GPU, prefill/decode numerics can differ across chunkings."""
    outs = []
    for cs in (1, 3, 8, 32):
        e = engine(config=loop_config(temperature=0.0, chunk_size=cs, max_new_tokens=40),
                   select=BlendSelect(1.0, "argmax"))
        r = e.generate(prompt_ids=[1, 2, 3])
        outs.append((r.response_ids, [s.position for s in r.trace]))
    assert outs[0][1] and all(o == outs[0] for o in outs)


def test_never_trigger_is_one_base_decoding_call(make_backend):
    be = make_backend()
    cfg = small_real_config(max_new_tokens=16, chunk_size=16, seed=5)
    prompt = prompt_for(be)
    r = PLuMCoT(be, trigger=NeverTrigger(), stages=cheap_bundle(), config=cfg).generate(
        prompt_ids=prompt)
    d = cfg.decoding
    direct = be.generate([prompt], n=1, max_tokens=16, temperature=d.temperature,
                         top_k=d.top_k, top_p=d.top_p, logprobs="dist",
                         seed=first_derived_seed(5))[0][0]
    assert r.response_ids == direct.token_ids
    assert r.stats["fired"] == 0 and r.trace == [] and r.stats["chunks"] == 1


def test_every_position_is_considered_after_a_triggered_step(make_backend):
    be = make_backend()
    r = PLuMCoT(be, trigger=AlwaysTrigger(), stages=cheap_bundle(),
                config=small_real_config()).generate(prompt_ids=prompt_for(be))
    assert [s.position for s in r.trace] == list(range(len(r.response_ids)))


def test_trigger_sees_the_right_prefix_within_a_chunk():
    """FireAt(k) must fire exactly at response position k, wherever k falls
    inside a chunk."""
    for k in (0, 3, 7, 8, 11):
        r = engine(trigger=FireAt(k), config=loop_config(chunk_size=8, max_new_tokens=15)) \
            .generate(prompt_ids=[1, 2, 3])
        assert [s.position for s in r.trace] == [k]
        assert r.trace[0].trigger_info == {"k": k}


# ===========================================================================
# termination
# ===========================================================================

def test_never_exceeds_max_new_tokens(make_backend):
    be = make_backend()
    for trig in (NeverTrigger(), AlwaysTrigger()):
        r = PLuMCoT(be, trigger=trig, stages=cheap_bundle(),
                    config=small_real_config(max_new_tokens=7, chunk_size=3)).generate(
            prompt_ids=prompt_for(be))
        assert r.stats["tokens"] == len(r.response_ids) <= 7


def test_stops_when_a_stop_token_is_selected():
    e = engine(trigger=AlwaysTrigger(), filter=FixedFilter([EOS, 5]), select=PickToken(EOS))
    r = e.generate(prompt_ids=[1, 2, 3])
    assert r.response_ids == [EOS] and r.text == "" and r.stats["chunks"] == 1


def test_stops_when_a_chunk_ends_with_a_stop_token():
    be = ScriptedBackend([chunk([5, EOS], finish="stop")])
    r = engine(be, trigger=NeverTrigger()).generate(prompt_ids=[1])
    assert r.response_ids == [5, EOS] and len(be.calls) == 1


def test_trigger_at_the_stop_position_replaces_the_stop_token():
    be = ScriptedBackend([chunk([5, EOS], finish="stop"), chunk([6, EOS], finish="stop")])
    r = engine(be, trigger=FireAt(1), filter=FixedFilter([7])).generate(prompt_ids=[1])
    assert r.response_ids == [5, 7, 6, EOS]
    assert be.calls[1]["contexts"] == [[1, 5, 7]]


@pytest.mark.parametrize("bad_chunk, match", [
    ([[cont([], [], [])]], "empty chunk"),
    ([[cont([5, 6], [-1, -1], None)]], "distribution for every position"),
    ([[cont([5, 6], [-1, -1], [dist_from_probs([0.5])])]], "distribution for every position"),
])
def test_bad_chunks_raise(bad_chunk, match):
    with pytest.raises(RuntimeError, match=match):
        engine(ScriptedBackend([bad_chunk]), trigger=NeverTrigger()).generate(prompt_ids=[1])


# ===========================================================================
# stage contract enforcement
# ===========================================================================

def test_scorer_returning_the_wrong_shape_raises():
    e = engine(scorer=ArrayScorer.make([1.0]), trigger=AlwaysTrigger(), filter=FixedFilter([5, 6]))
    with pytest.raises(RuntimeError, match="returned shape"):
        e.generate(prompt_ids=[1])


@pytest.mark.parametrize("idx", [99, -1, True, 1.0])
def test_select_returning_a_bad_index_raises(idx):
    e = engine(trigger=AlwaysTrigger(), filter=FixedFilter([5, 6]), select=ReturnIndex(idx))
    with pytest.raises(ValueError, match="Select returned"):
        e.generate(prompt_ids=[1])


@pytest.mark.parametrize("shape", [[1, 2], [0, 0], [1]])
def test_lookahead_returning_ragged_branches_raises(shape):
    class Ragged(Lookahead):
        def __call__(self, backend, context, candidates, logprobs, seed):
            from plumcot import Branch
            return [[Branch(int(c), 0.0, cont([9], [-1.0])) for _ in range(k)]
                    for c, k in zip(candidates.token_ids, shape)]

    e = engine(trigger=AlwaysTrigger(), filter=FixedFilter([5, 6]), lookahead=Ragged())
    with pytest.raises(RuntimeError, match="same number"):
        e.generate(prompt_ids=[1])


@pytest.mark.parametrize("idx", [2, -1, True, 1.0, None])
def test_configure_returning_a_bad_index_raises(idx):
    opts = [StageBundle(LogLikelihoodScorer()), StageBundle(LogLikelihoodScorer())]
    e = engine(stages=Choose(opts, lambda step, n: idx), trigger=AlwaysTrigger())
    with pytest.raises(ValueError, match="Configure.choose returned"):
        e.generate(prompt_ids=[1])


# ===========================================================================
# single-candidate steps
# ===========================================================================

def test_single_candidate_step_skips_lookahead_and_is_recorded():
    rec = RecordingBackend(FakeBackend())
    sc = ArrayScorer.make(fn=lambda b: 0.0)
    r = engine(rec, scorer=sc, trigger=FireAt(2), filter=FixedFilter([7]),
               config=loop_config(max_new_tokens=6)).generate(prompt_ids=[1])
    assert r.response_ids[2] == 7 and sc.calls == []
    assert len(rec.calls) == r.stats["chunks"]                # no lookahead call
    (s,) = r.trace
    assert (s.searched, s.chosen_index, s.candidate_ids) == (False, 0, [7])
    assert s.scores is s.sample_scores is s.select_probs is s.lookahead_texts is None
    assert (r.stats["fired"], r.stats["single_candidate"], r.stats["searched"]) == (1, 1, 0)


# ===========================================================================
# stats and trace bookkeeping
# ===========================================================================

def test_stats_and_trace_are_consistent(make_backend):
    rec = RecordingBackend(make_backend())
    la = RolloutLookahead(max_tokens=3, num_samples=2, temperature=1.0)
    r = PLuMCoT(rec, trigger=EntropyTrigger(0.0), config=small_real_config(max_new_tokens=10),
                stages=cheap_bundle(lookahead=la, filter=TopKTopPFilter(top_p=0.5))) \
        .generate(prompt_ids=prompt_for(rec))
    chunk_calls = [o for c, o in zip(rec.calls, rec.outputs) if c["logprobs"] == "dist"]
    look_calls = [o for c, o in zip(rec.calls, rec.outputs) if c["logprobs"] == "chosen"]
    st, tr = r.stats, r.trace
    assert st["chunks"] == len(chunk_calls)
    assert st["fired"] == len(tr)
    assert st["searched"] == sum(s.searched for s in tr) == len(look_calls)
    assert st["single_candidate"] == sum(not s.searched for s in tr)
    assert st["branches"] == sum(2 * len(s.candidate_ids) for s in tr if s.searched)
    assert st["lookahead_tokens"] == sum(len(c.token_ids) for o in look_calls
                                         for req in o for c in req)
    assert st["tokens"] == len(r.response_ids)
    tok = rec.tokenizer
    for s in tr:
        n = len(s.candidate_ids)
        assert r.response_ids[s.position] == s.candidate_ids[s.chosen_index]
        assert len(s.candidate_logprobs) == len(s.prior_logprobs) == len(s.candidate_texts) == n
        assert s.candidate_texts == [tok.decode([c]) for c in s.candidate_ids]
        assert np.exp(s.prior_logprobs).sum() == pytest.approx(1.0)
        assert 0.0 <= s.tail_mass <= 1.0 and "norm_entropy" in s.trigger_info
        if s.searched:
            assert sum(s.select_probs) == pytest.approx(1.0) and len(s.scores) == n
            assert [len(x) for x in s.lookahead_texts] == [2] * n
            assert np.shape(s.sample_scores) == (n, 2)
    assert [s.position for s in tr] == sorted({s.position for s in tr})


def test_keep_trace_false_changes_nothing_else(make_backend):
    be = make_backend()
    kw = dict(trigger=EntropyTrigger(0.0), stages=cheap_bundle(), config=small_real_config())
    a = PLuMCoT(be, **kw).generate(prompt_ids=prompt_for(be))
    b = PLuMCoT(be, **kw).generate(prompt_ids=prompt_for(be), keep_trace=False)
    assert a.response_ids == b.response_ids and a.stats == b.stats
    assert a.trace and b.trace == []


def test_full_vocabulary_gives_zero_tail_mass():
    r = engine(FakeBackend(num_logprobs=-1)).generate("hello")
    assert r.trace and all(s.tail_mass == pytest.approx(0.0, abs=1e-9) for s in r.trace)


# ===========================================================================
# seeds
# ===========================================================================

def test_same_seed_same_output(make_backend):
    be = make_backend()
    kw = dict(trigger=EntropyTrigger(0.2), stages=cheap_bundle(),
              config=small_real_config(seed=3))
    a = PLuMCoT(be, **kw).generate(prompt_ids=prompt_for(be))
    b = PLuMCoT(be, **kw).generate(prompt_ids=prompt_for(be))
    assert a.response_ids == b.response_ids


def test_different_seeds_diverge():
    outs = {tuple(engine(config=loop_config(seed=s)).generate("hello").response_ids)
            for s in range(5)}
    assert len(outs) > 1


def test_unseeded_runs_pass_no_seed():
    rec = RecordingBackend(FakeBackend())
    engine(rec, config=loop_config(seed=None, max_new_tokens=20)).generate("hello")
    assert rec.calls and all(c["seed"] is None for c in rec.calls)


def test_seeded_runs_derive_seeds_from_the_main_rng():
    rec = RecordingBackend(FakeBackend())
    engine(rec, trigger=NeverTrigger(), config=loop_config(seed=11)).generate("hello")
    assert rec.calls[0]["seed"] == first_derived_seed(11)


def test_golden_output():
    """Pins the fake-backend output for one configuration, so any change to
    the loop's behavior is deliberate. Update the expected values when a
    behavior change is intended."""
    r = engine(config=loop_config(seed=7, max_new_tokens=30)).generate(prompt_ids=[1, 2, 3])
    assert r.response_ids == [3, 12, 27, 21, 22, 21, 17, 24, 37, 9, 12, 24, 37, 26, 34, 10,
                              5, 17, 35, 1, 6, 18, 21, 2, 7, 16, 37, 18, 35, 23]
    assert r.stats == {"tokens": 30, "chunks": 22, "fired": 22, "searched": 22,
                       "single_candidate": 0, "branches": 165, "lookahead_tokens": 2609}


# ===========================================================================
# Configure inside the loop
# ===========================================================================

def test_configure_runs_only_on_triggered_steps(make_backend):
    be = make_backend()
    conf = Choose([cheap_bundle()])
    r = PLuMCoT(be, trigger=EntropyTrigger(0.2), stages=conf,
                config=small_real_config()).generate(prompt_ids=prompt_for(be))
    assert len(conf.steps) == r.stats["fired"] == len(r.trace)


def test_configure_sees_the_step_context():
    conf = Choose([StageBundle(LogLikelihoodScorer())])
    r = engine(stages=conf).generate(prompt_ids=[1, 2, 3])
    assert r.trace
    for step, rec in zip(conf.steps, r.trace):
        assert step.prompt_ids == [1, 2, 3]
        assert step.response_ids == r.response_ids[:rec.position]
        assert step.dist.tail_mass() == pytest.approx(rec.tail_mass)
        assert step.trigger_info == rec.trigger_info


def test_chosen_option_is_used_and_recorded():
    s0, s1 = ArrayScorer.make(fn=lambda b: 0.0), ArrayScorer.make(fn=lambda b: 0.0)
    opts = [StageBundle(s0, filter=TopKTopPFilter(top_k=3, top_p=1.0)),
            StageBundle(s1, filter=TopKTopPFilter(top_k=6, top_p=1.0))]
    r = engine(stages=Choose(opts, lambda step, n: n % 2), trigger=AlwaysTrigger(),
               config=loop_config(max_new_tokens=10)).generate(prompt_ids=[1])
    by_opt = {o: [s for s in r.trace if s.option == o] for o in (0, 1)}
    assert all(len(s.candidate_ids) == 3 for s in by_opt[0])
    assert all(len(s.candidate_ids) == 6 for s in by_opt[1])
    assert len(s0.calls) == len(by_opt[0]) and len(s1.calls) == len(by_opt[1])
    assert [s.option for s in r.trace] == [n % 2 for n in range(1, len(r.trace) + 1)]


def test_every_option_is_validated_at_construction():
    opts = [StageBundle(LogLikelihoodScorer()),
            StageBundle(LogLikelihoodScorer(), filter=TopKTopPFilter(top_k=30))]
    with pytest.raises(ValueError, match="needs 30"):
        engine(FakeBackend(num_logprobs=20), stages=Choose(opts))
    be = FakeBackend(num_logprobs="auto")
    engine(be, stages=Choose(opts), config=loop_config(top_k=20))
    assert be.num_logprobs == 30


def test_a_bundle_is_wrapped_in_fixed_configure():
    b = StageBundle(LogLikelihoodScorer())
    e = engine(stages=b)
    assert isinstance(e.configure, FixedConfigure) and e.configure.options == [b]


# ===========================================================================
# end-to-end behavior
# ===========================================================================

@pytest.mark.parametrize("scorer, la, agg", [
    (LogLikelihoodScorer(), RolloutLookahead(), SampleAggregator()),
    (SelfCertaintyScorer(), RolloutLookahead(num_samples=2, temperature=1.0),
     SampleAggregator(mode="logmeanexp")),
    (SelfCertaintyScorer(metric="max_logprob"), RolloutLookahead(num_samples=2, temperature=1.0),
     SampleAggregator()),
])
def test_loop_runs_with_intrinsic_scorers(scorer, la, agg):
    r = engine(scorer=scorer, lookahead=la, aggregate=agg,
               config=loop_config(max_new_tokens=60)).generate("hello")
    searched = [s for s in r.trace if s.searched]
    assert 0 < r.stats["tokens"] <= 60 and searched
    for s in searched:
        assert s.trigger_info["norm_entropy"] >= 0.3
        assert len(s.lookahead_texts[0]) == la.num_samples


def test_scale_is_applied_before_select():
    def first(scale):
        e = engine(scorer=LogLikelihoodScorer(scale=scale), trigger=EntropyTrigger(0.0))
        return e.generate("hello").trace[0]

    a, b = first(1.0), first(3.0)
    assert a.candidate_ids == b.candidate_ids
    assert b.scores == pytest.approx([3.0 * s for s in a.scores])


def test_guard_steers_away_from_harm_token():
    guard = FakeGuard()
    base = guard_messages()

    def build(ctx, text):
        guard.last_texts.append(text)
        return base(ctx, text)

    scorer = VerdictJudgeScorer(guard, build, positive=("safe",), negative=("unsafe",),
                                transform="prob")
    r = engine(scorer=scorer, trigger=EntropyTrigger(0.0), select=BlendSelect(mode="argmax"),
               config=loop_config(max_new_tokens=60)).generate("hello")
    searched = [s for s in r.trace if s.searched]
    assert any(HARM in s.candidate_ids for s in searched)
    assert HARM not in [s.candidate_ids[s.chosen_index] for s in searched]
    assert HARM not in r.response_ids


def test_triggered_steps_follow_the_base_temperature():
    """At w = 0, a triggered step samples from the same tempered policy as
    untriggered decoding: its select_probs equal the tempered prior."""
    r = engine(trigger=EntropyTrigger(0.0), select=BlendSelect(0.0),
               config=loop_config(temperature=0.6, max_new_tokens=15)).generate("hello")
    for s in r.trace:
        if s.searched:
            q = np.exp(np.asarray(s.candidate_logprobs) / 0.6)
            assert s.select_probs == pytest.approx(q / q.sum())