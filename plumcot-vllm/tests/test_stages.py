"""plumcot/stages.py: the default Trigger, Filter, Lookahead, aggregator and
Select implementations, plus StageBundle and Configure as units. How the
engine sequences them is in test_engine.py."""
import json
import math
import warnings

import numpy as np
import pytest

from conftest import (ArrayScorer, RecordingBackend, ScriptedBackend, candidates, cont,
                      dist_from_probs, loop_config, uniform_dist)
from fake_backend import EOS, FakeBackend, _logprobs, _raw_logits
from plumcot import (AlwaysTrigger, BlendSelect, Branch, Candidates, Configure, DecodingConfig,
                     EntropyTrigger, FixedConfigure, LogLikelihoodScorer, NeverTrigger,
                     RolloutLookahead, SampleAggregator, StageBundle, TopKTopPFilter,
                     base_policy_prior, calibrate_entropy_threshold)
from plumcot.base import ValidationContext


def vctx(temperature=1.0, top_k=10, top_p=0.9, num_logprobs=20, backend=None):
    return ValidationContext(loop_config(temperature, top_k, top_p), num_logprobs, backend)


# ===========================================================================
# Stage 1: Trigger
# ===========================================================================

def test_entropy_trigger_threshold_is_inclusive():
    d = dist_from_probs([0.4, 0.3, 0.2, 0.1])
    h = d.normalized_entropy()
    assert EntropyTrigger(threshold=h)(d, [], []) == (True, {"norm_entropy": h})
    fired, info = EntropyTrigger(threshold=h + 1e-9)(d, [], [])
    assert not fired and info["norm_entropy"] == h


def test_entropy_trigger_uses_its_tail_mode():
    d = dist_from_probs([0.2, 0.1])            # heavy tail: modes disagree
    lo = EntropyTrigger(tail="lumped")(d, [], [])[1]["norm_entropy"]
    hi = EntropyTrigger(tail="uniform")(d, [], [])[1]["norm_entropy"]
    assert lo == pytest.approx(d.normalized_entropy("lumped")) and hi > lo


def test_entropy_trigger_is_independent_of_decoding_temperature():
    """The trigger sees only the raw distribution; nothing about decoding."""
    d = uniform_dist(10)
    t = EntropyTrigger(0.2)
    assert t(d, [], []) == t(d, [1, 2], [3])
    assert t(d, [], [])[1]["norm_entropy"] == pytest.approx(d.normalized_entropy())


@pytest.mark.parametrize("kw, match", [(dict(threshold=math.nan), "threshold"),
                                       (dict(threshold=math.inf), "threshold"),
                                       (dict(tail="lump"), "tail")])
def test_entropy_trigger_rejects_bad_settings(kw, match):
    with pytest.raises(ValueError, match=match):
        EntropyTrigger(**kw)


def test_never_and_always_triggers():
    d = uniform_dist(10)
    assert NeverTrigger()(d, [], []) == (False, {})
    assert AlwaysTrigger()(d, [], []) == (True, {})


# -- calibration ------------------------------------------------------------------------

def test_calibrate_hits_the_target_rate():
    thr, hs = calibrate_entropy_threshold(
        FakeBackend(), [[1, 2, 3]] * 4, target_rate=0.2, max_tokens=50,
        decoding=DecodingConfig(1.0, 10, 0.9))
    assert np.mean(hs >= thr) == pytest.approx(0.2, abs=0.05)


def test_calibrate_decodes_with_the_given_policy():
    rec = RecordingBackend(FakeBackend())
    calibrate_entropy_threshold(rec, [[1, 2]], 0.1, decoding=DecodingConfig(0.7, 12, 0.8),
                                max_tokens=5, seed=4)
    call = rec.calls[0]
    assert (call["temperature"], call["top_k"], call["top_p"]) == (0.7, 12, 0.8)
    assert call["logprobs"] == "dist" and call["seed"] == 4 and call["max_tokens"] == 5


def test_calibrate_respects_tail_mode():
    kw = dict(decoding=DecodingConfig(1.0, 10, 0.9), max_tokens=20, seed=0)
    _, lo = calibrate_entropy_threshold(FakeBackend(), [[1, 2]], 0.1, tail="lumped", **kw)
    _, hi = calibrate_entropy_threshold(FakeBackend(), [[1, 2]], 0.1, tail="uniform", **kw)
    assert (hi >= lo - 1e-12).all() and (hi > lo).any()


@pytest.mark.parametrize("rate", [0.0, 1.0, -0.1])
def test_calibrate_rejects_bad_target_rate(rate):
    with pytest.raises(ValueError, match="target_rate"):
        calibrate_entropy_threshold(FakeBackend(), [[1]], rate, decoding=DecodingConfig())


def test_calibrate_rejects_unresolved_num_logprobs():
    with pytest.raises(ValueError, match="Resolve num_logprobs"):
        calibrate_entropy_threshold(FakeBackend(num_logprobs="auto"), [[1]], 0.1,
                                    decoding=DecodingConfig())


def test_calibrate_rejects_missing_or_empty_output():
    none = ScriptedBackend([[[cont([5])]]])                   # dists=None
    with pytest.raises(RuntimeError, match="no distributions"):
        calibrate_entropy_threshold(none, [[1]], 0.1, decoding=DecodingConfig())
    empty = ScriptedBackend([[[cont([], dists=[])]]])
    with pytest.raises(RuntimeError, match="No tokens"):
        calibrate_entropy_threshold(empty, [[1]], 0.1, decoding=DecodingConfig())


# ===========================================================================
# Stage 2: Filter
# ===========================================================================

@pytest.fixture
def fake_dist():
    """A full-vocabulary next-token distribution from the fake model."""
    lps = _logprobs(_raw_logits([1, 2, 3]))
    return FakeBackend(num_logprobs=-1)._dist(lps, int(np.argmax(lps))), lps


def test_filter_sentinels_follow_decoding():
    f = TopKTopPFilter()
    assert f.resolve(DecodingConfig(1.0, 17, 0.8)) == (17, 0.8)
    assert f.resolve(DecodingConfig(1.0, None, 0.8)) == (None, 0.8)
    assert TopKTopPFilter(top_k=5, top_p=0.5).resolve(DecodingConfig(1.0, 17, 0.8)) == (5, 0.5)
    assert TopKTopPFilter(top_k=None).resolve(DecodingConfig(1.0, 17, 0.8)) == (None, 0.8)


@pytest.mark.parametrize("t", [0.5, 0.7, 1.0, 1.5])
def test_filter_pool_is_the_base_policy_support(fake_dist, t):
    """The same pool the fake backend (like vLLM) samples from at temperature t."""
    d, lps = fake_dist
    c = TopKTopPFilter()(d, DecodingConfig(t, 10, 0.9))
    order = np.argsort(-lps)[:10]
    p = np.exp(lps[order] / t)
    p /= p.sum()
    n = int(np.searchsorted(np.cumsum(p), 0.9 - 1e-9)) + 1
    assert list(c.token_ids) == list(order[:n])
    assert c.logprobs == pytest.approx(lps[order[:n]])        # raw, not tempered


@pytest.mark.parametrize("t", [0.5, 1.0, 2.0])
def test_filter_prior_is_the_tempered_policy_over_the_pool(fake_dist, t):
    d, _ = fake_dist
    c = TopKTopPFilter()(d, DecodingConfig(t, 10, 0.9))
    q = np.exp(c.logprobs / t)
    assert np.exp(c.prior_logprobs) == pytest.approx(q / q.sum())


def test_filter_at_temperature_zero(fake_dist):
    """Greedy base decoding: the pool comes from the raw distribution (else it
    would always be a single token) and the prior is a point mass."""
    d, _ = fake_dist
    c0 = TopKTopPFilter()(d, DecodingConfig(0.0, 10, 0.9))
    c1 = TopKTopPFilter()(d, DecodingConfig(1.0, 10, 0.9))
    assert list(c0.token_ids) == list(c1.token_ids) and len(c0.token_ids) > 1
    assert c0.prior_logprobs[0] == 0.0 and np.isneginf(c0.prior_logprobs[1:]).all()


def test_filter_min_candidates(fake_dist):
    d, _ = fake_dist
    c = TopKTopPFilter(top_p=0.01, min_candidates=3)(d, DecodingConfig(1.0, 10, 0.9))
    assert len(c.token_ids) == 3


def test_filter_logprob_requirement_is_its_resolved_k():
    assert TopKTopPFilter().required_logprobs(loop_config(top_k=23)) == 23
    assert TopKTopPFilter(top_k=7).required_logprobs(loop_config(top_k=23)) == 7
    assert TopKTopPFilter().required_logprobs(loop_config(top_k=None)) is None


def test_filter_warns_when_top_k_disabled_on_a_slice():
    with pytest.warns(UserWarning, match="top-k disabled"):
        TopKTopPFilter().validate(vctx(top_k=None, num_logprobs=20))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        TopKTopPFilter().validate(vctx(top_k=None, num_logprobs=-1))   # full vocab: fine


@pytest.mark.parametrize("kw", [dict(top_k=0), dict(top_k=True), dict(top_k=2.5),
                                dict(top_k="auto"), dict(top_p=0.0), dict(top_p=1.1),
                                dict(top_p="x"), dict(min_candidates=0)])
def test_filter_rejects_bad_settings(kw):
    with pytest.raises(ValueError):
        TopKTopPFilter(**kw)


def test_candidates_validation():
    with pytest.raises(ValueError, match="at least one"):
        Candidates(np.array([], dtype=int), np.array([]), np.array([]))
    with pytest.raises(ValueError, match="same length"):
        Candidates(np.array([1, 2]), np.array([-1.0]), np.array([-1.0, -2.0]))


def test_base_policy_prior():
    lps = np.log([0.5, 0.3, 0.2])
    assert np.exp(base_policy_prior(lps, 1.0)) == pytest.approx([0.5, 0.3, 0.2])
    q = np.array([0.5, 0.3, 0.2]) ** 2
    assert np.exp(base_policy_prior(lps, 0.5)) == pytest.approx(q / q.sum())
    assert list(base_policy_prior(lps, 0.0)) == [0.0, -np.inf, -np.inf]


# ===========================================================================
# Stage 3: Lookahead
# ===========================================================================

def test_lookahead_plumbing():
    rec = RecordingBackend(FakeBackend())
    la = RolloutLookahead(max_tokens=5, num_samples=3, temperature=0.8, top_k=7, top_p=0.6,
                          stop=["\n\n"])
    c = candidates([0.6, 0.4], ids=[11, 12])
    branches = la(rec, [1, 2, 3], c, "chosen", 99)
    call = rec.calls[0]
    assert call["contexts"] == [[1, 2, 3, 11], [1, 2, 3, 12]]
    assert (call["n"], call["max_tokens"], call["temperature"]) == (3, 5, 0.8)
    assert (call["top_k"], call["top_p"], call["stop"]) == (7, 0.6, ("\n\n",))
    assert call["logprobs"] == "chosen" and call["seed"] == 99
    assert [len(bs) for bs in branches] == [3, 3]
    assert [b.candidate_id for b in branches[1]] == [12, 12, 12]
    assert branches[0][0].candidate_logprob == pytest.approx(math.log(0.6))
    assert all(b.continuation.token_logprobs for bs in branches for b in bs)


@pytest.mark.parametrize("mode, lps, dists", [("none", None, None), ("chosen", [], None),
                                              ("dist", [], [])])
def test_stop_candidates_get_empty_continuations(mode, lps, dists):
    rec = RecordingBackend(FakeBackend())
    c = candidates([0.5, 0.5], ids=[EOS, 5])
    branches = RolloutLookahead(num_samples=2, temperature=1.0)(rec, [1], c, mode, 0)
    assert rec.calls[0]["contexts"] == [[1, 5]]              # no rollout for the stop token
    stop = branches[0]
    assert len(stop) == 2 and stop[0] is not stop[1]
    for b in stop:
        assert b.candidate_id == EOS and b.continuation.token_ids == []
        assert b.continuation.finish_reason == "candidate_is_stop"
        assert b.continuation.token_logprobs == lps and b.continuation.dists == dists
    assert branches[1][0].continuation.token_ids


def test_all_stop_candidates_skip_the_backend():
    be = ScriptedBackend(stop_token_ids=(0, 1))
    branches = RolloutLookahead()(be, [3], candidates([0.5, 0.5], ids=[0, 1]), "none", 0)
    assert be.calls == [] and len(branches) == 2


@pytest.mark.parametrize("result", [
    [],                                          # missing a candidate
    [[cont([5])], [cont([6])]],                  # too many candidates
    [[cont([5]), cont([6])]],                    # too many samples
])
def test_lookahead_rejects_wrong_rollout_counts(result):
    be = ScriptedBackend([result])
    with pytest.raises(RuntimeError, match="wrong number of rollouts"):
        RolloutLookahead()(be, [3], candidates([1.0], ids=[5]), "none", 0)


def test_greedy_lookahead_forbids_multiple_samples():
    with pytest.raises(ValueError, match="identical"):
        RolloutLookahead(num_samples=2, temperature=0.0)
    RolloutLookahead(num_samples=2, temperature=0.5)


@pytest.mark.parametrize("kw", [dict(max_tokens=0), dict(max_tokens=1.5), dict(num_samples=0),
                                dict(temperature=-1.0), dict(temperature=math.nan),
                                dict(top_k=0), dict(top_p=0.0), dict(top_p=2.0)])
def test_lookahead_rejects_bad_settings(kw):
    with pytest.raises(ValueError):
        RolloutLookahead(**kw)


def test_lookahead_stop_strings_become_a_tuple():
    assert RolloutLookahead(stop=["a", "b"]).stop == ("a", "b")


# ===========================================================================
# Stage 4 (second half): SampleAggregator
# ===========================================================================

def test_default_mode_follows_scorer_kind():
    agg = SampleAggregator()
    raw = np.log(np.array([[0.9, 0.1], [0.5, 0.5]]))
    s, _ = agg(raw, ArrayScorer.make(kind="logprob"))
    assert s == pytest.approx(np.log([0.5, 0.5]))            # log of the mean probability
    s, _ = agg(raw, ArrayScorer.make(kind="utility"))
    assert s == pytest.approx(raw.mean(axis=1))


@pytest.mark.parametrize("mode, expect", [("mean", [5 / 3, 2.0]), ("max", [4.0, 3.0]),
                                          ("min", [0.0, 1.0])])
def test_explicit_modes(mode, expect):
    raw = np.array([[0.0, 1.0, 4.0], [1.0, 2.0, 3.0]])
    s, _ = SampleAggregator(mode=mode)(raw, ArrayScorer.make(kind="logprob"))
    assert s == pytest.approx(expect)


def test_logmeanexp_temperature_limits():
    raw = np.array([[0.0, 1.0, 4.0]])
    sc = ArrayScorer.make()

    def agg(tau):
        return float(SampleAggregator(mode="logmeanexp", temperature=tau)(raw, sc)[0][0])

    assert agg(0.02) == pytest.approx(4.0, abs=0.05)          # -> max
    assert agg(-0.02) == pytest.approx(0.0, abs=0.05)         # -> min
    assert agg(1e4) == pytest.approx(raw.mean(), abs=1e-3)    # -> mean
    assert agg(-1e4) == pytest.approx(raw.mean(), abs=1e-3)
    assert agg(-1.0) < raw.mean() < agg(1.0)
    assert agg(1.0) == pytest.approx(math.log(np.exp(raw).mean()))


def test_scale_is_applied_after_filling_in_the_scorers_units():
    """Fill in log-probability space, then scale. Filling after scaling would
    average 2·log p values instead (log 0.41 here instead of 2·log 0.5)."""
    raw = np.array([[math.log(0.9)], [math.log(0.1)], [math.nan]])
    s, x = SampleAggregator()(raw, ArrayScorer.make(kind="logprob", scale=2.0))
    assert x[2, 0] == pytest.approx(2 * math.log(0.5))
    assert x[:2, 0] == pytest.approx(2 * raw[:2, 0])


def test_nan_is_filled_from_all_candidates_at_the_step():
    raw = np.array([[1.0, math.nan], [3.0, 5.0]])
    _, x = SampleAggregator()(raw, ArrayScorer.make(kind="utility"))
    assert x[0, 1] == pytest.approx(3.0)                      # mean of 1, 3, 5


def test_on_all_nan_policies():
    raw = np.full((2, 1), math.nan)
    sc = ArrayScorer.make()
    with pytest.warns(UserWarning, match="every branch scored NaN"):
        s, _ = SampleAggregator()(raw, sc)
    assert list(s) == [0.0, 0.0]
    with pytest.raises(RuntimeError, match="every branch scored NaN"):
        SampleAggregator(on_all_nan="raise")(raw, sc)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert list(SampleAggregator(on_all_nan="ignore")(raw, sc)[0]) == [0.0, 0.0]


@pytest.mark.parametrize("kw", [dict(mode="avg"), dict(temperature=0.0),
                                dict(temperature=math.inf), dict(on_all_nan="skip")])
def test_aggregator_rejects_bad_settings(kw):
    with pytest.raises(ValueError):
        SampleAggregator(**kw)


# ===========================================================================
# Stage 5: BlendSelect
# ===========================================================================

def select(w, prior_probs, scores, mode="sample", seed=0):
    return BlendSelect(judge_weight=w, mode=mode)(
        candidates(prior_probs), np.asarray(scores, dtype=float), np.random.default_rng(seed))


def test_select_prior_and_score_combine():
    # candidate 2 judged 4x less "ok"; w = 0.5 is the sqrt of the posterior [0.8, 0.2]
    _, probs = select(0.5, [0.5, 0.5], [0.0, math.log(0.25)])
    assert probs == pytest.approx([2 / 3, 1 / 3])


def test_select_endpoints():
    prior, scores = [0.7, 0.2, 0.1], np.array([0.0, 1.0, 3.0])
    assert select(0.0, prior, scores)[1] == pytest.approx(prior)            # base decoding
    e = np.exp(scores)
    assert select(1.0, prior, scores)[1] == pytest.approx(e / e.sum())      # score only


def test_select_is_tempered_bayes():
    """p^(1-w) V^w = [p V^k]^(1-w) with k = w / (1 - w)."""
    rng = np.random.default_rng(3)
    p = rng.dirichlet(np.ones(6))
    v = rng.uniform(0.05, 1.0, size=6)
    for w in (0.25, 0.5, 0.8):
        post = (p * v ** (w / (1 - w))) ** (1 - w)
        assert select(w, p, np.log(v))[1] == pytest.approx(post / post.sum())
    assert select(0.5, p, np.log(v), mode="argmax")[0] == int(np.argmax(p * v))   # MAP


def test_w0_is_base_decoding_at_any_temperature(fake_dist):
    d, _ = fake_dist
    for t in (0.5, 1.0, 1.7):
        c = TopKTopPFilter()(d, DecodingConfig(t, 10, 0.9))
        _, probs = BlendSelect(0.0)(c, np.zeros(len(c.token_ids)), np.random.default_rng(0))
        q = np.exp(c.logprobs / t)
        assert probs == pytest.approx(q / q.sum())


# -- the prior-vs-score trade-off ------------------------------------------------------------

def test_likely_token_beats_better_scored_rare_token_until_w_is_high():
    prior, scores = [0.9, 0.1], [0.0, 1.0]
    assert select(0.5, prior, scores, "argmax")[0] == 0
    assert select(1.0, prior, scores, "argmax")[0] == 1


def test_argmax_flips_at_the_analytic_crossover():
    prior, scores = np.array([0.9, 0.1]), np.array([0.0, 3.0])
    dp, ds = math.log(prior[0] / prior[1]), scores[0] - scores[1]
    w_star = dp / (dp - ds)                 # (1-w)·dp + w·ds = 0
    assert select(w_star - 0.01, prior, scores, "argmax")[0] == 0
    assert select(w_star + 0.01, prior, scores, "argmax")[0] == 1
    assert select(w_star, prior, scores)[1] == pytest.approx([0.5, 0.5])


def test_probability_of_better_scored_token_rises_with_w():
    ws = np.linspace(0, 1, 11)
    p_hi = [select(w, [0.8, 0.15, 0.05], [0.0, 0.5, 2.0])[1][2] for w in ws]
    assert np.all(np.diff(p_hi) > 0)


def test_score_shift_invariance_and_scale_sensitivity():
    prior, scores = [0.5, 0.3, 0.2], np.array([0.1, -1.0, 2.0])
    for w in (0.3, 0.5, 1.0):
        assert select(w, prior, scores + 7.5)[1] == pytest.approx(select(w, prior, scores)[1])
    assert not np.allclose(select(0.5, prior, 2 * scores)[1], select(0.5, prior, scores)[1])


def test_prior_is_used_up_to_a_constant():
    c = candidates([0.5, 0.3, 0.2])
    shifted = Candidates(c.token_ids, c.logprobs, c.prior_logprobs - 4.0)
    rng = np.random.default_rng
    s = np.array([0.0, 1.0, -1.0])
    assert BlendSelect(0.4)(shifted, s, rng(0))[1] == pytest.approx(
        BlendSelect(0.4)(c, s, rng(0))[1])


# -- infinities and invalid scores -------------------------------------------------------------

def test_neg_inf_score_rules_a_candidate_out():
    _, probs = select(0.5, [0.5, 0.3, 0.2], [0.0, -math.inf, 0.0])
    assert probs[1] == 0.0 and probs.sum() == pytest.approx(1.0)


def test_w1_ignores_a_neg_inf_prior_without_nan():
    c = candidates([0.6, 0.4], temperature=0.0)             # prior [0, -inf]
    idx, probs = BlendSelect(1.0, "argmax")(c, np.array([0.0, 1.0]), np.random.default_rng(0))
    assert idx == 1 and np.isfinite(probs).all()


def test_w0_ignores_scores_entirely():
    _, probs = select(0.0, [0.7, 0.3], [math.nan, math.inf])
    assert probs == pytest.approx([0.7, 0.3])


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_nan_or_pos_inf_scores_raise(bad):
    with pytest.raises(ValueError, match="NaN or \\+inf"):
        select(0.5, [0.5, 0.5], [0.0, bad])


def test_all_neg_inf_utilities_raise():
    with pytest.raises(ValueError, match="nothing to choose"):
        select(0.5, [0.5, 0.5], [-math.inf, -math.inf])


# -- sampling ---------------------------------------------------------------------------------

def test_sampling_frequencies_match_probabilities():
    c, s = candidates([0.5, 0.3, 0.2]), np.array([0.0, 0.7, 1.2])
    rng = np.random.default_rng(0)
    sel = BlendSelect(0.5)
    probs = sel(c, s, rng)[1]
    n = 20000
    counts = np.bincount([sel(c, s, rng)[0] for _ in range(n)], minlength=3)
    assert counts / n == pytest.approx(probs, abs=4 * math.sqrt(0.25 / n))


def test_argmax_ties_go_to_the_first_candidate():
    assert select(0.5, [0.5, 0.5], [1.0, 1.0], "argmax")[0] == 0


def test_one_step_beam_with_summed_loglik():
    """reduce="sum", w = 0.5, argmax: maximizes log p(c) + log p(lookahead | c)."""
    c = candidates([0.5, 0.3, 0.2], ids=[10, 11, 12])
    look = {10: [-1.5, -1.5], 11: [-0.5, -0.5], 12: [-0.4, -0.4]}   # sums -3, -1, -0.8
    branches = [Branch(int(t), float(lp), cont([1] * len(look[int(t)]), look[int(t)]))
                for t, lp in zip(c.token_ids, c.logprobs)]
    ll = LogLikelihoodScorer(reduce="sum")
    scores, _ = SampleAggregator()(ll.score(None, branches).reshape(3, 1), ll)
    idx, _ = BlendSelect(0.5, "argmax")(c, scores, np.random.default_rng(0))
    joint = c.logprobs + np.array([sum(look[int(t)]) for t in c.token_ids])
    assert idx == int(np.argmax(joint)) == 1
    assert idx != int(np.argmax(c.logprobs)) and idx != int(np.argmax(scores))


# -- settings and the temperature-0 guard -----------------------------------------------------

@pytest.mark.parametrize("kw", [dict(judge_weight=-0.1), dict(judge_weight=1.5),
                                dict(mode="arg_max")])
def test_select_rejects_bad_settings(kw):
    with pytest.raises(ValueError):
        BlendSelect(**kw)


def test_temperature_zero_guard():
    with pytest.raises(ValueError, match="decoding.temperature=0"):
        BlendSelect(0.5).validate(vctx(temperature=0.0))
    BlendSelect(1.0).validate(vctx(temperature=0.0))
    BlendSelect(0.5).validate(vctx(temperature=0.3))


# ===========================================================================
# StageBundle and Configure
# ===========================================================================

def test_bundle_defaults_and_stage_list():
    sc = LogLikelihoodScorer()
    b = StageBundle(sc)
    assert isinstance(b.filter, TopKTopPFilter) and isinstance(b.lookahead, RolloutLookahead)
    assert isinstance(b.aggregate, SampleAggregator) and isinstance(b.select, BlendSelect)
    assert b.stages() == [b.filter, b.lookahead, sc, b.aggregate, b.select]
    assert StageBundle(sc).filter is not b.filter             # fresh default objects


@pytest.mark.parametrize("field", ["filter", "lookahead", "aggregate", "select"])
def test_bundle_type_checks(field):
    with pytest.raises(TypeError, match=field):
        StageBundle(LogLikelihoodScorer(), **{field: object()})
    with pytest.raises(TypeError, match="scorer"):
        StageBundle(object())


def test_bundle_to_dict_is_json_serializable():
    d = json.loads(json.dumps(StageBundle(LogLikelihoodScorer(reduce="sum")).to_dict()))
    assert d["filter"] == {"type": "TopKTopPFilter", "top_k": "decoding", "top_p": "decoding",
                           "min_candidates": 1}
    assert d["scorer"]["reduce"] == "sum" and d["select"]["judge_weight"] == 0.5


def test_configure_validation():
    with pytest.raises(ValueError, match="at least one"):
        Configure([])
    with pytest.raises(TypeError, match="StageBundle"):
        Configure([object()])
    with pytest.raises(NotImplementedError):
        Configure([StageBundle(LogLikelihoodScorer())]).choose(None)


def test_fixed_configure():
    b = StageBundle(LogLikelihoodScorer())
    fc = FixedConfigure(b)
    assert fc.options == [b] and fc.choose(None) == 0
    assert fc.to_dict()["type"] == "FixedConfigure" and len(fc.to_dict()["options"]) == 1