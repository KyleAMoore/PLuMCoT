import math
import sys
from pathlib import Path
 
import numpy as np
import pytest
 
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
 
from fake_backend import EOS, HARM, V, FakeBackend, FakeGuard  # noqa: E402
from plumcot import (AUTO, FULL_VOCAB, BlendSelect, Branch, Candidates,  # noqa: E402
                     CompositeScorer, Continuation, DecodingConfig, EntropyTrigger,
                     LogLikelihoodScorer, NeverTrigger, PLuMCoT, PLuMCoTConfig,
                     RolloutLookahead, SampleAggregator, ScoreContext, SelfCertaintyScorer,
                     StageBundle, TokenDist, TopKTopPFilter, VerdictJudgeScorer,
                     calibrate_entropy_threshold, guard_messages, policy_judge_scorer,
                     resolve_num_logprobs, top_k_top_p)
 
 
def _cfg(**kw):
    """Loop config used throughout: top-k 10, top-p 0.9, temperature 1."""
    cfg = PLuMCoTConfig(decoding=DecodingConfig(temperature=1.0, top_k=10, top_p=0.9),
                        max_new_tokens=60, seed=1, chunk_size=8)
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg
 
 
def _engine(scorer=None, *, backend=None, trigger=None, config=None, **bundle):
    return PLuMCoT(backend or FakeBackend(), trigger=trigger or EntropyTrigger(0.3),
                   stages=StageBundle(scorer or LogLikelihoodScorer(), **bundle),
                   config=config or _cfg())
 
 
# -- dist ----------------------------------------------------------------------
 
def test_entropy_bounds_and_uniform():
    full = TokenDist.from_items(((i, -math.log(V)) for i in range(V)), V)
    assert full.normalized_entropy() == pytest.approx(1.0)
    top = TokenDist.from_items(((i, -math.log(V)) for i in range(10)), V)
    lo, hi = top.entropy("lumped"), top.entropy("uniform")
    assert lo < math.log(V) and hi == pytest.approx(math.log(V))
 
 
def test_filter_top_k_then_top_p():
    lps = np.log(np.array([0.5, 0.3, 0.1, 0.05, 0.05]))
    d = TokenDist.from_items(zip(range(5), lps), 10)
    ids, _ = top_k_top_p(d, top_k=4, top_p=0.8)
    assert list(ids) == [0, 1]           # 0.5/0.95 + 0.3/0.95 ≥ 0.8
    ids, _ = top_k_top_p(d, top_k=2, top_p=1.0)
    assert list(ids) == [0, 1]
    ids, _ = top_k_top_p(d, top_k=5, top_p=0.1, min_candidates=3)
    assert list(ids) == [0, 1, 2]
 
 
# -- select ----------------------------------------------------------------------
 
def _select(w, lps, scores, mode="sample"):
    lps = np.asarray(lps, dtype=np.float64)
    cands = Candidates(np.arange(len(lps)), lps, lps)   # prior = raw at temperature 1
    return BlendSelect(judge_weight=w, mode=mode)(cands, np.asarray(scores),
                                                  np.random.default_rng(0))
 
 
def test_select_prior_and_score_combine():
    lps = np.log([0.5, 0.5])
    scores = np.array([0.0, math.log(0.25)])  # candidate 2 judged 4x less "ok"
    # w = 0.5: sqrt of the Bayesian posterior [0.8, 0.2], i.e. temperature 2.
    _, probs = _select(0.5, lps, scores)
    assert probs == pytest.approx([2 / 3, 1 / 3])
 
 
def test_select_endpoints():
    lps = np.log([0.7, 0.2, 0.1])
    scores = np.array([0.0, 1.0, 3.0])
    _, probs = _select(0.0, lps, scores)
    assert probs == pytest.approx([0.7, 0.2, 0.1])       # plain decoding
    _, probs = _select(1.0, lps, scores)
    e = np.exp(scores)
    assert probs == pytest.approx(e / e.sum())           # score only, uniform prior
 
 
def test_select_is_tempered_bayes():
    """p^(1-w) V^w = [p V^k]^(1-w) with k = w / (1 - w)."""
    rng = np.random.default_rng(3)
    p = rng.dirichlet(np.ones(6))
    v = rng.uniform(0.05, 1.0, size=6)
    for w in (0.25, 0.5, 0.8):
        k = w / (1 - w)
        post = (p * v ** k) ** (1 - w)
        _, probs = _select(w, np.log(p), np.log(v))
        assert probs == pytest.approx(post / post.sum())
    # w = 0.5 in argmax mode is exact Bayesian MAP.
    idx, _ = _select(0.5, np.log(p), np.log(v), mode="argmax")
    assert idx == int(np.argmax(p * v))
 
 
# -- aggregation -------------------------------------------------------------------
 
def test_aggregate_defaults_follow_scorer_kind():
    raw = np.log(np.array([[0.9, 0.1], [0.5, 0.5]]))
    judge = VerdictJudgeScorer(FakeGuard(), guard_messages(), ("safe",), ("unsafe",))
    assert judge.kind == "logprob"
    agg = SampleAggregator()
    assert agg.mode_for(judge) == "logmeanexp"
    # log of the mean probability: both candidates have V = 0.5.
    scores, _ = agg(raw, judge)
    assert scores == pytest.approx(np.log([0.5, 0.5]))
 
    ll = LogLikelihoodScorer()
    assert agg.mode_for(ll) == "mean"
    assert agg(raw, ll)[0] == pytest.approx(raw.mean(axis=1))
 
    assert SampleAggregator(mode="min").mode_for(judge) == "min"   # explicit setting wins
 
    prob_judge = VerdictJudgeScorer(FakeGuard(), guard_messages(), ("safe",), ("unsafe",),
                                    transform="prob")
    assert prob_judge.kind == "utility"
    assert CompositeScorer([(judge, 1.0)]).kind == "utility"
 
 
def test_negative_aggregate_temperature_is_soft_min():
    raw = np.array([[0.0, 1.0, 4.0]])
    vals = {}
    for tau in (-0.05, -1.0, -1e3, 1.0):
        agg = SampleAggregator(mode="logmeanexp", temperature=tau)
        vals[tau] = float(agg(raw, LogLikelihoodScorer())[0][0])
    assert vals[-0.05] == pytest.approx(0.0, abs=0.1)          # near min
    assert 0.0 < vals[-1.0] < raw.mean() < vals[1.0]           # pessimistic < mean < optimistic
    assert vals[-1e3] == pytest.approx(raw.mean(), abs=1e-2)   # |tau| -> inf: mean
 
 
def test_bad_select_and_aggregate_settings_rejected():
    for w in (-0.1, 1.5):
        with pytest.raises(ValueError, match="judge_weight"):
            BlendSelect(judge_weight=w)
    with pytest.raises(ValueError, match="temperature"):
        SampleAggregator(temperature=0.0)
 
 
def test_scale_is_applied_before_select():
    backend = FakeBackend()
    cfg = _cfg()
 
    def run(scale):
        eng = _engine(LogLikelihoodScorer(scale=scale), backend=backend,
                      trigger=EntropyTrigger(0.0), config=cfg)
        return eng.generate("hello").trace[0]
 
    a, b = run(1.0), run(3.0)
    assert a.candidate_ids == b.candidate_ids
    assert b.scores == pytest.approx([3.0 * s for s in a.scores])
 
 
def test_composite_applies_component_scales():
    backend = FakeBackend()
    ctx = ScoreContext("hi", [1, 2], [20, 21], backend.tokenizer)
    branches = [Branch(24, -1.0, Continuation([22, 23], [-0.5, -1.5]))]
    base = LogLikelihoodScorer().score(ctx, branches)
    comp = CompositeScorer([(LogLikelihoodScorer(scale=2.0), 0.5)])
    assert comp.score(ctx, branches) == pytest.approx(base)   # 0.5 * 2.0 = 1
 
 
# -- judge parsing -----------------------------------------------------------------
 
def _guard_scorer():
    guard = FakeGuard()
    base = guard_messages()
 
    def build(ctx, text):
        guard.last_texts.append(text)
        return base(ctx, text)
 
    scorer = VerdictJudgeScorer(guard, build, positive=("safe",), negative=("unsafe",),
                                transform="prob")
    return guard, scorer
 
 
def test_verdict_judge_scores():
    backend = FakeBackend()
    _, scorer = _guard_scorer()
    ctx = ScoreContext("hi", [1, 2], [20, 21], backend.tokenizer)
    branches = [Branch(HARM, -1.0, Continuation([22, 23])),
                Branch(24, -1.0, Continuation([22, 23]))]
    s = scorer.score(ctx, branches)
    assert s[0] == pytest.approx(0.05) and s[1] == pytest.approx(0.95)
    assert scorer.n_failures == 0
 
 
def test_label_prefix_matching():
    backend = FakeBackend()
    sc = policy_judge_scorer(backend)
    assert sc._label_of(3) == "yes" and sc._label_of(4) == "no"
    assert sc._label_of(25) is None
 
 
# -- full loop ----------------------------------------------------------------------
 
@pytest.mark.parametrize("scorer_fn", [
    LogLikelihoodScorer,
    SelfCertaintyScorer,
    lambda: SelfCertaintyScorer(metric="max_logprob"),
])
def test_loop_runs_with_intrinsic_scorers(scorer_fn):
    eng = _engine(scorer_fn(), lookahead=RolloutLookahead(num_samples=2, temperature=1.0),
                  aggregate=SampleAggregator(mode="logmeanexp"))
    res = eng.generate("hello")
    assert 0 < res.stats["tokens"] <= 60
    searched = [s for s in res.trace if s.searched]
    assert res.stats["searched"] == len(searched) > 0
    assert res.stats["fired"] == len(res.trace)
    for step in searched:
        assert sum(step.select_probs) == pytest.approx(1.0)
        assert step.trigger_info["norm_entropy"] >= 0.3
        assert len(step.lookahead_texts[0]) == 2
 
 
def test_loop_reproducible_with_seed():
    a = _engine().generate("hello")
    b = _engine().generate("hello")
    assert a.response_ids == b.response_ids
 
 
def test_never_trigger_equals_standard_decoding():
    res = _engine(trigger=NeverTrigger()).generate("hello")
    assert res.stats["fired"] == 0 and not res.trace
 
 
def test_guard_steers_away_from_harm_token():
    backend = FakeBackend()
    guard, scorer = _guard_scorer()
    res = _engine(scorer, backend=backend, trigger=EntropyTrigger(0.0),
                  select=BlendSelect(mode="argmax")).generate("hello")
    searched = [s for s in res.trace if s.searched]
    chosen = [s.candidate_ids[s.chosen_index] for s in searched]
    offered = sum(HARM in s.candidate_ids for s in searched)
    assert HARM not in chosen
    assert HARM not in res.response_ids
    print(f"HARM offered at {offered} of {len(searched)} steps, never chosen")
 
 
def test_composite_and_stop_candidate():
    backend = FakeBackend()
    _, guard_sc = _guard_scorer()
    comp = CompositeScorer([(guard_sc, 1.0), (LogLikelihoodScorer(), 0.5)])
    assert comp.needs_logprobs == "chosen"
    ids = np.array([EOS, 5])
    lps = np.log(np.array([0.5, 0.5]))
    branches = RolloutLookahead()(backend, [1, 2, 3], Candidates(ids, lps, lps), "chosen", 0)
    assert branches[0][0].continuation.finish_reason == "candidate_is_stop"
    assert branches[1][0].continuation.token_ids
 
 
def test_calibrate_threshold():
    backend = FakeBackend()
    thr, hs = calibrate_entropy_threshold(
        backend, [[1, 2, 3]] * 4, target_rate=0.2, max_tokens=50,
        decoding=DecodingConfig(temperature=1.0, top_k=10, top_p=0.9))
    assert np.mean(hs >= thr) == pytest.approx(0.2, abs=0.05)
 
 
# -- num_logprobs resolution ---------------------------------------------------------
 
def test_auto_resolves_to_filter_k():
    backend = FakeBackend(num_logprobs=AUTO)
    _engine(backend=backend, config=_cfg(decoding=DecodingConfig(top_k=25)))
    assert backend.num_logprobs == 25
 
 
def test_explicit_num_logprobs_is_kept_separate_from_k():
    backend = FakeBackend(num_logprobs=30)
    _engine(backend=backend)   # top_k=10
    assert backend.num_logprobs == 30
 
 
def test_full_vocab_passes_and_gives_exact_entropy():
    res = _engine(backend=FakeBackend(num_logprobs=FULL_VOCAB)).generate("hello")
    assert res.trace and all(s.tail_mass == pytest.approx(0.0, abs=1e-9) for s in res.trace)
 
 
def test_small_num_logprobs_warns_at_init():
    with pytest.warns(UserWarning, match="less reliable"):
        _engine(backend=FakeBackend(num_logprobs=AUTO),
                config=_cfg(decoding=DecodingConfig(top_k=5)))
 
 
def test_top_k_above_num_logprobs_errors():
    with pytest.raises(ValueError, match="needs 30 logprobs"):
        _engine(backend=FakeBackend(num_logprobs=20),
                config=_cfg(decoding=DecodingConfig(top_k=30)))
 
 
@pytest.mark.parametrize("bad", [0, -2, None, "all", "top_k", True])
def test_invalid_num_logprobs_rejected(bad):
    with pytest.raises(ValueError):
        resolve_num_logprobs(bad, _cfg(), [TopKTopPFilter()])
 
 
def test_auto_with_top_k_disabled_errors():
    cfg = _cfg(decoding=DecodingConfig(top_k=None))
    with pytest.raises(ValueError, match="no stage declares one"):
        resolve_num_logprobs(AUTO, cfg, [TopKTopPFilter()])
 
 
def test_llm_max_logprobs_cap_check():
    from types import SimpleNamespace
    from plumcot.backend import VLLMBackend
 
    def fake(n, cap):
        llm = SimpleNamespace(model_config=SimpleNamespace(max_logprobs=cap))
        return SimpleNamespace(llm=llm, num_logprobs=n)
 
    VLLMBackend.check_logprob_cap(fake(50, 50))
    VLLMBackend.check_logprob_cap(fake(FULL_VOCAB, FULL_VOCAB))
    VLLMBackend.check_logprob_cap(fake(50, FULL_VOCAB))
    with pytest.raises(ValueError, match="max_logprobs=50"):
        VLLMBackend.check_logprob_cap(fake(50, 20))
    with pytest.raises(ValueError):
        VLLMBackend.check_logprob_cap(fake(FULL_VOCAB, 64))
    with pytest.raises(ValueError, match="not been resolved"):
        VLLMBackend.check_logprob_cap(fake(AUTO, 64))
