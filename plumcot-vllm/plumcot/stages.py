"""PLuMCoT stages (dissertation §7.4.2, Algorithm 1) as pluggable objects.
 
  Stage 1    Trigger           fixed for the whole generation (passed to PLuMCoT)
  Stage 1.5  Configure         runs only on triggered steps; picks one of its
                               declared StageBundles
  Stage 2    Filter            ┐
  Stage 3    Lookahead         │ the StageBundle Configure picked
  Stage 4    Scorer + SampleAggregator
  Stage 5    Select            ┘
 
Every stage is an object with its own settings, validated when constructed
(bad values raise immediately) and again against the loop configuration when
PLuMCoT is constructed (Stage.validate). The default implementations here
reproduce the proof of concept's behavior; subclass the base classes to try
variations.
 
Couplings between stages that a new implementation must respect:
 
* A Trigger is evaluated per position of an already-decoded chunk, so it may
  depend only on that position's next-token distribution and the token ids
  before it.
* The generator's logprob count is fixed at startup. Stages declare what
  they need through `required_logprobs`, and every option Configure can
  return is checked when PLuMCoT is constructed.
* A Filter returns candidates with both their raw logprobs and their prior
  under the base decoding policy; Select uses the prior.
"""
from __future__ import annotations
 
import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union
 
import numpy as np
 
from .backend import Backend, Continuation, LogprobMode
from .base import Stage
from .config import AUTO, FULL_VOCAB, DecodingConfig
from .dist import TokenDist, check_tail_mode, log_softmax, logmeanexp, softmax, top_k_top_p
from .scorers import (ALL_NAN_POLICIES, Branch, Scorer, _check_choice, fill_nan,
                      handle_all_nan)
 
MATCH_DECODING = "decoding"
 
 
def _is_int(v) -> bool:
    return isinstance(v, (int, np.integer)) and not isinstance(v, bool)
 
 
@dataclass
class StepContext:
    """What Configure sees at a triggered step."""
    prompt_ids: List[int]
    response_ids: List[int]          # committed response before this step
    dist: TokenDist                  # the next-token distribution at this step
    trigger_info: Dict[str, float]   # whatever the Trigger reported
 
 
# ---------------------------------------------------------------------------
# Stage 1: Trigger
# ---------------------------------------------------------------------------
 
class Trigger(Stage):
    """Decides, per position, whether to run the PLuMCoT step there.
 
    __call__(dist, prompt_ids, response_ids) -> (fire, info)
      dist:         raw next-token distribution at this position
      response_ids: the response tokens before this position
      info:         floats recorded in the trace (may be empty)
    """
 
    def __call__(self, dist: TokenDist, prompt_ids: Sequence[int],
                 response_ids: Sequence[int]) -> Tuple[bool, Dict[str, float]]:
        raise NotImplementedError
 
 
@dataclass
class EntropyTrigger(Trigger):
    """Fire when the normalized raw next-token entropy H(p) / log|V| is at
    or above `threshold`. The entropy is always computed on the model's raw
    (temperature-1) distribution, whatever the decoding temperature.
 
    tail: how the entropy of the mass outside the returned top-N logprobs is
    estimated ("lumped" = lower bound, "uniform" = upper bound). The default
    threshold is a starting point; calibrate it with
    `calibrate_entropy_threshold`.
    """
    threshold: float = 0.15
    tail: str = "lumped"
 
    def __post_init__(self):
        if not math.isfinite(self.threshold):
            raise ValueError(f"threshold must be finite; got {self.threshold}")
        check_tail_mode(self.tail)
 
    def __call__(self, dist, prompt_ids, response_ids):
        h = dist.normalized_entropy(self.tail)
        return h >= self.threshold, {"norm_entropy": h}
 
 
@dataclass
class NeverTrigger(Trigger):
    """Never fires: PLuMCoT reduces to base decoding."""
 
    def __call__(self, dist, prompt_ids, response_ids):
        return False, {}
 
 
@dataclass
class AlwaysTrigger(Trigger):
    """Fires at every position (expensive; mainly for experiments)."""
 
    def __call__(self, dist, prompt_ids, response_ids):
        return True, {}
 
 
def calibrate_entropy_threshold(
    backend: Backend,
    prompts: Sequence[List[int]],
    target_rate: float = 0.1,
    *,
    decoding: DecodingConfig,
    max_tokens: int = 256,
    tail: str = "lumped",
    seed: Optional[int] = None,
) -> Tuple[float, np.ndarray]:
    """Decode with the base policy and return the normalized-entropy
    threshold at which an EntropyTrigger would fire on roughly
    `target_rate` of tokens, plus all observed normalized entropies.
 
    The backend's num_logprobs must already be an int (the count the engine
    will use), since it changes the entropy estimate.
    """
    if not 0.0 < target_rate < 1.0:
        raise ValueError(f"target_rate must be in (0, 1); got {target_rate}")
    check_tail_mode(tail)
    decoding.validate()
    if backend.num_logprobs == AUTO:
        raise ValueError('Resolve num_logprobs before calibrating (e.g. with '
                         'resolve_num_logprobs); "auto" depends on the stages.')
    outs = backend.generate(prompts, n=1, max_tokens=max_tokens,
                            temperature=decoding.temperature, top_k=decoding.top_k,
                            top_p=decoding.top_p, logprobs="dist", seed=seed)
    hs = []
    for req in outs:
        if req[0].dists is None:
            raise RuntimeError("The backend returned no distributions for calibration.")
        hs += [d.normalized_entropy(tail) for d in req[0].dists]
    hs = np.array(hs)
    if hs.size == 0:
        raise RuntimeError("No tokens generated during calibration.")
    return float(np.quantile(hs, 1.0 - target_rate)), hs
 
 
# ---------------------------------------------------------------------------
# Stage 2: Filter
# ---------------------------------------------------------------------------
 
@dataclass
class Candidates:
    """Filter output, sorted by descending raw logprob.
 
    logprobs:       the model's raw (temperature-1) log-probabilities.
    prior_logprobs: log-probabilities under the base decoding policy
                    (decoding temperature applied), renormalized over the
                    candidates. Select's prior.
    """
    token_ids: np.ndarray
    logprobs: np.ndarray
    prior_logprobs: np.ndarray
 
    def __post_init__(self):
        n = len(self.token_ids)
        if n == 0:
            raise ValueError("A Filter must return at least one candidate.")
        if len(self.logprobs) != n or len(self.prior_logprobs) != n:
            raise ValueError("Candidates arrays must have the same length.")
 
 
def base_policy_prior(raw_logprobs: np.ndarray, temperature: float) -> np.ndarray:
    """log p_T(c) renormalized over the candidates, where p_T ∝ p^(1/T).
    At T = 0 the base policy is greedy: a point mass on the top candidate
    (the first, since candidates are sorted by descending logprob)."""
    lps = np.asarray(raw_logprobs, dtype=np.float64)
    if temperature == 0.0:
        prior = np.full(len(lps), -np.inf)
        prior[int(np.argmax(lps))] = 0.0
        return prior
    return log_softmax(lps / temperature)
 
 
class Filter(Stage):
    """__call__(dist, decoding) -> Candidates"""
 
    def __call__(self, dist: TokenDist, decoding: DecodingConfig) -> Candidates:
        raise NotImplementedError
 
 
@dataclass
class TopKTopPFilter(Filter):
    """Candidate pool = top-k, then top-p over the renormalized top-k,
    matching how vLLM samples: at the base decoding temperature T, top-p is
    computed on p^(1/T), so the pool is exactly the support of the base
    policy at this position.
 
    top_k: an int >= 1, None (disabled), or "decoding" (use decoding.top_k).
    top_p: a float in (0, 1], or "decoding" (use decoding.top_p).
    min_candidates: keep at least this many (capped at k).
 
    At T = 0 the base policy is greedy, so its support is a single token and
    a pool matching it would make every step single-candidate. The pool is
    then computed on the raw (T = 1) distribution instead; the prior is still
    the greedy point mass (see BlendSelect for what that implies).
    """
    top_k: Union[int, None, str] = MATCH_DECODING
    top_p: Union[float, str] = MATCH_DECODING
    min_candidates: int = 1
 
    def __post_init__(self):
        if self.top_k != MATCH_DECODING and self.top_k is not None and (
                not _is_int(self.top_k) or self.top_k < 1):
            raise ValueError(f'top_k must be an int >= 1, None, or "{MATCH_DECODING}"; '
                             f"got {self.top_k!r}")
        if self.top_p != MATCH_DECODING and not (
                isinstance(self.top_p, (int, float)) and 0.0 < self.top_p <= 1.0):
            raise ValueError(f'top_p must be in (0, 1] or "{MATCH_DECODING}"; got {self.top_p!r}')
        if not _is_int(self.min_candidates) or self.min_candidates < 1:
            raise ValueError(f"min_candidates must be an int >= 1; got {self.min_candidates!r}")
 
    def resolve(self, decoding: DecodingConfig) -> Tuple[Optional[int], float]:
        k = decoding.top_k if self.top_k == MATCH_DECODING else self.top_k
        p = decoding.top_p if self.top_p == MATCH_DECODING else self.top_p
        return k, float(p)
 
    def required_logprobs(self, config):
        k, _ = self.resolve(config.decoding)
        return k
 
    def validate(self, ctx):
        k, _ = self.resolve(ctx.config.decoding)
        if k is None and ctx.num_logprobs != FULL_VOCAB:
            warnings.warn(
                f"TopKTopPFilter has top-k disabled, so top-p is computed over only the top "
                f"{ctx.num_logprobs} tokens and the candidate pool may be truncated.",
                stacklevel=4,
            )
 
    def __call__(self, dist, decoding):
        k, p = self.resolve(decoding)
        t = decoding.temperature
        ids, lps = top_k_top_p(dist, k, p, self.min_candidates,
                               temperature=t if t > 0.0 else 1.0)
        return Candidates(ids, lps, base_policy_prior(lps, t))
 
 
# ---------------------------------------------------------------------------
# Stage 3: Lookahead
# ---------------------------------------------------------------------------
 
class Lookahead(Stage):
    """__call__(backend, context, candidates, logprobs, seed) -> branches
 
    branches[i] is the list of Branches (one per sample) for candidate i;
    every candidate must have the same number of samples. `logprobs` is what
    the bundle's Scorer needs per lookahead token.
    """
 
    def __call__(self, backend: Backend, context: List[int], candidates: Candidates,
                 logprobs: LogprobMode, seed: Optional[int]) -> List[List[Branch]]:
        raise NotImplementedError
 
 
@dataclass
class RolloutLookahead(Lookahead):
    """Continue each candidate with the generator for up to `max_tokens`
    tokens, `num_samples` times, stopping at a stop token or a `stop` string.
 
    temperature 0 is greedy, which is deterministic, so num_samples must then
    be 1. top_k None disables top-k; top_p 1 disables top-p.
 
    A candidate that is itself a stop token ends the response, so it gets
    `num_samples` empty continuations (finish_reason "candidate_is_stop")
    instead of a rollout.
    """
    max_tokens: int = 16
    num_samples: int = 1
    temperature: float = 0.0
    top_k: Optional[int] = None
    top_p: float = 1.0
    stop: Tuple[str, ...] = ()
 
    def __post_init__(self):
        if not _is_int(self.max_tokens) or self.max_tokens < 1:
            raise ValueError(f"max_tokens must be an int >= 1; got {self.max_tokens!r}")
        if not _is_int(self.num_samples) or self.num_samples < 1:
            raise ValueError(f"num_samples must be an int >= 1; got {self.num_samples!r}")
        if not (math.isfinite(self.temperature) and self.temperature >= 0.0):
            raise ValueError(f"temperature must be finite and >= 0; got {self.temperature}")
        if self.temperature == 0.0 and self.num_samples != 1:
            raise ValueError(
                f"num_samples={self.num_samples} with temperature=0: greedy rollouts are "
                "identical, so extra samples add cost without information. Use "
                "num_samples=1, or a temperature > 0 to sample distinct rollouts."
            )
        if self.top_k is not None and (not _is_int(self.top_k) or self.top_k < 1):
            raise ValueError(f"top_k must be None (disabled) or an int >= 1; got {self.top_k!r}")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1]; got {self.top_p}")
        self.stop = tuple(self.stop)
 
    def __call__(self, backend, context, candidates, logprobs, seed):
        n = self.num_samples
        ids = [int(c) for c in candidates.token_ids]
        stop_ids = backend.stop_token_ids
        live = [i for i, c in enumerate(ids) if c not in stop_ids]
        conts = backend.generate(
            [list(context) + [ids[i]] for i in live],
            n=n, max_tokens=self.max_tokens, temperature=self.temperature,
            top_k=self.top_k, top_p=self.top_p, stop=self.stop,
            logprobs=logprobs, seed=seed,
        ) if live else []
        if len(conts) != len(live) or any(len(cs) != n for cs in conts):
            raise RuntimeError(f"The backend returned the wrong number of rollouts "
                               f"(expected {n} for each of {len(live)} candidates).")
        by_cand: Dict[int, List[Continuation]] = dict(zip(live, conts))
        empty = [] if logprobs != "none" else None
        branches = []
        for i, (c, lp) in enumerate(zip(ids, candidates.logprobs)):
            if i in by_cand:
                cs = by_cand[i]
            else:
                cs = [Continuation([], empty, [] if logprobs == "dist" else None,
                                   "candidate_is_stop") for _ in range(n)]
            branches.append([Branch(c, float(lp), cont) for cont in cs])
        return branches
 
 
# ---------------------------------------------------------------------------
# Stage 4 (second half): combine samples into one score per candidate
# ---------------------------------------------------------------------------
 
AGGREGATE_MODES = ("mean", "max", "min", "logmeanexp")
 
 
@dataclass
class SampleAggregator(Stage):
    """Turns the Scorer's per-sample output into one score per candidate:
    fills NaNs (see scorers.fill_nan, which follows the scorer's kind),
    multiplies by `scorer.scale`, then reduces over samples.
 
    mode: None follows the scorer's kind ("logmeanexp" for "logprob"
        scorers, i.e. the log of the mean probability at temperature 1;
        "mean" for "utility" scorers), or one of "mean", "max", "min",
        "logmeanexp".
    temperature: tau for "logmeanexp", tau · log mean exp(score / tau), the
        entropic risk measure. tau > 0 is optimistic (-> "max" as tau -> 0+),
        tau < 0 pessimistic (-> "min" as tau -> 0-), and |tau| -> inf gives
        the mean.
    on_all_nan: what to do when every branch at a step scores NaN (scores
        become 0, so Select uses the prior alone): "warn", "raise" or "ignore".
    """
    mode: Optional[str] = None
    temperature: float = 1.0
    on_all_nan: str = "warn"
 
    def __post_init__(self):
        if self.mode is not None:
            _check_choice("mode", self.mode, AGGREGATE_MODES)
        if not math.isfinite(self.temperature) or self.temperature == 0.0:
            raise ValueError("temperature must be finite and nonzero (negative values give "
                             "a pessimistic soft-min).")
        _check_choice("on_all_nan", self.on_all_nan, ALL_NAN_POLICIES)
 
    def mode_for(self, scorer: Scorer) -> str:
        return self.mode if self.mode is not None else scorer.default_aggregate
 
    def __call__(self, raw: np.ndarray, scorer: Scorer) -> Tuple[np.ndarray, np.ndarray]:
        """raw: (num_candidates, num_samples) scores as the scorer returned
        them. Returns (per-candidate scores, filled and scaled samples)."""
        raw = np.asarray(raw, dtype=np.float64)
        filled, all_nan = fill_nan(raw.ravel(), scorer.kind)
        if all_nan:
            handle_all_nan(self.on_all_nan, f"scorer {scorer.name!r}")
        x = filled.reshape(raw.shape) * scorer.scale
        mode = self.mode_for(scorer)
        if mode == "mean":
            out = x.mean(axis=1)
        elif mode == "max":
            out = x.max(axis=1)
        elif mode == "min":
            out = x.min(axis=1)
        else:
            tau = self.temperature
            out = tau * logmeanexp(x / tau, axis=1)
        return out, x
 
 
# ---------------------------------------------------------------------------
# Stage 5: Select
# ---------------------------------------------------------------------------
 
class Select(Stage):
    """__call__(candidates, scores, rng) -> (chosen index, selection probabilities)"""
 
    def __call__(self, candidates: Candidates, scores: np.ndarray,
                 rng: np.random.Generator) -> Tuple[int, np.ndarray]:
        raise NotImplementedError
 
 
@dataclass
class BlendSelect(Select):
    """Choose among candidates by blending the base policy with the score:
 
        U(c) = (1 - w)·log p_T(c) + w·score(c),   w = judge_weight
        P(c) = softmax(U)(c)                      (temperature 1)
 
    p_T is the candidate's prior under base decoding (Candidates.prior_logprobs),
    so at w = 0 this is exactly base decoding restricted to the Filter's pool.
    The softmax renormalizes over the candidates. mode="sample" draws from P;
    mode="argmax" takes its mode.
 
    A log-linear (logarithmic opinion) pool of the model and the judge: the
    minimizer of (1 - w)·KL(q‖p_T) + w·KL(q‖q_S), where q_S ∝ exp(score).
    w = 1 selects on the score alone (a uniform prior over the pool).
 
    With score = log P(ok | c) (a kind="logprob" scorer), this is a tempered
    Bayesian update: p_T(c)^(1-w)·P(ok|c)^w = [p_T(c)·P(ok|c)^k]^(1-w) with
    evidence weight k = w / (1 - w), sampled at temperature 1 / (1 - w).
    w = 0.5 gives exact Bayesian MAP in argmax mode.
 
    Scores may be -inf (candidate ruled out) but not +inf or NaN, and at
    least one candidate must have a finite utility.
    """
    judge_weight: float = 0.5
    mode: str = "sample"
 
    def __post_init__(self):
        if not 0.0 <= self.judge_weight <= 1.0:
            raise ValueError(f"judge_weight must be in [0, 1]; got {self.judge_weight}")
        _check_choice("mode", self.mode, ("sample", "argmax"))
 
    def validate(self, ctx):
        # Guard: at decoding temperature 0 the base policy is a point mass on
        # the top candidate, so log p_T(c) = -inf for every other candidate.
        # With w < 1 that term dominates any finite score and Select always
        # picks the top candidate, i.e. the lookahead and scoring would run
        # but could never change the output. Only w = 1 (score only, which
        # ignores the prior) is meaningful at temperature 0.
        if ctx.config.decoding.temperature == 0.0 and self.judge_weight < 1.0:
            raise ValueError(
                f"BlendSelect(judge_weight={self.judge_weight}) cannot be used with "
                "decoding.temperature=0. Greedy base decoding makes the prior a point mass "
                "on the top token, so every triggered step would pick that token regardless "
                "of the scores. Use judge_weight=1 (select on the score alone), or a "
                "decoding temperature > 0."
            )
 
    def __call__(self, candidates, scores, rng):
        scores = np.asarray(scores, dtype=np.float64)
        w = self.judge_weight
        u = np.zeros(len(candidates.token_ids))
        # Each term is added only when its weight is nonzero, so a -inf prior
        # (greedy base policy) or -inf score does not produce 0·(-inf) = NaN.
        if w < 1.0:
            u += (1.0 - w) * candidates.prior_logprobs
        if w > 0.0:
            if np.isnan(scores).any() or np.isposinf(scores).any():
                raise ValueError("Select received NaN or +inf scores.")
            u += w * scores
        if not np.isfinite(u).any():
            raise ValueError("Every candidate has utility -inf, so Select has nothing to choose.")
        probs = softmax(u)
        idx = int(np.argmax(probs)) if self.mode == "argmax" else int(
            rng.choice(len(probs), p=probs))
        return idx, probs
 
 
# ---------------------------------------------------------------------------
# Bundles and Stage 1.5: Configure
# ---------------------------------------------------------------------------
 
@dataclass
class StageBundle:
    """One complete choice of Stages 2-5. Unspecified stages use their
    default implementations with default settings."""
    scorer: Scorer
    filter: Filter = field(default_factory=TopKTopPFilter)
    lookahead: Lookahead = field(default_factory=RolloutLookahead)
    aggregate: SampleAggregator = field(default_factory=SampleAggregator)
    select: Select = field(default_factory=BlendSelect)
 
    def __post_init__(self):
        for name, cls in (("scorer", Scorer), ("filter", Filter), ("lookahead", Lookahead),
                          ("aggregate", SampleAggregator), ("select", Select)):
            if not isinstance(getattr(self, name), cls):
                raise TypeError(f"StageBundle.{name} must be a {cls.__name__}; "
                                f"got {type(getattr(self, name)).__name__}")
 
    def stages(self) -> List[Stage]:
        return [self.filter, self.lookahead, self.scorer, self.aggregate, self.select]
 
    def to_dict(self) -> dict:
        return {"filter": self.filter.to_dict(), "lookahead": self.lookahead.to_dict(),
                "scorer": self.scorer.to_dict(), "aggregate": self.aggregate.to_dict(),
                "select": self.select.to_dict()}
 
 
class Configure(Stage):
    """Stage 1.5. Runs only on triggered steps and picks which StageBundle
    to use there.
 
    Every bundle it can pick must be declared up front in `options`, so all
    of them are validated (including logprob requirements) when PLuMCoT is
    constructed. `choose(step)` returns an index into `options`.
    """
 
    def __init__(self, options: Sequence[StageBundle]):
        self.options = list(options)
        if not self.options:
            raise ValueError("Configure needs at least one StageBundle option.")
        for b in self.options:
            if not isinstance(b, StageBundle):
                raise TypeError(f"Configure options must be StageBundles; got {type(b).__name__}")
 
    def choose(self, step: StepContext) -> int:
        raise NotImplementedError
 
    def to_dict(self) -> dict:
        return {"type": type(self).__name__, "options": [b.to_dict() for b in self.options]}
 
 
class FixedConfigure(Configure):
    """The same bundle at every triggered step."""
 
    def __init__(self, bundle: StageBundle):
        super().__init__([bundle])
 
    def choose(self, step: StepContext) -> int:
        return 0
