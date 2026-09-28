"""Configuration for each PLuMCoT choice point.

Configure (Stage 1.5) is currently constant: a single PLuMCoTConfig is used at
every generation iteration. `PLuMCoT.configure()` is the hook for varying it.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple, Union

# -- how many logprobs the backend requests per position ---------------------
# vLLM already gives None ("no logprobs") and -1 ("full vocabulary") special
# meanings, so matching the Filter's k uses a named sentinel instead.
MATCH_TOP_K = "top_k"
FULL_VOCAB = -1
MIN_RECOMMENDED_LOGPROBS = 20  # also vLLM's default max_logprobs

NumLogprobs = Union[int, Literal["top_k"]]


def resolve_num_logprobs(num_logprobs: NumLogprobs, cfg: "PLuMCoTConfig") -> int:
    """Turn a NumLogprobs setting into the int passed to vLLM.

    "top_k" -> cfg.filter.top_k; -1 -> full vocabulary; n >= 1 -> n.
    Pass the result as `LLM(max_logprobs=...)` too.
    """
    if num_logprobs == MATCH_TOP_K:
        if cfg.filter.top_k is None or cfg.filter.top_k <= 0:
            raise ValueError(
                'num_logprobs="top_k" needs filter.top_k > 0; with top-k disabled, '
                "set num_logprobs explicitly (or -1 for the full vocabulary)."
            )
        return int(cfg.filter.top_k)
    if isinstance(num_logprobs, bool) or not isinstance(num_logprobs, int) or (
        num_logprobs < 1 and num_logprobs != FULL_VOCAB
    ):
        raise ValueError(f'num_logprobs must be "top_k", -1, or >= 1; got {num_logprobs!r}')
    return num_logprobs


@dataclass
class TriggerConfig:
    """Stage 1: entropy gate.

    Lookahead runs only when the normalized next-token entropy H(p) / log|V|
    is at or above `threshold`. The default is a starting point; calibrate it
    to a target firing rate with `plumcot.engine.calibrate_threshold`.
    """
    enabled: bool = True
    threshold: float = 0.15
    # vLLM returns only the top-N logprobs, so the entropy contributed by the
    # remaining tail mass has to be estimated:
    #   "lumped"  - tail treated as a single token (lower bound on H)
    #   "uniform" - tail spread evenly over the rest of the vocab (upper bound)
    tail: Literal["lumped", "uniform"] = "lumped"


@dataclass
class FilterConfig:
    """Stage 2: candidate pool = top-k, then top-p over the renormalized top-k.

    The same k/p are used for ordinary (untriggered) sampling.
    """
    top_k: int = 50
    top_p: float = 0.9
    min_candidates: int = 1


@dataclass
class LookaheadConfig:
    """Stage 3: the continuation generated after each candidate token."""
    max_tokens: int = 16
    num_samples: int = 1           # rollouts per candidate (forced to 1 when greedy)
    temperature: float = 0.0       # 0.0 = greedy, i.e. a deterministic base-policy rollout
    top_p: float = 1.0
    top_k: Optional[int] = None    # None = disabled
    stop: Tuple[str, ...] = ()     # optional stop strings, e.g. ("\n\n",)
    # How per-sample scores combine into one score per candidate. None uses the
    # scorer's default: "logmeanexp" for kind="logprob" scorers (with tau = 1
    # this is log of the mean probability, log V-hat, the Bayesian estimate) and
    # "mean" for kind="utility" scorers (risk-neutral expected utility).
    aggregate: Optional[Literal["mean", "max", "min", "logmeanexp"]] = None
    # tau for "logmeanexp": tau * log mean exp(score / tau), the entropic risk
    # measure. tau > 0 is optimistic (-> "max" as tau -> 0+), tau < 0 is
    # pessimistic (-> "min" as tau -> 0-), and |tau| -> inf recovers "mean".
    aggregate_temperature: float = 1.0


@dataclass
class SelectConfig:
    """Stage 5: choose among candidates.

        P(c) ∝ p(c)^(1 - w) · exp(w · score(c)),   w = judge_weight

    A log-linear (logarithmic opinion) pool of the model and the judge: the
    minimizer of (1 - w)·KL(q‖p) + w·KL(q‖q_S), where q_S ∝ exp(score).
    Equivalently, a Boltzmann choice over the blended utility
    U(c) = (1 - w)·log p(c) + w·score(c): the immediate term (how natural
    the token is, in nats) against the lookahead's value.

    w = 0 is plain decoding over the filtered candidates; w = 1 selects on
    the score alone (a uniform prior over the filtered candidates).

    With score = log P(ok | c) (a kind="logprob" scorer), this is a tempered
    Bayesian update: p(c)^(1-w)·P(ok|c)^w = [p(c)·P(ok|c)^k]^(1-w) with
    evidence weight k = w / (1 - w), sampled at temperature 1 / (1 - w).
    w = 0.5 gives exact Bayesian MAP in argmax mode.
    """
    mode: Literal["sample", "argmax"] = "sample"
    judge_weight: float = 0.5


@dataclass
class PLuMCoTConfig:
    trigger: TriggerConfig = field(default_factory=TriggerConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    lookahead: LookaheadConfig = field(default_factory=LookaheadConfig)
    select: SelectConfig = field(default_factory=SelectConfig)
    # Temperature for ordinary (untriggered) sampling.
    temperature: float = 1.0
    max_new_tokens: int = 512
    # Untriggered tokens are decoded in chunks, and each chunk is truncated at
    # the first position where the trigger fires. This is exact: the kept
    # tokens were sampled from the standard policy either way.
    chunk_size: int = 32
    seed: Optional[int] = None

    def validate(self, num_logprobs: int) -> None:
        """`num_logprobs` is the resolved per-position logprob count (see
        resolve_num_logprobs); -1 means the full vocabulary."""
        full = num_logprobs == FULL_VOCAB
        top_k_on = self.filter.top_k is not None and self.filter.top_k > 0
        if not full and top_k_on and self.filter.top_k > num_logprobs:
            raise ValueError(
                f"filter.top_k={self.filter.top_k} exceeds num_logprobs={num_logprobs}; "
                "raise num_logprobs (and vLLM's max_logprobs) or lower top_k."
            )
        if not full and not top_k_on:
            warnings.warn(
                f"filter.top_k is disabled, so top-p is computed over only the top "
                f"{num_logprobs} tokens and the candidate pool may be truncated.",
                stacklevel=3,
            )
        if not full and num_logprobs < MIN_RECOMMENDED_LOGPROBS:
            warnings.warn(
                f"num_logprobs={num_logprobs} is below {MIN_RECOMMENDED_LOGPROBS}. The "
                "entropy Trigger and entropy-based scores see only the top "
                f"{num_logprobs} tokens, which can make PLuMCoT's behavior, and its "
                "improvement over base decoding, less reliable.",
                stacklevel=3,
            )
        if not 0.0 < self.filter.top_p <= 1.0:
            raise ValueError("filter.top_p must be in (0, 1].")
        if not 0.0 <= self.select.judge_weight <= 1.0:
            raise ValueError("select.judge_weight must be in [0, 1].")
        tau = self.lookahead.aggregate_temperature
        if not math.isfinite(tau) or tau == 0.0:
            raise ValueError("lookahead.aggregate_temperature must be finite and nonzero "
                             "(negative values give a pessimistic soft-min).")
        if self.lookahead.aggregate not in (None, "mean", "max", "min", "logmeanexp"):
            raise ValueError(f"unknown lookahead.aggregate {self.lookahead.aggregate!r}")
        if self.chunk_size < 1 or self.lookahead.max_tokens < 1:
            raise ValueError("chunk_size and lookahead.max_tokens must be >= 1.")
