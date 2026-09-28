"""Next-token distributions reconstructed from top-N logprobs, plus the
entropy (Trigger) and top-k/top-p (Filter) computations on them."""
from __future__ import annotations
 
import math
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple
 
import numpy as np
 
_NEG = -1e9  # stand-in for -inf logprobs so arithmetic on the slice stays finite
 
# How the entropy of the probability mass outside the returned top-N is estimated:
#   "lumped"  - the tail is treated as a single token (lower bound on H)
#   "uniform" - the tail is spread evenly over the rest of the vocab (upper bound)
TAIL_MODES = ("lumped", "uniform")
 
 
def check_tail_mode(tail: str) -> None:
    if tail not in TAIL_MODES:
        raise ValueError(f"unknown tail mode {tail!r}; choose from {TAIL_MODES}")
 
 
@dataclass
class TokenDist:
    """Top-N slice of a next-token distribution, sorted by descending logprob.
 
    Logprobs are the model's raw (temperature-1) values, so the listed
    probabilities sum to at most 1; the remainder is `tail_mass()`.
    """
    token_ids: np.ndarray   # int64
    logprobs: np.ndarray    # float64
    vocab_size: int
 
    @classmethod
    def from_items(cls, items: Iterable[Tuple[int, float]], vocab_size: int) -> "TokenDist":
        """-inf logprobs become -1e9. NaN or +inf logprobs are corrupt input
        and raise."""
        items = list(items)
        ids = np.array([int(t) for t, _ in items], dtype=np.int64)
        lps = np.array([float(lp) for _, lp in items], dtype=np.float64)
        if np.isnan(lps).any() or np.isposinf(lps).any():
            raise ValueError("TokenDist logprobs must not be NaN or +inf.")
        lps = np.where(np.isneginf(lps), _NEG, lps)
        order = np.argsort(-lps, kind="stable")
        return cls(ids[order], lps[order], vocab_size)
 
    @classmethod
    def from_vllm(cls, position: dict, vocab_size: int) -> "TokenDist":
        """`position` is one entry of vLLM's CompletionOutput.logprobs:
        {token_id: Logprob}. It holds the top-N tokens plus the sampled token."""
        return cls.from_items(
            ((t, getattr(lp, "logprob", lp)) for t, lp in position.items()), vocab_size
        )
 
    @property
    def probs(self) -> np.ndarray:
        return np.exp(self.logprobs)
 
    def tail_mass(self) -> float:
        return float(max(0.0, 1.0 - self.probs.sum()))
 
    def entropy(self, tail: str = "lumped") -> float:
        """Entropy in nats. See TAIL_MODES for the tail estimate."""
        check_tail_mode(tail)
        p = self.probs
        h = float(-(p * self.logprobs).sum())
        r = self.tail_mass()
        n_tail = self.vocab_size - len(self.token_ids)
        if r > 1e-12 and n_tail > 0:
            if tail == "lumped":
                h -= r * math.log(r)
            else:
                h -= r * math.log(r / n_tail)
        return h
 
    def normalized_entropy(self, tail: str = "lumped") -> float:
        return self.entropy(tail) / math.log(self.vocab_size)
 
    def max_logprob(self) -> float:
        return float(self.logprobs[0])
 
 
def top_k_top_p(
    dist: TokenDist, top_k: Optional[int], top_p: float, min_candidates: int = 1,
    temperature: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (token_ids, raw logprobs) of the candidates that survive top-k
    and then top-p, where top-p is computed over the top-k renormalized at
    `temperature` (as vLLM does: temperature, then top-k, then top-p).
 
    top_k=None disables top-k; top_p=1 disables top-p. At least
    `min_candidates` (capped at k) are kept.
    """
    if top_k is not None and top_k < 1:
        raise ValueError(f"top_k must be None (disabled) or >= 1; got {top_k}")
    if not 0.0 < top_p <= 1.0:
        raise ValueError(f"top_p must be in (0, 1]; got {top_p}")
    if min_candidates < 1:
        raise ValueError(f"min_candidates must be >= 1; got {min_candidates}")
    if not (math.isfinite(temperature) and temperature > 0.0):
        raise ValueError(f"temperature must be finite and > 0; got {temperature}")
    n_avail = len(dist.token_ids)
    k = n_avail if top_k is None else min(top_k, n_avail)
    ids, lps = dist.token_ids[:k], dist.logprobs[:k]
    if top_p < 1.0:
        z = lps / temperature
        p = np.exp(z - z.max())
        p /= p.sum()
        n = int(np.searchsorted(np.cumsum(p), top_p - 1e-9)) + 1
    else:
        n = k
    n = min(k, max(n, min_candidates))
    return ids[:n], lps[:n]
 
 
def softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    e = np.exp(z)
    return e / e.sum()
 
 
def log_softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    return z - np.log(np.exp(z).sum())
 
 
def logmeanexp(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """log(mean(exp(x))) along `axis`, exact for -inf entries (an all--inf
    slice gives -inf) and for +inf (gives +inf)."""
    x = np.asarray(x, dtype=np.float64)
    m = np.max(x, axis=axis, keepdims=True)
    m_safe = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(divide="ignore", over="ignore"):
        out = m_safe + np.log(np.mean(np.exp(x - m_safe), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)
