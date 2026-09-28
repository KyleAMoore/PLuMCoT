"""Next-token distributions reconstructed from top-N logprobs, plus the
entropy (Trigger) and top-k/top-p (Filter) computations on them."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np

_NEG = -1e9  # stand-in for -inf so arithmetic stays finite


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
        items = list(items)
        ids = np.array([int(t) for t, _ in items], dtype=np.int64)
        lps = np.array([float(lp) for _, lp in items], dtype=np.float64)
        lps = np.where(np.isfinite(lps), lps, _NEG)
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
        """Entropy in nats. See TriggerConfig.tail for the tail estimate."""
        p = self.probs
        h = float(-(p * self.logprobs).sum())
        r = self.tail_mass()
        n_tail = self.vocab_size - len(self.token_ids)
        if r > 1e-12 and n_tail > 0:
            if tail == "lumped":
                h -= r * math.log(r)
            elif tail == "uniform":
                h -= r * math.log(r / n_tail)
            else:
                raise ValueError(f"unknown tail mode {tail!r}")
        return h

    def normalized_entropy(self, tail: str = "lumped") -> float:
        return self.entropy(tail) / math.log(self.vocab_size)

    def max_logprob(self) -> float:
        return float(self.logprobs[0])


def top_k_top_p(
    dist: TokenDist, top_k: int, top_p: float, min_candidates: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (token_ids, raw logprobs) of the candidates that survive top-k
    and then top-p (computed over the renormalized top-k, as in HF)."""
    n_avail = len(dist.token_ids)
    k = n_avail if (top_k is None or top_k <= 0) else min(top_k, n_avail)
    ids, lps = dist.token_ids[:k], dist.logprobs[:k]
    if top_p < 1.0:
        p = np.exp(lps - lps.max())
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


def logmeanexp(x: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    out = m + np.log(np.mean(np.exp(x - m), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)
