"""Shared fixtures and helpers for the plumcot test suite.

Real-vLLM tests
---------------
Tests marked `vllm`, and the "vllm" parametrization of `make_backend`, run
against a real model when PLUMCOT_TEST_MODEL names one (a Hugging Face id or
path) and are skipped otherwise:

    PLUMCOT_TEST_MODEL=Qwen/Qwen2.5-0.5B-Instruct pytest tests/
    pytest tests/ -m "not vllm"          # fake backend only

Optional: PLUMCOT_TEST_GPU_MEM (default 0.5), PLUMCOT_TEST_MAX_LOGPROBS
(default 64). One LLM is built per session and shared; each test wraps it in
a fresh VLLMBackend because PLuMCoT resolves num_logprobs on the backend.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fake_backend import EOS, V, FakeBackend, FakeTokenizer  # noqa: E402
from plumcot import (Candidates, Continuation, DecodingConfig, PLuMCoTConfig,  # noqa: E402
                     TokenDist)

MODEL = os.environ.get("PLUMCOT_TEST_MODEL")


def pytest_configure(config):
    config.addinivalue_line("markers", "vllm: needs a real vLLM model (PLUMCOT_TEST_MODEL)")


# ---------------------------------------------------------------------------
# backends
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def vllm_llm():
    if not MODEL:
        pytest.skip("set PLUMCOT_TEST_MODEL to run real-vLLM tests")
    from vllm import LLM

    return LLM(MODEL, max_logprobs=int(os.environ.get("PLUMCOT_TEST_MAX_LOGPROBS", "64")),
               enable_prefix_caching=True,
               gpu_memory_utilization=float(os.environ.get("PLUMCOT_TEST_GPU_MEM", "0.5")))


@pytest.fixture
def make_vllm_backend(vllm_llm):
    from plumcot import VLLMBackend

    def make(num_logprobs=20):
        return VLLMBackend(vllm_llm, num_logprobs=num_logprobs, model_name=MODEL)
    return make


@pytest.fixture(params=["fake", pytest.param("vllm", marks=pytest.mark.vllm)])
def make_backend(request):
    """Factory for a generator backend; tests using it run on the fake
    backend and, when PLUMCOT_TEST_MODEL is set, on the real model."""
    if request.param == "fake":
        return lambda num_logprobs=20: FakeBackend(num_logprobs=num_logprobs)
    return request.getfixturevalue("make_vllm_backend")


def prompt_for(backend) -> List[int]:
    """A short prompt in the backend's token space."""
    return list(backend.tokenizer.encode("The capital of France is"))


class RecordingBackend:
    """Wraps a backend and records every generate() call's arguments."""

    def __init__(self, inner):
        self.inner = inner
        self.calls: List[dict] = []
        self.outputs: List[list] = []
        self.cap_checks = 0

    @property
    def tokenizer(self):
        return self.inner.tokenizer

    @property
    def vocab_size(self):
        return self.inner.vocab_size

    @property
    def stop_token_ids(self):
        return self.inner.stop_token_ids

    @property
    def num_logprobs(self):
        return self.inner.num_logprobs

    @num_logprobs.setter
    def num_logprobs(self, v):
        self.inner.num_logprobs = v

    def check_logprob_cap(self):
        self.cap_checks += 1
        self.inner.check_logprob_cap()

    def generate(self, contexts, **kw):
        self.calls.append(dict(contexts=[list(c) for c in contexts], **kw))
        out = self.inner.generate(contexts, **kw)
        self.outputs.append(out)
        return out


class ScriptedBackend:
    """Returns queued results from generate(), in order; records calls.
    Uses the fake tokenizer. `results` items are List[List[Continuation]]."""

    def __init__(self, results=(), num_logprobs=20, stop_token_ids=(EOS,)):
        self.tokenizer = FakeTokenizer()
        self.vocab_size = V
        self.num_logprobs = num_logprobs
        self.stop_token_ids = set(stop_token_ids)
        self.queue = list(results)
        self.calls: List[dict] = []

    def check_logprob_cap(self):
        pass

    def generate(self, contexts, **kw):
        self.calls.append(dict(contexts=[list(c) for c in contexts], **kw))
        if not self.queue:
            raise AssertionError("ScriptedBackend ran out of scripted results")
        return self.queue.pop(0)


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------

def dist_from_probs(probs: Sequence[float], ids: Optional[Sequence[int]] = None,
                    vocab_size: int = V) -> TokenDist:
    ids = list(range(len(probs))) if ids is None else list(ids)
    return TokenDist.from_items(zip(ids, np.log(np.asarray(probs, dtype=float))), vocab_size)


def uniform_dist(n: int, vocab_size: int = V) -> TokenDist:
    return dist_from_probs([1.0 / vocab_size] * n, vocab_size=vocab_size)


def candidates(raw_probs: Sequence[float], ids=None, temperature: float = 1.0) -> Candidates:
    from plumcot import base_policy_prior

    lps = np.log(np.asarray(raw_probs, dtype=float))
    ids = np.arange(len(lps)) if ids is None else np.asarray(ids)
    return Candidates(ids, lps, base_policy_prior(lps, temperature))


def cont(ids: Sequence[int], lps: Optional[Sequence[float]] = None, dists=None,
         finish="length") -> Continuation:
    return Continuation(list(ids), None if lps is None else list(lps), dists, finish)


def loop_config(temperature=1.0, top_k=10, top_p=0.9, **kw) -> PLuMCoTConfig:
    kw.setdefault("max_new_tokens", 40)
    kw.setdefault("seed", 1)
    kw.setdefault("chunk_size", 8)
    return PLuMCoTConfig(decoding=DecodingConfig(temperature, top_k, top_p), **kw)


def entropy_of(probs: Sequence[float]) -> float:
    p = np.asarray(probs, dtype=float)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def first_derived_seed(seed: int) -> int:
    """The seed PLuMCoT passes with its first backend call."""
    return int(np.random.default_rng(seed).integers(2**31 - 1))


LN2 = math.log(2.0)


class ArrayScorer:
    """Factory for a Scorer that returns fixed values (or a function of the
    branches), with a chosen kind and scale."""

    @staticmethod
    def make(values=None, kind="utility", scale=1.0, fn=None, needs_logprobs="none",
             name="array"):
        from plumcot import Scorer

        class _S(Scorer):
            pass

        s = _S()
        s.kind, s.scale, s.needs_logprobs, s.name = kind, scale, needs_logprobs, name
        s.calls = []

        def score(ctx, branches):
            s.calls.append((ctx, list(branches)))
            if fn is not None:
                return np.asarray([fn(b) for b in branches], dtype=float)
            return np.asarray(values, dtype=float)
        s.score = score
        return s