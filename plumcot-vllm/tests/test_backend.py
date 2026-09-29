"""plumcot/backend.py: VLLMBackend.

The first part stubs the `vllm` (and `transformers`) modules, so it runs
anywhere. Tests marked `vllm` check, on a real model, the assumptions the
engine makes about vLLM's behavior; they need PLUMCOT_TEST_MODEL (see
conftest.py)."""
import sys
import types
import warnings
from types import SimpleNamespace as NS

import numpy as np
import pytest

from plumcot import AUTO, FULL_VOCAB, DecodingConfig, TokenDist, TopKTopPFilter


# ===========================================================================
# stubbed vllm
# ===========================================================================

class StubTokenizer:
    def __init__(self, eos=0, n=10):
        self.eos_token_id = eos
        self.n = n

    def __len__(self):
        return self.n


class StubLLM:
    """Records generate() calls and replays scripted request outputs."""

    def __init__(self, outputs=None, max_logprobs=20, eos=0, config_at="llm", model="stub/m"):
        self.tok = StubTokenizer(eos)
        self.outputs = outputs
        self.calls = []
        cfg = NS(model=model, max_logprobs=max_logprobs)
        if config_at == "llm":
            self.model_config = cfg
        elif config_at == "engine":
            self.llm_engine = NS(model_config=cfg)

    def get_tokenizer(self):
        return self.tok

    def generate(self, prompts, params, use_tqdm=True):
        self.calls.append(dict(prompts=prompts, params=params, use_tqdm=use_tqdm))
        return self.outputs(prompts) if callable(self.outputs) else self.outputs


def lp(x):
    return NS(logprob=x)


def out(ids, logprobs=None, finish="length"):
    return NS(token_ids=ids, logprobs=logprobs, finish_reason=finish)


@pytest.fixture
def stub_vllm(monkeypatch):
    """Installs fake vllm and transformers modules; returns a handle whose
    `eos` attribute sets what GenerationConfig reports (an Exception raises)."""
    vllm = types.ModuleType("vllm")
    vllm.SamplingParams = lambda **kw: dict(kw)
    inputs = types.ModuleType("vllm.inputs")
    inputs.TokensPrompt = lambda prompt_token_ids: ("TP", tuple(prompt_token_ids))
    handle = NS(eos=None, loaded=[])

    class GenerationConfig:
        @staticmethod
        def from_pretrained(name):
            handle.loaded.append(name)
            if isinstance(handle.eos, Exception):
                raise handle.eos
            return NS(eos_token_id=handle.eos)

    tr = types.ModuleType("transformers")
    tr.GenerationConfig = GenerationConfig
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.inputs", inputs)
    monkeypatch.setitem(sys.modules, "transformers", tr)
    return handle


def backend(llm, **kw):
    from plumcot import VLLMBackend

    kw.setdefault("num_logprobs", 20)
    return VLLMBackend(llm, **kw)


# -- construction and stop tokens -----------------------------------------------------------

def test_basic_attributes(stub_vllm):
    b = backend(StubLLM(eos=3), stop_token_ids=[7, 8])
    assert len(b.tokenizer) == b.vocab_size == 10
    assert b.stop_token_ids == {7, 8} and stub_vllm.loaded == []   # explicit: no lookup


@pytest.mark.parametrize("gen_eos, expect", [(None, {0}), (5, {0, 5}), ([5, 6], {0, 5, 6}),
                                             ((0, 9), {0, 9})])
def test_default_stop_ids_combine_tokenizer_and_generation_config(stub_vllm, gen_eos, expect):
    stub_vllm.eos = gen_eos
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert backend(StubLLM(eos=0)).stop_token_ids == expect
    assert stub_vllm.loaded == ["stub/m"]                 # model name from the model config


def test_default_stop_ids_prefer_an_explicit_model_name(stub_vllm):
    backend(StubLLM(), model_name="other/name")
    assert stub_vllm.loaded == ["other/name"]


def test_generation_config_failure_warns_and_keeps_eos(stub_vllm):
    stub_vllm.eos = OSError("no file")
    with pytest.warns(UserWarning, match="generation_config.json .*OSError"):
        b = backend(StubLLM(eos=2))
    assert b.stop_token_ids == {2}


def test_no_stop_ids_at_all_raises(stub_vllm):
    with pytest.raises(ValueError, match="No stop token ids"):
        backend(StubLLM(eos=None))


# -- model config lookup and logprob cap ------------------------------------------------------

@pytest.mark.parametrize("where", ["llm", "engine"])
def test_model_config_found_in_either_location(stub_vllm, where):
    from plumcot.backend import find_model_config

    llm = StubLLM(config_at=where, max_logprobs=20)
    assert find_model_config(llm).max_logprobs == 20
    b = backend(llm, num_logprobs=50, stop_token_ids=[0])
    with pytest.raises(ValueError, match="max_logprobs=50"):
        b.check_logprob_cap()


def test_missing_model_config_warns_instead_of_checking(stub_vllm):
    b = backend(StubLLM(config_at=None), stop_token_ids=[0])
    with pytest.warns(UserWarning, match="cannot be checked"):
        b.check_logprob_cap()


def test_check_logprob_cap_cases():
    from plumcot.backend import VLLMBackend

    def fake(n, cap):
        return NS(llm=NS(model_config=NS(max_logprobs=cap)), num_logprobs=n)

    VLLMBackend.check_logprob_cap(fake(50, 50))
    VLLMBackend.check_logprob_cap(fake(FULL_VOCAB, FULL_VOCAB))
    VLLMBackend.check_logprob_cap(fake(50, FULL_VOCAB))
    with pytest.raises(ValueError, match="max_logprobs=50"):
        VLLMBackend.check_logprob_cap(fake(50, 20))
    with pytest.raises(ValueError):
        VLLMBackend.check_logprob_cap(fake(FULL_VOCAB, 64))
    with pytest.raises(ValueError, match="not been resolved"):
        VLLMBackend.check_logprob_cap(fake(AUTO, 64))


# -- generate(): sampling parameters ----------------------------------------------------------

def run(stub_kwargs=None, backend_kwargs=None, **gen_kw):
    llm = StubLLM(outputs=lambda prompts: [NS(outputs=[out([1], [{1: lp(-0.1)}])])
                                           for _ in prompts], **(stub_kwargs or {}))
    b = backend(llm, stop_token_ids=[9, 2], **(backend_kwargs or {}))
    gen_kw.setdefault("max_tokens", 4)
    gen_kw.setdefault("temperature", 0.7)
    res = b.generate([[5, 6], [7]], **gen_kw)
    return llm.calls[0], res


def test_sampling_params_base_fields(stub_vllm):
    call, _ = run(n=2, top_p=0.8, seed=3)
    assert call["params"] == dict(n=2, max_tokens=4, temperature=0.7, top_p=0.8,
                                  stop_token_ids=[2, 9], seed=3)
    assert call["prompts"] == [("TP", (5, 6)), ("TP", (7,))] and call["use_tqdm"] is False


def test_top_k_and_stop_only_when_set(stub_vllm):
    call, _ = run(top_k=None, stop=())
    assert "top_k" not in call["params"] and "stop" not in call["params"]
    call, _ = run(top_k=8, stop=("\n\n",))
    assert call["params"]["top_k"] == 8 and call["params"]["stop"] == ["\n\n"]
    with pytest.raises(ValueError, match="top_k"):
        run(top_k=0)


@pytest.mark.parametrize("mode, expect", [("none", None), ("chosen", 0), ("dist", 20)])
def test_logprob_request_per_mode(stub_vllm, mode, expect):
    call, _ = run(logprobs=mode)
    assert call["params"].get("logprobs") == expect


def test_full_vocab_dist_request(stub_vllm):
    call, _ = run(backend_kwargs=dict(num_logprobs=FULL_VOCAB), logprobs="dist")
    assert call["params"]["logprobs"] == -1


def test_unresolved_auto_with_dist_raises(stub_vllm):
    with pytest.raises(RuntimeError, match="not been resolved"):
        run(backend_kwargs=dict(num_logprobs=AUTO), logprobs="dist")
    run(backend_kwargs=dict(num_logprobs=AUTO), logprobs="chosen")    # chosen does not need it


def test_invalid_logprob_mode_raises(stub_vllm):
    with pytest.raises(ValueError, match="logprobs must be one of"):
        run(logprobs="full")


def test_empty_contexts_do_not_call_vllm(stub_vllm):
    llm = StubLLM()
    assert backend(llm, stop_token_ids=[0]).generate([], max_tokens=1, temperature=1.0) == []
    assert llm.calls == []


# -- generate(): output conversion ------------------------------------------------------------

def convert(outputs, mode, n_logprobs=20):
    llm = StubLLM(outputs=[NS(outputs=outputs)])
    return backend(llm, stop_token_ids=[0], num_logprobs=n_logprobs).generate(
        [[1]], max_tokens=2, temperature=1.0, logprobs=mode)[0]


def test_convert_none_mode_ignores_logprobs(stub_vllm):
    (c,) = convert([out([3, 4], None, "stop")], "none")
    assert (c.token_ids, c.token_logprobs, c.dists, c.finish_reason) == ([3, 4], None, None, "stop")


def test_convert_chosen_and_dist(stub_vllm):
    pos = [{3: lp(-0.1), 5: lp(-2.5)}, {4: lp(-0.3), 6: -1.9}]
    (c,) = convert([out([3, 4], pos)], "chosen")
    assert c.token_logprobs == pytest.approx([-0.1, -0.3]) and c.dists is None
    (c,) = convert([out([3, 4], pos)], "dist")
    assert c.token_logprobs == pytest.approx([-0.1, -0.3])
    assert all(isinstance(d, TokenDist) for d in c.dists)
    assert list(c.dists[1].token_ids) == [4, 6] and c.dists[1].vocab_size == 10


def test_convert_multiple_outputs_per_request(stub_vllm):
    outs = convert([out([3], [{3: lp(-0.1)}]), out([4], [{4: lp(-0.2)}])], "chosen")
    assert [c.token_ids for c in outs] == [[3], [4]]


@pytest.mark.parametrize("logprobs, match", [
    (None, "returned none positions"),
    ([{3: lp(-0.1)}], "returned 1 positions for 2 tokens"),
    ([{3: lp(-0.1)}, {7: lp(-0.2)}], "omitted the sampled token 4"),
    ([{3: lp(-0.1)}, None], "omitted the sampled token 4"),
])
def test_convert_rejects_incomplete_logprobs(stub_vllm, logprobs, match):
    with pytest.raises(RuntimeError, match=match):
        convert([out([3, 4], logprobs)], "chosen")


# ===========================================================================
# real vLLM: the engine's assumptions about vLLM
# ===========================================================================

def _prompt(b):
    return list(b.tokenizer.encode("The capital of France is"))


@pytest.mark.vllm
def test_real_model_config_and_stop_ids(make_vllm_backend):
    b = make_vllm_backend(20)
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # model_config must be found
        b.check_logprob_cap()
    if b.tokenizer.eos_token_id is not None:
        assert b.tokenizer.eos_token_id in b.stop_token_ids


@pytest.mark.vllm
def test_real_logprobs_are_raw_not_tempered(make_vllm_backend):
    """The Trigger and Filter assume vLLM returns the raw distribution
    (logprobs_mode="raw_logprobs"), whatever the sampling temperature."""
    b = make_vllm_backend(20)
    d = [b.generate([_prompt(b)], max_tokens=1, temperature=t, logprobs="dist", seed=0)[0][0]
         .dists[0] for t in (1.0, 0.3)]
    top = {int(t): lp for t, lp in zip(d[0].token_ids, d[0].logprobs)}
    for t, lp_ in zip(d[1].token_ids, d[1].logprobs):
        if int(t) in top:
            assert lp_ == pytest.approx(top[int(t)], abs=1e-3)
    assert np.exp(d[0].logprobs).sum() <= 1.0 + 1e-3


@pytest.mark.vllm
def test_real_sampled_tokens_come_with_logprobs(make_vllm_backend):
    b = make_vllm_backend(20)
    (c,) = b.generate([_prompt(b)], max_tokens=16, temperature=1.0, logprobs="dist", seed=1)[0]
    assert len(c.token_ids) == len(c.token_logprobs) == len(c.dists) > 0
    for tid, lp_, d in zip(c.token_ids, c.token_logprobs, c.dists):
        assert tid in set(int(t) for t in d.token_ids)
        assert np.isfinite(lp_)


@pytest.mark.vllm
def test_real_sampling_support_matches_the_filter_pool(make_vllm_backend):
    """vLLM applies temperature, then top-k, then top-p over the renormalized
    top-k. TopKTopPFilter's pool must be exactly that support: every sample
    falls inside it, and every reasonably likely pool member shows up."""
    b = make_vllm_backend(20)
    dec = DecodingConfig(temperature=0.7, top_k=20, top_p=0.9)
    outs = b.generate([_prompt(b)], n=500, max_tokens=1, temperature=dec.temperature,
                      top_k=dec.top_k, top_p=dec.top_p, logprobs="dist", seed=2)[0]
    pool = TopKTopPFilter()(outs[0].dists[0], dec)
    ids = [int(t) for t in pool.token_ids]
    sampled = [o.token_ids[0] for o in outs]
    assert set(sampled) <= set(ids)
    likely = {t for t, lpr in zip(ids, pool.prior_logprobs) if np.exp(lpr) >= 0.03}
    assert likely <= set(sampled)


@pytest.mark.vllm
def test_real_returns_n_continuations_per_context(make_vllm_backend):
    b = make_vllm_backend(20)
    res = b.generate([_prompt(b), _prompt(b)[:-1]], n=3, max_tokens=2, temperature=1.0, seed=3)
    assert [len(r) for r in res] == [3, 3]


@pytest.mark.vllm
def test_real_seed_is_reproducible(make_vllm_backend):
    b = make_vllm_backend(20)
    kw = dict(max_tokens=12, temperature=1.0, top_k=20, top_p=0.9, seed=4)
    a = b.generate([_prompt(b)], **kw)[0][0].token_ids
    c = b.generate([_prompt(b)], **kw)[0][0].token_ids
    assert a == c