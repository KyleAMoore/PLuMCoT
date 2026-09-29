"""Secondary: the test helpers in fake_backend.py behave like the backend
contract the primary tests rely on."""
import numpy as np
import pytest

from fake_backend import EOS, HARM, V, FakeBackend, FakeGuard, _logprobs, _raw_logits


def test_distributions_are_deterministic_per_context():
    a = _logprobs(_raw_logits([1, 2, 3]))
    b = _logprobs(_raw_logits([9, 1, 2, 3]))          # only the last 3 tokens matter
    assert np.array_equal(a, b)
    assert np.exp(a).sum() == pytest.approx(1.0)


@pytest.mark.parametrize("n", [5, 20, -1])
def test_dist_slice_size_and_sampled_token(n):
    be = FakeBackend(num_logprobs=n)
    (c,) = be.generate([[1, 2]], max_tokens=6, temperature=1.0, logprobs="dist", seed=0)[0]
    for tid, d in zip(c.token_ids, c.dists):
        assert tid in set(int(t) for t in d.token_ids)
        assert len(d.token_ids) in ((V,) if n == -1 else (n, n + 1))


def test_greedy_is_argmax():
    be = FakeBackend()
    (c,) = be.generate([[1, 2]], max_tokens=5, temperature=0.0, logprobs="chosen")[0]
    ctx = [1, 2]
    for t in c.token_ids:
        assert t == int(np.argmax(_logprobs(_raw_logits(ctx))))
        ctx.append(t)


def test_sampling_stays_in_the_tempered_top_k_top_p_support():
    be = FakeBackend()
    lps = _logprobs(_raw_logits([4, 5]))
    order = np.argsort(-lps)[:6]
    p = np.exp(lps[order] / 0.5)
    p /= p.sum()
    support = set(order[:int(np.searchsorted(np.cumsum(p), 0.8 - 1e-9)) + 1])
    outs = be.generate([[4, 5]], n=300, max_tokens=1, temperature=0.5, top_k=6, top_p=0.8, seed=0)
    assert {o.token_ids[0] for o in outs[0]} <= support


def test_logprob_modes_and_stop():
    be = FakeBackend()
    none = be.generate([[1]], max_tokens=3, temperature=1.0, seed=0)[0][0]
    assert none.token_logprobs is None and none.dists is None
    chosen = be.generate([[1]], max_tokens=3, temperature=1.0, logprobs="chosen", seed=0)[0][0]
    assert len(chosen.token_logprobs) == len(chosen.token_ids) and chosen.dists is None
    # sample until some continuation hits EOS: it must end there, marked "stop"
    outs = be.generate([[1]], n=200, max_tokens=40, temperature=2.0, seed=1)[0]
    stopped = [o for o in outs if o.finish_reason == "stop"]
    assert stopped and all(o.token_ids[-1] == EOS and EOS not in o.token_ids[:-1]
                           for o in stopped)
    assert all(len(o.token_ids) == 40 for o in outs if o.finish_reason == "length")


def test_seeded_generation_is_reproducible():
    be = FakeBackend()
    kw = dict(max_tokens=10, temperature=1.0, seed=5)
    assert be.generate([[1]], **kw)[0][0].token_ids == be.generate([[1]], **kw)[0][0].token_ids


def test_fake_guard_flags_the_harm_token():
    g = FakeGuard()
    g.last_texts = [f"a w{HARM} b", "harmless"]
    unsafe, safe = g.generate([[0], [0]])
    assert unsafe[0].token_ids[-1] == 2 and safe[0].token_ids[-1] == 1
    assert np.exp(unsafe[0].dists[1].logprobs).max() == pytest.approx(0.95)
    assert g.last_texts == []