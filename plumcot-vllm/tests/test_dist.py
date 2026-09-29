"""plumcot/dist.py: top-N distributions, entropy estimates, top-k/top-p."""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import dist_from_probs, entropy_of
from plumcot import TokenDist, top_k_top_p
from plumcot.dist import _NEG, log_softmax, logmeanexp, softmax

V = 40


# -- construction ----------------------------------------------------------------------

def test_from_items_sorts_descending_and_stably():
    d = TokenDist.from_items([(5, -2.0), (7, -1.0), (3, -2.0), (9, -0.5)], V)
    assert list(d.token_ids) == [9, 7, 5, 3]          # ties keep input order
    assert list(d.logprobs) == [-0.5, -1.0, -2.0, -2.0]


def test_from_items_maps_neg_inf_to_finite_stand_in():
    d = TokenDist.from_items([(1, -0.1), (2, -math.inf)], V)
    assert d.logprobs[-1] == _NEG and np.isfinite(d.logprobs).all()


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_from_items_rejects_nan_and_pos_inf(bad):
    with pytest.raises(ValueError, match="NaN or \\+inf"):
        TokenDist.from_items([(1, -0.1), (2, bad)], V)


def test_from_vllm_accepts_logprob_objects_and_floats():
    pos = {4: SimpleNamespace(logprob=-0.2), 6: -1.7}
    d = TokenDist.from_vllm(pos, V)
    assert list(d.token_ids) == [4, 6] and d.logprobs == pytest.approx([-0.2, -1.7])


# -- tail mass and entropy -------------------------------------------------------------------

def test_tail_mass_and_clamping():
    assert dist_from_probs([0.5, 0.3]).tail_mass() == pytest.approx(0.2)
    # rounding can push the listed mass slightly over 1; the tail is clamped at 0
    d = TokenDist(np.array([0, 1]), np.log(np.array([0.6, 0.4 + 1e-12])), V)
    assert d.tail_mass() == 0.0


def test_entropy_exact_with_full_vocabulary():
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(V))
    d = dist_from_probs(p)
    assert d.tail_mass() == pytest.approx(0.0, abs=1e-12)
    for tail in ("lumped", "uniform"):
        assert d.entropy(tail) == pytest.approx(entropy_of(p))


def test_entropy_tail_estimates_bracket_the_truth():
    rng = np.random.default_rng(1)
    p = np.sort(rng.dirichlet(np.ones(V) * 0.3))[::-1]
    d = dist_from_probs(p[:8])
    lo, hi, exact = d.entropy("lumped"), d.entropy("uniform"), entropy_of(p)
    assert lo <= exact + 1e-12 <= hi + 2e-12


def test_uniform_tail_on_uniform_distribution_is_exact():
    top = dist_from_probs([1 / V] * 10)
    assert top.entropy("uniform") == pytest.approx(math.log(V))
    assert top.entropy("lumped") < math.log(V)


def test_normalized_entropy_range():
    assert dist_from_probs([1 / V] * V).normalized_entropy() == pytest.approx(1.0)
    assert dist_from_probs([1.0]).normalized_entropy() == pytest.approx(0.0)


def test_unknown_tail_mode_raises_even_without_tail_mass():
    d = dist_from_probs([1 / V] * V)
    with pytest.raises(ValueError, match="unknown tail mode"):
        d.entropy("lump")


def test_max_logprob_is_the_top_entry():
    assert dist_from_probs([0.2, 0.7, 0.1]).max_logprob() == pytest.approx(math.log(0.7))


# -- top_k_top_p ---------------------------------------------------------------------------

@pytest.fixture
def five():
    return dist_from_probs([0.5, 0.3, 0.1, 0.05, 0.05], vocab_size=10)


def test_top_k_then_top_p_over_renormalized_top_k(five):
    ids, lps = top_k_top_p(five, top_k=4, top_p=0.8)
    assert list(ids) == [0, 1]           # 0.5/0.95 + 0.3/0.95 ≥ 0.8
    assert lps == pytest.approx(np.log([0.5, 0.3]))   # raw logprobs are returned
    ids, _ = top_k_top_p(five, top_k=2, top_p=1.0)
    assert list(ids) == [0, 1]


def test_top_k_none_keeps_everything_available(five):
    ids, _ = top_k_top_p(five, top_k=None, top_p=1.0)
    assert list(ids) == [0, 1, 2, 3, 4]
    ids, _ = top_k_top_p(five, top_k=100, top_p=1.0)   # k above what is available
    assert len(ids) == 5


def test_min_candidates_is_kept_but_capped_at_k(five):
    ids, _ = top_k_top_p(five, top_k=5, top_p=0.1, min_candidates=3)
    assert list(ids) == [0, 1, 2]
    ids, _ = top_k_top_p(five, top_k=2, top_p=0.1, min_candidates=4)
    assert list(ids) == [0, 1]


def test_exact_cumulative_boundary_is_included(five):
    # cumulative mass reaches exactly 0.8 at the second token (0.5 + 0.3)
    ids, _ = top_k_top_p(five, top_k=None, top_p=0.8)
    assert list(ids) == [0, 1]


def test_lower_temperature_shrinks_the_pool(five):
    sizes = [len(top_k_top_p(five, None, 0.9, temperature=t)[0]) for t in (2.0, 1.0, 0.5, 0.1)]
    assert sizes == sorted(sizes, reverse=True) and sizes[0] > sizes[-1]


def test_temperature_pool_matches_manual_computation(five):
    t = 0.7
    p = np.array([0.5, 0.3, 0.1, 0.05, 0.05]) ** (1 / t)
    p /= p.sum()
    n = int(np.searchsorted(np.cumsum(p), 0.9 - 1e-9)) + 1
    ids, _ = top_k_top_p(five, None, 0.9, temperature=t)
    assert len(ids) == n


@pytest.mark.parametrize("kw, match", [
    (dict(top_k=0), "top_k"), (dict(top_p=0.0), "top_p"), (dict(top_p=1.5), "top_p"),
    (dict(min_candidates=0), "min_candidates"), (dict(temperature=0.0), "temperature"),
    (dict(temperature=math.nan), "temperature"),
])
def test_top_k_top_p_validates_arguments(five, kw, match):
    args = dict(top_k=5, top_p=0.9)
    args.update(kw)
    with pytest.raises(ValueError, match=match):
        top_k_top_p(five, **args)


# -- numeric helpers ---------------------------------------------------------------------------

def test_softmax_and_log_softmax_agree():
    x = np.array([1.0, -2.0, 0.5, 30.0])
    assert softmax(x).sum() == pytest.approx(1.0)
    assert np.exp(log_softmax(x)) == pytest.approx(softmax(x))
    assert softmax(np.array([0.0, -np.inf]))[1] == 0.0


def test_logmeanexp_matches_definition():
    x = np.array([[0.0, 1.0, 4.0], [-3.0, -3.0, -3.0]])
    assert logmeanexp(x, axis=1) == pytest.approx(np.log(np.exp(x).mean(axis=1)))


def test_logmeanexp_handles_infinities():
    assert logmeanexp(np.array([-np.inf, -np.inf])) == -np.inf
    assert logmeanexp(np.array([-np.inf, 0.0])) == pytest.approx(math.log(0.5))
    assert logmeanexp(np.array([np.inf, 0.0])) == np.inf
    assert logmeanexp(np.array([1e308, 1e308])) == pytest.approx(1e308)   # no overflow