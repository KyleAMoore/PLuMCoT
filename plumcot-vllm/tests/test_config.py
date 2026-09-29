"""plumcot/config.py: loop-level settings and logprob-count resolution."""
import json
import math

import pytest

from plumcot import AUTO, FULL_VOCAB, DecodingConfig, PLuMCoTConfig, resolve_num_logprobs
from plumcot.base import Stage
from plumcot.config import MIN_RECOMMENDED_LOGPROBS, warn_if_few_logprobs


class Needs(Stage):
    """A stage declaring a fixed logprob requirement."""

    def __init__(self, n):
        self.n = n

    def required_logprobs(self, config):
        return self.n


# -- DecodingConfig / PLuMCoTConfig validation ----------------------------------------

def test_defaults_are_valid():
    cfg = PLuMCoTConfig()
    cfg.validate()
    assert (cfg.decoding.temperature, cfg.decoding.top_k, cfg.decoding.top_p) == (1.0, 50, 0.9)
    assert (cfg.max_new_tokens, cfg.chunk_size, cfg.seed) == (512, 32, None)


@pytest.mark.parametrize("kw, match", [
    (dict(temperature=-0.1), "temperature"),
    (dict(temperature=math.nan), "temperature"),
    (dict(temperature=math.inf), "temperature"),
    (dict(top_k=0), "top_k"),
    (dict(top_k=-1), "top_k"),
    (dict(top_k=True), "top_k"),
    (dict(top_k=5.0), "top_k"),
    (dict(top_p=0.0), "top_p"),
    (dict(top_p=1.01), "top_p"),
])
def test_decoding_config_rejects_bad_values(kw, match):
    with pytest.raises(ValueError, match=match):
        DecodingConfig(**kw).validate()


def test_decoding_config_accepts_edge_values():
    DecodingConfig(temperature=0.0, top_k=None, top_p=1.0).validate()
    DecodingConfig(temperature=2.0, top_k=1, top_p=1e-3).validate()


@pytest.mark.parametrize("kw, match", [
    (dict(max_new_tokens=0), "max_new_tokens"),
    (dict(max_new_tokens=2.0), "max_new_tokens"),
    (dict(chunk_size=0), "chunk_size"),
    (dict(chunk_size=True), "chunk_size"),
    (dict(seed=1.5), "seed"),
    (dict(seed=True), "seed"),
])
def test_loop_config_rejects_bad_values(kw, match):
    with pytest.raises(ValueError, match=match):
        PLuMCoTConfig(**kw).validate()


def test_loop_config_rejects_non_decoding_config():
    with pytest.raises(TypeError, match="DecodingConfig"):
        PLuMCoTConfig(decoding={"temperature": 1.0}).validate()


def test_loop_config_validates_its_decoding():
    with pytest.raises(ValueError, match="decoding.top_p"):
        PLuMCoTConfig(decoding=DecodingConfig(top_p=2.0)).validate()


def test_to_dict_is_json_serializable():
    d = json.loads(json.dumps(PLuMCoTConfig(seed=3).to_dict()))
    assert d["decoding"]["top_k"] == 50 and d["seed"] == 3


# -- resolve_num_logprobs -----------------------------------------------------------------

def test_auto_takes_the_largest_requirement():
    assert resolve_num_logprobs(AUTO, PLuMCoTConfig(), [Needs(10), Needs(None), Needs(30)]) == 30


def test_auto_is_full_vocab_if_any_stage_needs_it():
    assert resolve_num_logprobs(AUTO, PLuMCoTConfig(), [Needs(10), Needs(FULL_VOCAB)]) == FULL_VOCAB


def test_auto_without_any_requirement_raises():
    with pytest.raises(ValueError, match="no stage declares one"):
        resolve_num_logprobs(AUTO, PLuMCoTConfig(), [Needs(None), Stage()])


def test_explicit_values_are_kept():
    assert resolve_num_logprobs(40, PLuMCoTConfig(), [Needs(10)]) == 40
    assert resolve_num_logprobs(FULL_VOCAB, PLuMCoTConfig(), [Needs(10), Needs(FULL_VOCAB)]) == -1
    assert resolve_num_logprobs(7, PLuMCoTConfig(), []) == 7


def test_explicit_value_below_a_requirement_names_the_stage():
    with pytest.raises(ValueError, match="Needs needs 30 logprobs .* num_logprobs=20"):
        resolve_num_logprobs(20, PLuMCoTConfig(), [Needs(30)])


def test_explicit_value_when_a_stage_needs_full_vocab_raises():
    with pytest.raises(ValueError, match="full vocabulary"):
        resolve_num_logprobs(100, PLuMCoTConfig(), [Needs(FULL_VOCAB)])


@pytest.mark.parametrize("bad", [0, -2, None, "all", "top_k", True, 3.0])
def test_invalid_num_logprobs_rejected(bad):
    with pytest.raises(ValueError):
        resolve_num_logprobs(bad, PLuMCoTConfig(), [Needs(10)])


# -- warning for small logprob counts -------------------------------------------------------

def test_warns_below_recommended_count():
    with pytest.warns(UserWarning, match="less reliable"):
        warn_if_few_logprobs(MIN_RECOMMENDED_LOGPROBS - 1)


@pytest.mark.parametrize("n", [MIN_RECOMMENDED_LOGPROBS, 100, FULL_VOCAB])
def test_no_warning_at_or_above_recommended_count(n, recwarn):
    warn_if_few_logprobs(n)
    assert not recwarn.list