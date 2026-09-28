"""Loop-level configuration and logprob-count resolution.
 
Only settings that are fixed for a whole generation live here: base decoding
(the "standard generation iteration" of Algorithm 1), the stopping criterion,
chunking and the main-loop seed. Each stage's own settings live on the stage
object itself (plumcot.stages), so Configure can swap whole stages per
triggered step without touching these.
 
Stopping criteria are deliberately not visible to Configure: letting a
per-step choice change when generation ends would open the door to
non-halting behavior.
"""
from __future__ import annotations
 
import math
import warnings
from dataclasses import asdict, dataclass, field
from typing import Iterable, Literal, Optional, Union
 
# -- how many logprobs the backend requests per position ---------------------
# vLLM already gives None ("no logprobs") and -1 ("full vocabulary") special
# meanings, so "as many as the stages need" uses a named sentinel instead.
AUTO = "auto"
FULL_VOCAB = -1
MIN_RECOMMENDED_LOGPROBS = 20  # also vLLM's default max_logprobs
 
NumLogprobs = Union[int, Literal["auto"]]
 
 
@dataclass
class DecodingConfig:
    """The base policy: how untriggered tokens are sampled.
 
    temperature: 0 is greedy. See BlendSelect for why temperature 0 is only
        allowed together with judge_weight = 1.
    top_k: None disables top-k (no other "off" value is accepted).
    top_p: in (0, 1]; 1 disables top-p. Applied after temperature, over the
        renormalized top-k, as vLLM does.
    """
    temperature: float = 1.0
    top_k: Optional[int] = 50
    top_p: float = 0.9
 
    def validate(self) -> None:
        if not (math.isfinite(self.temperature) and self.temperature >= 0.0):
            raise ValueError(f"decoding.temperature must be finite and >= 0; got {self.temperature}")
        if self.top_k is not None and (isinstance(self.top_k, bool)
                                       or not isinstance(self.top_k, int) or self.top_k < 1):
            raise ValueError(f"decoding.top_k must be None (disabled) or an int >= 1; "
                             f"got {self.top_k!r}")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"decoding.top_p must be in (0, 1]; got {self.top_p}")
 
 
@dataclass
class PLuMCoTConfig:
    """Settings held constant across every generation step.
 
    max_new_tokens: stopping criterion (with the backend's stop tokens).
    chunk_size: untriggered tokens are decoded in chunks, and each chunk is
        truncated at the first position where the Trigger fires. This is
        exact: the kept tokens were sampled from the base policy either way.
    seed: seeds the main loop's random stream (chunk sampling, lookahead
        seeds and Select's draws). None means nondeterministic.
    """
    decoding: DecodingConfig = field(default_factory=DecodingConfig)
    max_new_tokens: int = 512
    chunk_size: int = 32
    seed: Optional[int] = None
 
    def validate(self) -> None:
        if not isinstance(self.decoding, DecodingConfig):
            raise TypeError("config.decoding must be a DecodingConfig.")
        self.decoding.validate()
        for name in ("max_new_tokens", "chunk_size"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ValueError(f"config.{name} must be an int >= 1; got {v!r}")
        if self.seed is not None and (isinstance(self.seed, bool) or not isinstance(self.seed, int)):
            raise ValueError(f"config.seed must be None or an int; got {self.seed!r}")
 
    def to_dict(self) -> dict:
        return asdict(self)
 
 
def resolve_num_logprobs(num_logprobs: NumLogprobs, config: PLuMCoTConfig,
                         stages: Iterable[object]) -> int:
    """Turn a NumLogprobs setting into the int passed to vLLM.
 
    Every stage may declare, through `required_logprobs(config)`, how many
    top logprobs per position it needs (None = no requirement, -1 = the full
    vocabulary).
 
      "auto" -> the largest declared requirement (-1 if any stage needs the
                full vocabulary); an error if no stage declares one.
      -1     -> the full vocabulary.
      n >= 1 -> n, which must cover every declared requirement.
 
    Pass the result as `LLM(max_logprobs=...)` too.
    """
    reqs = []
    for s in stages:
        r = s.required_logprobs(config)
        if r is not None:
            reqs.append((type(s).__name__, int(r)))
    if num_logprobs == AUTO:
        if not reqs:
            raise ValueError(
                'num_logprobs="auto" uses the largest logprob count any stage requires, '
                "but no stage declares one (for example, a Filter with top-k disabled). "
                "Set num_logprobs to an int, or -1 for the full vocabulary."
            )
        values = [r for _, r in reqs]
        return FULL_VOCAB if FULL_VOCAB in values else max(values)
    if isinstance(num_logprobs, bool) or not isinstance(num_logprobs, int) or (
        num_logprobs < 1 and num_logprobs != FULL_VOCAB
    ):
        raise ValueError(f'num_logprobs must be "auto", -1, or >= 1; got {num_logprobs!r}')
    if num_logprobs != FULL_VOCAB:
        for name, r in reqs:
            if r == FULL_VOCAB or r > num_logprobs:
                need = "the full vocabulary (-1)" if r == FULL_VOCAB else str(r)
                raise ValueError(
                    f"{name} needs {need} logprobs per position but num_logprobs="
                    f"{num_logprobs}; raise num_logprobs (and vLLM's max_logprobs) or "
                    "change the stage."
                )
    return num_logprobs
 
 
def warn_if_few_logprobs(num_logprobs: int) -> None:
    if num_logprobs != FULL_VOCAB and num_logprobs < MIN_RECOMMENDED_LOGPROBS:
        warnings.warn(
            f"num_logprobs={num_logprobs} is below {MIN_RECOMMENDED_LOGPROBS}. The "
            "entropy Trigger and entropy-based scores see only the top "
            f"{num_logprobs} tokens, which can make PLuMCoT's behavior, and its "
            "improvement over base decoding, less reliable.",
            stacklevel=3,
        )
