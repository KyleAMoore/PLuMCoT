"""Shared base for every pluggable stage (Trigger, Filter, Lookahead, Scorer,
aggregator, Select)."""
from __future__ import annotations
 
from dataclasses import dataclass, fields, is_dataclass
from typing import TYPE_CHECKING, Optional
 
if TYPE_CHECKING:
    from .backend import Backend
    from .config import PLuMCoTConfig
 
 
@dataclass
class ValidationContext:
    """What a stage may check itself against when PLuMCoT is constructed."""
    config: "PLuMCoTConfig"
    num_logprobs: int        # resolved per-position logprob count (-1 = full vocabulary)
    backend: "Backend"       # the generator backend
 
 
def jsonable(v):
    if isinstance(v, tuple):
        return [jsonable(x) for x in v]
    if isinstance(v, (set, frozenset)):
        return sorted(jsonable(x) for x in v)
    if isinstance(v, list):
        return [jsonable(x) for x in v]
    if isinstance(v, dict):
        return {k: jsonable(x) for k, x in v.items()}
    return v
 
 
class Stage:
    """Hooks every stage shares. All have no-op defaults.
 
    required_logprobs(config): how many top logprobs per position this stage
        needs from the generator (None = no requirement, -1 = full
        vocabulary). Used to resolve num_logprobs="auto" and to reject an
        explicit num_logprobs that is too small.
    validate(ctx): raise (or warn) at PLuMCoT construction if this stage
        cannot work with the configuration, so problems surface at startup
        rather than mid-generation.
    to_dict(): a JSON-serializable description, recorded with eval runs.
        Dataclass stages list their fields automatically.
    """
 
    def required_logprobs(self, config: "PLuMCoTConfig") -> Optional[int]:
        return None
 
    def validate(self, ctx: ValidationContext) -> None:
        return None
 
    def to_dict(self) -> dict:
        d = {"type": type(self).__name__}
        if is_dataclass(self):
            d.update({f.name: jsonable(getattr(self, f.name)) for f in fields(self)})
        return d
