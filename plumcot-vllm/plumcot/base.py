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
