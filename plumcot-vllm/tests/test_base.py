"""plumcot/base.py (the Stage hooks) and the package's public names."""
import json
from dataclasses import dataclass

import plumcot
from plumcot.base import Stage, ValidationContext, jsonable


def test_stage_hooks_default_to_no_ops():
    s = Stage()
    assert s.required_logprobs(None) is None
    assert s.validate(ValidationContext(None, 20, None)) is None
    assert s.to_dict() == {"type": "Stage"}


def test_dataclass_stages_describe_their_fields():
    @dataclass
    class Mine(Stage):
        a: int = 1
        b: tuple = ("x", "y")
        c: frozenset = frozenset({3, 1})

    d = Mine().to_dict()
    assert d == {"type": "Mine", "a": 1, "b": ["x", "y"], "c": [1, 3]}
    json.dumps(d)


def test_jsonable_converts_nested_containers():
    assert jsonable({"k": (1, {2, 0}), "l": [(3,)]}) == {"k": [1, [0, 2]], "l": [[3]]}


def test_public_names_are_importable():
    for name in plumcot.__all__:
        assert hasattr(plumcot, name), name