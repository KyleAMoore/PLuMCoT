"""Benchmark loaders. Each returns a list of `Item`s.
 
Sources are the benchmarks' own GitHub releases, downloaded once and cached.
Pass `--data-dir` to use local copies of the same CSVs instead.
 
  HarmBench    centerforaisafety/HarmBench, text behaviors (test split by
               default). Copyright behaviors are excluded: HarmBench scores
               them with a hashing check, not a harm classifier. Contextual
               behaviors are prompted as HarmBench's direct-request baseline
               does: "{context}\\n\\n---\\n\\n{behavior}".
  StrongREJECT alexandrasouly/strongreject (313 forbidden prompts; 60 with
               --strongreject-small).
  XSTest       paul-rottger/exaggerated-safety, 250 safe prompts (over-refusal
               test) and 200 unsafe contrast prompts.
"""
from __future__ import annotations
 
import csv
import io
import random
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence
 
SOURCES = {
    "harmbench_test": "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/"
                      "data/behavior_datasets/harmbench_behaviors_text_test.csv",
    "harmbench_all": "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/"
                     "data/behavior_datasets/harmbench_behaviors_text_all.csv",
    "strongreject": "https://raw.githubusercontent.com/alexandrasouly/strongreject/main/"
                    "strongreject_dataset/strongreject_dataset.csv",
    "strongreject_small": "https://raw.githubusercontent.com/alexandrasouly/strongreject/main/"
                          "strongreject_dataset/strongreject_small_dataset.csv",
    "xstest": "https://raw.githubusercontent.com/paul-rottger/exaggerated-safety/main/"
              "xstest_prompts.csv",
}
 
BENCHMARKS = ("harmbench", "strongreject", "xstest")
 
# Subsets used in the report. "harmful" subsets: the goal is no harmful
# responses (refusing is fine). "benign" subsets: the goal is no refusals.
SUBSETS = {
    "harmbench": "harmful",
    "strongreject": "harmful",
    "xstest_unsafe": "harmful",
    "xstest_safe": "benign",
}
 
 
@dataclass
class Item:
    id: str                 # unique across benchmarks, e.g. "xstest:17"
    benchmark: str          # harmbench | strongreject | xstest
    subset: str             # a key of SUBSETS
    prompt: str             # the user message sent to the model
    meta: Dict[str, str] = field(default_factory=dict)
 
    def to_dict(self) -> dict:
        return asdict(self)
 
 
def _read_csv(key: str, data_dir: Optional[Path], cache_dir: Path) -> List[dict]:
    fname = SOURCES[key].rsplit("/", 1)[-1]
    if data_dir is not None:
        path = data_dir / fname
        if not path.exists():
            raise FileNotFoundError(f"{path} not found (expected a copy of {SOURCES[key]})")
        text = path.read_text(encoding="utf-8")
    else:
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / fname
        if not path.exists():
            with urllib.request.urlopen(SOURCES[key], timeout=60) as r:
                path.write_bytes(r.read())
        text = path.read_text(encoding="utf-8")
    return list(csv.DictReader(io.StringIO(text)))
 
 
def load_harmbench(rows: List[dict]) -> List[Item]:
    items = []
    for r in rows:
        cat = r["FunctionalCategory"]
        if cat not in ("standard", "contextual"):
            continue
        ctx = (r.get("ContextString") or "").strip()
        prompt = f"{ctx}\n\n---\n\n{r['Behavior']}" if ctx else r["Behavior"]
        items.append(Item(
            id=f"harmbench:{r['BehaviorID']}", benchmark="harmbench", subset="harmbench",
            prompt=prompt,
            meta={"behavior": r["Behavior"], "context": ctx, "functional_category": cat,
                  "category": r["SemanticCategory"]},
        ))
    return items
 
 
def load_strongreject(rows: List[dict]) -> List[Item]:
    return [
        Item(id=f"strongreject:{i}", benchmark="strongreject", subset="strongreject",
             prompt=r["forbidden_prompt"], meta={"category": r["category"], "source": r["source"]})
        for i, r in enumerate(rows)
    ]
 
 
def load_xstest(rows: List[dict]) -> List[Item]:
    return [
        Item(id=f"xstest:{r['id']}", benchmark="xstest", subset=f"xstest_{r['label']}",
             prompt=r["prompt"], meta={"category": r["type"], "focus": r["focus"]})
        for r in rows
    ]
 
 
def load_benchmarks(
    names: Sequence[str] = BENCHMARKS,
    limit: Optional[int] = None,
    seed: int = 0,
    harmbench_split: str = "test",
    strongreject_small: bool = False,
    data_dir: Optional[str] = None,
    cache_dir: str = "~/.cache/plumcot_evals",
) -> List[Item]:
    """Load the requested benchmarks. `limit` caps each *subset* (so XSTest
    keeps both safe and unsafe prompts), using a seeded shuffle so a limited
    run still samples across categories."""
    ddir = Path(data_dir).expanduser() if data_dir else None
    cdir = Path(cache_dir).expanduser()
    items: List[Item] = []
    for name in names:
        if name == "harmbench":
            items += load_harmbench(_read_csv(f"harmbench_{harmbench_split}", ddir, cdir))
        elif name == "strongreject":
            key = "strongreject_small" if strongreject_small else "strongreject"
            items += load_strongreject(_read_csv(key, ddir, cdir))
        elif name == "xstest":
            items += load_xstest(_read_csv("xstest", ddir, cdir))
        else:
            raise ValueError(f"unknown benchmark {name!r}; choose from {BENCHMARKS}")
    if limit is not None:
        rng = random.Random(seed)
        by_subset: Dict[str, List[Item]] = {}
        for it in items:
            by_subset.setdefault(it.subset, []).append(it)
        items = []
        for subset_items in by_subset.values():
            subset_items = list(subset_items)
            rng.shuffle(subset_items)
            items += subset_items[:limit]
    return items
