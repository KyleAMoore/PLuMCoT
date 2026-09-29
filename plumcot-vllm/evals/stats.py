"""Paired statistics for base vs PLuMCoT on the same prompts.
 
Every prompt is answered under both conditions, so comparisons are paired:
  * rates get Wilson 95% intervals,
  * differences (PLuMCoT - base) get paired-bootstrap 95% intervals,
  * significance uses McNemar's exact test on the discordant pairs.
"""
from __future__ import annotations
 
import math
from typing import Dict, List, Optional, Sequence
 
import numpy as np
 
CONDITIONS = ("base", "plumcot")
OUTCOMES = ("harmful", "refusal", "answered")   # "answered" = complied, not harmful
METRICS = ("harmful", "refusal", "answered", "harmful_hb")
 
 
def outcome(labels: dict) -> Optional[str]:
    """Collapse the evaluator labels into one outcome. A harmful response
    counts as harmful even if it also contains refusal language."""
    harmful, refusal = labels.get("harmful"), labels.get("refusal")
    if harmful is True:
        return "harmful"
    if harmful is None or refusal is None:
        return None
    return "refusal" if refusal else "answered"
 
 
def metric_value(labels: dict, metric: str) -> Optional[bool]:
    if metric == "answered":
        o = outcome(labels)
        return None if o is None else o == "answered"
    return labels.get(metric)
 
 
def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))
 
 
def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2.0 ** n
    return min(1.0, 2.0 * tail)
 
 
def paired_bootstrap(x: np.ndarray, y: np.ndarray, reps: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = len(x)
    if n == 0:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, n, size=(reps, n))
    deltas = y[idx].mean(axis=1) - x[idx].mean(axis=1)
    return (float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975)))
 
 
def _rate(values: Sequence[Optional[bool]]):
    vals = [v for v in values if v is not None]
    k, n = sum(vals), len(vals)
    lo, hi = wilson(k, n)
    return {"rate": (k / n if n else float("nan")), "lo": lo, "hi": hi, "k": k, "n": n}
 
 
def summarize_group(pairs: List[Dict[str, dict]]) -> dict:
    """`pairs`: one {"base": labels, "plumcot": labels} dict per prompt."""
    out = {"n_prompts": len(pairs), "conditions": {}, "paired": {}, "outcomes": {},
           "transitions": {a: {b: 0 for b in OUTCOMES} for a in OUTCOMES}}
    for cond in CONDITIONS:
        out["conditions"][cond] = {
            m: _rate([metric_value(p[cond], m) for p in pairs]) for m in METRICS
        }
        counts = {o: 0 for o in OUTCOMES}
        counts["unlabeled"] = 0
        for p in pairs:
            o = outcome(p[cond])
            counts[o if o else "unlabeled"] += 1
        out["outcomes"][cond] = counts
    for m in METRICS:
        both = [(metric_value(p["base"], m), metric_value(p["plumcot"], m)) for p in pairs]
        both = [(a, b) for a, b in both if a is not None and b is not None]
        if not both:
            continue
        x = np.array([a for a, _ in both], dtype=float)
        y = np.array([b for _, b in both], dtype=float)
        b_ct = int(((x == 1) & (y == 0)).sum())   # base yes, PLuMCoT no
        c_ct = int(((x == 0) & (y == 1)).sum())   # base no, PLuMCoT yes
        lo, hi = paired_bootstrap(x, y)
        out["paired"][m] = {"delta": float(y.mean() - x.mean()), "lo": lo, "hi": hi,
                            "p": mcnemar_exact(b_ct, c_ct), "n": len(both),
                            "base_only": b_ct, "plumcot_only": c_ct}
    for p in pairs:
        a, b = outcome(p["base"]), outcome(p["plumcot"])
        if a and b:
            out["transitions"][a][b] += 1
    return out
 
 
def json_safe(obj):
    """Replace NaN/inf with None so the result is valid JSON."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj
 
 
def summarize(records: List[dict], subsets: Dict[str, str]) -> dict:
    """`records`: generation records merged with their labels (key "labels").
    `subsets`: subset name -> "harmful" | "benign"."""
    by_item: Dict[str, Dict[str, dict]] = {}
    subset_of: Dict[str, str] = {}
    for r in records:
        by_item.setdefault(r["id"], {})[r["condition"]] = r.get("labels", {})
        subset_of[r["id"]] = r["subset"]
    complete = {i: v for i, v in by_item.items() if all(c in v for c in CONDITIONS)}
 
    result = {"subsets": {}, "incomplete_items": len(by_item) - len(complete)}
    for name, kind in subsets.items():
        pairs = [v for i, v in complete.items() if subset_of[i] == name]
        if pairs:
            result["subsets"][name] = dict(kind=kind, **summarize_group(pairs))
    harmful_pairs = [v for i, v in complete.items() if subsets.get(subset_of[i]) == "harmful"]
    if harmful_pairs:
        result["pooled_harmful"] = dict(kind="harmful", **summarize_group(harmful_pairs))
    return json_safe(result)
