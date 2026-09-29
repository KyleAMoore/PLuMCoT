"""Write a synthetic run directory (generations + WildGuard-style judgments)
so the report and statistics can be exercised without a GPU.
 
    python tests/synthetic_run.py OUT_DIR [--data-dir CSV_DIR] [--limit N]
 
Uses real benchmark prompts when the CSVs are available, with placeholder
responses. The run is marked synthetic and the report says so.
"""
from __future__ import annotations
 
import argparse
import json
import random
import sys
from pathlib import Path
 
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
 
from evals.benchmarks import Item, load_benchmarks  # noqa: E402
from evals.safety_eval import append_jsonl  # noqa: E402
 
# P(outcome) under normal decoding, and PLuMCoT transition probabilities.
BASE = {
    "harmbench": {"harmful": 0.34, "refusal": 0.56, "answered": 0.10},
    "strongreject": {"harmful": 0.22, "refusal": 0.70, "answered": 0.08},
    "xstest_unsafe": {"harmful": 0.14, "refusal": 0.72, "answered": 0.14},
    "xstest_safe": {"harmful": 0.00, "refusal": 0.09, "answered": 0.91},
}
MOVE = {  # base outcome -> distribution under PLuMCoT
    "harmful": {"harmful": 0.45, "refusal": 0.30, "answered": 0.25},
    "refusal": {"harmful": 0.02, "refusal": 0.90, "answered": 0.08},
    "answered": {"harmful": 0.02, "refusal": 0.04, "answered": 0.94},
}
TEXT = {
    "harmful": "[synthetic] Sure, here is how you could do that: ...",
    "refusal": "[synthetic] I can't help with that request.",
    "answered": "[synthetic] Here's some general, safe information on this topic ...",
}
LABELS = {
    "harmful": {"harmful": True, "refusal": False},
    "refusal": {"harmful": False, "refusal": True},
    "answered": {"harmful": False, "refusal": False},
}
# Same shape as PLuMCoT.describe() (abridged).
PLUMCOT_CONFIG = {
    "loop": {"decoding": {"temperature": 1.0, "top_k": 50, "top_p": 0.9},
             "max_new_tokens": 512, "chunk_size": 32, "seed": 0},
    "trigger": {"type": "EntropyTrigger", "threshold": 0.15, "tail": "lumped"},
    "configure": {"type": "FixedConfigure", "options": [{
        "lookahead": {"type": "RolloutLookahead", "max_tokens": 16, "num_samples": 1},
        "select": {"type": "BlendSelect", "judge_weight": 0.5, "mode": "sample"},
    }]},
}
 
 
def _draw(rng, dist):
    r, acc = rng.random(), 0.0
    for k, p in dist.items():
        acc += p
        if r < acc:
            return k
    return k
 
 
def fake_items(limit: int):
    items = []
    for subset in BASE:
        bench = "xstest" if subset.startswith("xstest") else subset
        for i in range(limit):
            items.append(Item(id=f"{bench}:{subset}-{i}", benchmark=bench, subset=subset,
                              prompt=f"Synthetic {subset} prompt #{i}",
                              meta={"behavior": "synthetic", "context": ""}))
    return items
 
 
def write_synthetic_run(out: Path, data_dir=None, limit: int = 60, seed: int = 0) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    try:
        items = load_benchmarks(limit=limit, seed=seed, data_dir=data_dir)
    except Exception:
        items = fake_items(limit)
    gens, judg = [], []
    for it in items:
        ob = _draw(rng, BASE[it.subset])
        op = _draw(rng, MOVE[ob]) if it.subset != "xstest_safe" else _draw(rng, {
            "refusal": 0.35, "answered": 0.65} if ob == "refusal" else {
            "refusal": 0.03, "answered": 0.97})
        for cond, o in (("base", ob), ("plumcot", op)):
            n_tok = rng.randint(40, 400)
            stats = None
            if cond == "plumcot":
                searched = rng.randint(0, 12)
                single = rng.randint(0, 3)
                stats = {"tokens": n_tok, "chunks": n_tok // 32 + 1,
                         "fired": searched + single, "searched": searched,
                         "single_candidate": single, "branches": searched * 5,
                         "lookahead_tokens": searched * 5 * 16}
            gens.append({**it.to_dict(), "condition": cond, "response": TEXT[o],
                         "n_tokens": n_tok,
                         "seconds": 0.05 if cond == "base" else 4.0 + stats["searched"],
                         "plumcot_stats": stats})
            judg.append({"id": it.id, "condition": cond, "harmful_request": it.subset != "xstest_safe",
                         **LABELS[o], "raw": "synthetic"})
    (out / "generations.jsonl").unlink(missing_ok=True)
    (out / "judgments_wildguard.jsonl").unlink(missing_ok=True)
    append_jsonl(out / "generations.jsonl", gens)
    append_jsonl(out / "judgments_wildguard.jsonl", judg)
    (out / "run_config.json").write_text(json.dumps({
        "model": "synthetic/model", "scorer": "qwen3guard",
        "scorer_model": "Qwen/Qwen3Guard-Gen-0.6B", "synthetic": True,
        "plumcot_config": PLUMCOT_CONFIG,
    }, indent=2))
    return out
 
 
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--limit", type=int, default=60)
    a = ap.parse_args()
    print(write_synthetic_run(Path(a.out), a.data_dir, a.limit))
