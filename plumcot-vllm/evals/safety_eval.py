"""Base model vs the same model inside PLuMCoT on HarmBench, StrongREJECT and XSTest.
 
Three stages, each resumable. Run them in one go:
 
    python evals/safety_eval.py all --model Qwen/Qwen2.5-7B-Instruct \\
        --scorer qwen3guard --out runs/qwen7b-qwen3guard
 
or separately (each stage loads different models, so separate processes keep
GPU memory clean):
 
    python evals/safety_eval.py generate --model ... --scorer ... --out RUN
    python evals/safety_eval.py judge    --out RUN --evaluator wildguard
    python evals/safety_eval.py judge    --out RUN --evaluator harmbench   # optional
    python evals/safety_eval.py report   --out RUN
 
Outputs in RUN/: generations.jsonl, judgments_<evaluator>.jsonl,
summary.json, summary.csv and report.html.
 
Both conditions use the same model, prompt formatting and base decoding
settings (temperature, top-k, top-p, max tokens). The base condition samples
from the same policy as PLuMCoT with a Trigger that never fires, generated in
one batch for speed.
"""
from __future__ import annotations
 
import argparse
import csv
import json
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List
 
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
 
from evals.benchmarks import BENCHMARKS, SUBSETS, load_benchmarks  # noqa: E402
from evals.evaluators import DEFAULT_MODELS, EVALUATORS, run_evaluator  # noqa: E402
from evals.stats import summarize  # noqa: E402
 
SCORER_DEFAULT_MODELS = {
    "qwen3guard": "Qwen/Qwen3Guard-Gen-0.6B",
    "llamaguard": "meta-llama/Llama-Guard-3-1B",
    "composite": "Qwen/Qwen3Guard-Gen-0.6B",
    "policy": None,  # None = the generator judges itself
}
 
 
# ---------------------------------------------------------------------------
# jsonl helpers
# ---------------------------------------------------------------------------
 
def read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
 
 
def append_jsonl(path: Path, rows: List[dict]) -> None:
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
 
 
def _progress(it, total: int, desc: str):
    try:
        from tqdm import tqdm
        return tqdm(it, total=total, desc=desc)
    except ImportError:
        def gen():
            for i, x in enumerate(it, 1):
                if i == 1 or i % 10 == 0 or i == total:
                    print(f"{desc}: {i}/{total}", flush=True)
                yield x
        return gen()
 
 
# ---------------------------------------------------------------------------
# stage 1: generate
# ---------------------------------------------------------------------------
 
def build_config(args):
    from plumcot import DecodingConfig, PLuMCoTConfig
 
    cfg = PLuMCoTConfig(
        decoding=DecodingConfig(temperature=args.temperature, top_k=args.top_k,
                                top_p=args.top_p),
        max_new_tokens=args.max_new_tokens, seed=args.seed,
    )
    cfg.validate()
    return cfg
 
 
def build_trigger(args):
    from plumcot import EntropyTrigger
 
    return EntropyTrigger(threshold=args.threshold)
 
 
def build_stage_parts(args) -> dict:
    """Every StageBundle member except the scorer, which needs the judge
    model loaded. The Filter matches the base decoding's top-k and top-p."""
    from plumcot import BlendSelect, RolloutLookahead, SampleAggregator, TopKTopPFilter
 
    return dict(
        filter=TopKTopPFilter(),
        lookahead=RolloutLookahead(max_tokens=args.lookahead_tokens,
                                   num_samples=args.lookahead_samples,
                                   temperature=args.lookahead_temp),
        aggregate=SampleAggregator(mode=args.aggregate, temperature=args.aggregate_temp),
        select=BlendSelect(judge_weight=args.judge_weight),
    )
 
 
def describe_setup(cfg, trigger, parts: dict) -> dict:
    """Same shape as PLuMCoT.describe(), without the scorer (recorded
    separately as run_config["scorer"] / ["scorer_model"])."""
    return {"loop": cfg.to_dict(), "trigger": trigger.to_dict(),
            "configure": {"type": "FixedConfigure",
                          "options": [{k: v.to_dict() for k, v in parts.items()}]}}
 
 
def build_scorer(name: str, judge):
    from plumcot import (CompositeScorer, LogLikelihoodScorer, llama_guard_scorer,
                         policy_judge_scorer, qwen3guard_scorer)
 
    if name == "qwen3guard":
        return qwen3guard_scorer(judge)
    if name == "llamaguard":
        return llama_guard_scorer(judge)
    if name == "policy":
        return policy_judge_scorer(judge)
    if name == "composite":
        return CompositeScorer([(qwen3guard_scorer(judge), 1.0), (LogLikelihoodScorer(), 0.5)])
    raise ValueError(name)
 
 
def cmd_generate(args) -> None:
    from vllm import LLM
 
    from plumcot import (PLuMCoT, StageBundle, ValidationContext, VLLMBackend,
                         resolve_num_logprobs)
 
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = build_config(args)
    trigger = build_trigger(args)
    parts = build_stage_parts(args)
    scorer_model = args.scorer_model or SCORER_DEFAULT_MODELS[args.scorer]
    run_config = {
        "model": args.model, "chat": not args.no_chat, "system": args.system,
        "scorer": args.scorer, "scorer_model": scorer_model or args.model,
        "benchmarks": args.benchmarks, "limit": args.limit, "seed": args.seed,
        "harmbench_split": args.harmbench_split, "strongreject_small": args.strongreject_small,
        "plumcot_config": describe_setup(cfg, trigger, parts),
    }
    cfg_path = out / "run_config.json"
    if cfg_path.exists():
        old = json.loads(cfg_path.read_text())
        for key in ("model", "scorer", "scorer_model", "plumcot_config"):
            if old.get(key) != run_config[key]:
                raise SystemExit(
                    f"{out} already holds a run with a different {key!r}. Use a new --out "
                    "directory, or delete the old run to start over.")
    cfg_path.write_text(json.dumps(run_config, indent=2))
 
    items = load_benchmarks(args.benchmarks, limit=args.limit, seed=args.seed,
                            harmbench_split=args.harmbench_split,
                            strongreject_small=args.strongreject_small,
                            data_dir=args.data_dir)
    gen_path = out / "generations.jsonl"
    done = {(r["id"], r["condition"]) for r in read_jsonl(gen_path)}
    print(f"{len(items)} prompts; {len(done)} generations already on disk.")
 
    num_logprobs = resolve_num_logprobs(args.num_logprobs, cfg, [trigger, *parts.values()])
    # Check the scorer-independent stages before spending minutes loading models
    # (PLuMCoT repeats this, with the scorer, once the models are up).
    pre = ValidationContext(cfg, num_logprobs, backend=None)
    for stage in [trigger, *parts.values()]:
        stage.validate(pre)
    separate = scorer_model is not None and scorer_model != args.model
    gen_llm = LLM(args.model, max_logprobs=num_logprobs, enable_prefix_caching=True,
                  gpu_memory_utilization=args.gpu_mem or (0.5 if separate else 0.85))
    gen = VLLMBackend(gen_llm, num_logprobs=num_logprobs, model_name=args.model)
    judge = gen
    if separate:
        judge_llm = LLM(scorer_model, max_logprobs=20, enable_prefix_caching=True,
                        gpu_memory_utilization=args.scorer_gpu_mem)
        judge = VLLMBackend(judge_llm, num_logprobs=20, model_name=scorer_model)
    engine = PLuMCoT(gen, trigger=trigger,
                     stages=StageBundle(build_scorer(args.scorer, judge), **parts), config=cfg)
    tok = gen.tokenizer
 
    def record(it, cond, ids, seconds, stats=None):
        return {**it.to_dict(), "condition": cond,
                "response": tok.decode(ids, skip_special_tokens=True),
                "n_tokens": len(ids), "seconds": seconds, "plumcot_stats": stats}
 
    prompt_ids = {it.id: engine.build_prompt(it.prompt, chat=not args.no_chat,
                                             system=args.system) for it in items}
 
    if "base" in args.conditions:
        todo = [it for it in items if (it.id, "base") not in done]
        if todo:
            print(f"base: generating {len(todo)} responses in one batch")
            dec = cfg.decoding
            t0 = time.time()
            outs = gen.generate([prompt_ids[it.id] for it in todo], n=1,
                                max_tokens=cfg.max_new_tokens, temperature=dec.temperature,
                                top_k=dec.top_k, top_p=dec.top_p, seed=cfg.seed)
            per = (time.time() - t0) / len(todo)
            append_jsonl(gen_path, [record(it, "base", o[0].token_ids, per)
                                    for it, o in zip(todo, outs)])
 
    if "plumcot" in args.conditions:
        todo = [it for it in items if (it.id, "plumcot") not in done]
        for it in _progress(todo, len(todo), "plumcot"):
            t0 = time.time()
            res = engine.generate(it.prompt, prompt_ids=prompt_ids[it.id], keep_trace=False)
            append_jsonl(gen_path, [record(it, "plumcot", res.response_ids,
                                           time.time() - t0, res.stats)])
    print(f"generations written to {gen_path}")
 
 
# ---------------------------------------------------------------------------
# stage 2: judge
# ---------------------------------------------------------------------------
 
def cmd_judge(args) -> None:
    out = Path(args.out)
    gens = read_jsonl(out / "generations.jsonl")
    if not gens:
        raise SystemExit(f"No generations in {out}; run the generate stage first.")
    run_config = json.loads((out / "run_config.json").read_text())
    ev_model = args.evaluator_model or DEFAULT_MODELS.get(args.evaluator)
    if ev_model and ev_model == run_config.get("scorer_model"):
        warnings.warn(
            f"The evaluator ({ev_model}) is the model PLuMCoT scored with. Results will "
            "overstate the safety gain, because PLuMCoT optimized against this judge.")
    path = out / f"judgments_{args.evaluator}.jsonl"
    done = {(r["id"], r["condition"]) for r in read_jsonl(path)}
    todo = [g for g in gens if (g["id"], g["condition"]) not in done]
    if args.evaluator == "harmbench":
        todo = [g for g in todo if g["benchmark"] == "harmbench"]
    if not todo:
        print("Nothing new to judge.")
        return
    print(f"{args.evaluator}: judging {len(todo)} responses")
    labels = run_evaluator(args.evaluator, todo, model=ev_model, gpu_mem=args.gpu_mem,
                           max_response_tokens=args.max_response_tokens)
    append_jsonl(path, labels)
    unparsed = sum(1 for r in labels
                   if any(v is None for k, v in r.items() if k in ("harmful", "refusal",
                                                                   "harmful_hb")))
    print(f"wrote {len(labels)} judgments to {path} ({unparsed} with an unparsed label)")
 
 
# ---------------------------------------------------------------------------
# stage 3: report
# ---------------------------------------------------------------------------
 
def merge_labels(out: Path) -> List[dict]:
    """Attach each generation's labels. Harm comes from WildGuard; refusal from
    WildGuard, else the keyword check; harmful_hb from the HarmBench classifier."""
    gens = read_jsonl(out / "generations.jsonl")
    judg: Dict[str, Dict[tuple, dict]] = {
        ev: {(r["id"], r["condition"]): r for r in read_jsonl(out / f"judgments_{ev}.jsonl")}
        for ev in EVALUATORS
    }
    for g in gens:
        key = (g["id"], g["condition"])
        wg, kw, hb = (judg[e].get(key, {}) for e in ("wildguard", "keyword", "harmbench"))
        g["labels"] = {
            "harmful": wg.get("harmful"),
            "refusal": wg.get("refusal", kw.get("refusal")),
            "harmful_hb": hb.get("harmful_hb"),
        }
    return gens
 
 
def write_summary_csv(summary: dict, path: Path) -> None:
    rows = []
    groups = dict(summary["subsets"])
    if "pooled_harmful" in summary:
        groups["all_harmful"] = summary["pooled_harmful"]
    for name, g in groups.items():
        for metric, pr in g["paired"].items():
            b = g["conditions"]["base"][metric]
            p = g["conditions"]["plumcot"][metric]
            rows.append({"subset": name, "kind": g["kind"], "metric": metric, "n": pr["n"],
                         "base_rate": b["rate"], "plumcot_rate": p["rate"],
                         "delta": pr["delta"], "delta_lo": pr["lo"], "delta_hi": pr["hi"],
                         "mcnemar_p": pr["p"]})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["subset"])
        w.writeheader()
        w.writerows(rows)
 
 
def cmd_report(args) -> None:
    from evals.report import render_report
 
    out = Path(args.out)
    records = merge_labels(out)
    if not records:
        raise SystemExit(f"No generations in {out}.")
    run_config = json.loads((out / "run_config.json").read_text())
    summary = summarize(records, SUBSETS)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    write_summary_csv(summary, out / "summary.csv")
    html = render_report(summary, records, run_config, title=args.title)
    (out / "report.html").write_text(html, encoding="utf-8")
    print(f"report written to {out / 'report.html'}")
 
 
# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
 
def _num_logprobs(s: str):
    return s if s == "auto" else int(s)
 
 
def generate_options(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("model and scorer")
    g.add_argument("--model", required=True, help="generator (HF id or path)")
    g.add_argument("--no-chat", action="store_true", help="raw prompts, no chat template")
    g.add_argument("--system", default=None, help="optional system prompt (both conditions)")
    g.add_argument("--scorer", default="qwen3guard", choices=sorted(SCORER_DEFAULT_MODELS))
    g.add_argument("--scorer-model", default=None,
                   help="guard/judge model PLuMCoT scores with (default depends on --scorer)")
    g.add_argument("--gpu-mem", type=float, default=None)
    g.add_argument("--scorer-gpu-mem", type=float, default=0.3)
    g = p.add_argument_group("benchmarks")
    g.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS), choices=BENCHMARKS)
    g.add_argument("--limit", type=int, default=None, help="max prompts per subset")
    g.add_argument("--harmbench-split", default="test", choices=["test", "all"])
    g.add_argument("--strongreject-small", action="store_true")
    g.add_argument("--data-dir", default=None, help="local copies of the benchmark CSVs")
    g.add_argument("--conditions", nargs="+", default=["base", "plumcot"],
                   choices=["base", "plumcot"])
    g = p.add_argument_group("base decoding (shared by both conditions)")
    g.add_argument("--max-new-tokens", type=int, default=512)
    g.add_argument("--temperature", type=float, default=1.0,
                   help="0 (greedy) requires --judge-weight 1")
    g.add_argument("--top-k", type=int, default=50)
    g.add_argument("--top-p", type=float, default=0.9)
    g.add_argument("--seed", type=int, default=0)
    g = p.add_argument_group("PLuMCoT")
    g.add_argument("--threshold", type=float, default=0.15)
    g.add_argument("--lookahead-tokens", type=int, default=16)
    g.add_argument("--lookahead-samples", type=int, default=1,
                   help="more than 1 requires --lookahead-temp > 0")
    g.add_argument("--lookahead-temp", type=float, default=0.0)
    g.add_argument("--judge-weight", type=float, default=0.5,
                   help="w in P(c) ∝ p(c)^(1-w)·exp(w·score); 0 = plain decoding, 1 = score only")
    g.add_argument("--aggregate", default=None, choices=["mean", "max", "min", "logmeanexp"],
                   help="combine lookahead samples per candidate (default: scorer's kind)")
    g.add_argument("--aggregate-temp", type=float, default=1.0,
                   help="tau for logmeanexp; negative = pessimistic soft-min")
    g.add_argument("--num-logprobs", type=_num_logprobs, default="auto",
                   help='"auto" (what the stages need), -1 (full vocabulary), or an int')
 
 
def _forward(ns: argparse.Namespace, parser: argparse.ArgumentParser) -> List[str]:
    """Rebuild the command-line flags a sub-stage needs from parsed args."""
    argv = []
    for a in parser._actions:
        if not a.option_strings or a.dest == "help":
            continue
        val = getattr(ns, a.dest, a.default)
        if val == a.default or val is None:
            continue
        flag = a.option_strings[-1]
        if isinstance(a, argparse._StoreTrueAction):
            argv.append(flag)
        elif isinstance(val, list):
            argv += [flag, *map(str, val)]
        else:
            argv += [flag, str(val)]
    return argv
 
 
def cmd_all(args, gen_parser) -> None:
    me = [sys.executable, str(Path(__file__).resolve())]
    steps = [me + ["generate"] + _forward(args, gen_parser)]
    for ev in args.evaluators:
        steps.append(me + ["judge", "--out", args.out, "--evaluator", ev,
                           "--gpu-mem", str(args.evaluator_gpu_mem)])
    steps.append(me + ["report", "--out", args.out])
    for cmd in steps:
        print("\n$ " + " ".join(cmd[1:]), flush=True)
        subprocess.run(cmd, check=True)
 
 
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)
 
    p_gen = sub.add_parser("generate", help="generate base and PLuMCoT responses")
    p_gen.add_argument("--out", required=True)
    generate_options(p_gen)
 
    p_judge = sub.add_parser("judge", help="label responses with an evaluator model")
    p_judge.add_argument("--out", required=True)
    p_judge.add_argument("--evaluator", default="wildguard", choices=EVALUATORS)
    p_judge.add_argument("--evaluator-model", default=None)
    p_judge.add_argument("--gpu-mem", type=float, default=0.85)
    p_judge.add_argument("--max-response-tokens", type=int, default=2048)
 
    p_rep = sub.add_parser("report", help="compute statistics and write report.html")
    p_rep.add_argument("--out", required=True)
    p_rep.add_argument("--title", default=None)
 
    p_all = sub.add_parser("all", help="generate, judge, and report")
    p_all.add_argument("--out", required=True)
    generate_options(p_all)
    p_all.add_argument("--evaluators", nargs="+", default=["wildguard"], choices=EVALUATORS)
    p_all.add_argument("--evaluator-gpu-mem", type=float, default=0.85)
 
    args = ap.parse_args(argv)
    if args.stage == "generate":
        cmd_generate(args)
    elif args.stage == "judge":
        cmd_judge(args)
    elif args.stage == "report":
        cmd_report(args)
    else:
        cmd_all(args, p_gen)
 
 
if __name__ == "__main__":
    main()
