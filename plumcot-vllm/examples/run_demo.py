"""Run PLuMCoT on one prompt with a chosen scorer.
 
Examples
--------
# Baselines using only the generator
python examples/run_demo.py --scorer loglik
python examples/run_demo.py --scorer certainty --lookahead-samples 3 --lookahead-temp 1.0
 
# Alignment compliance with a guard model on a base (non-chat) generator
python examples/run_demo.py --model Qwen/Qwen2.5-1.5B --no-chat \
    --scorer qwen3guard --judge-model Qwen/Qwen3Guard-Gen-0.6B
 
# Natural-language policy judged by the generator itself (no extra model)
python examples/run_demo.py --scorer policy
 
# Calibrate the trigger threshold to fire on ~10% of tokens
python examples/run_demo.py --calibrate 0.1
"""
from __future__ import annotations
 
import argparse
import sys
from pathlib import Path
 
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
 
import numpy as np
 
from plumcot import (BlendSelect, CompositeScorer, DecodingConfig, EntropyTrigger,
                     LogLikelihoodScorer, PLuMCoT, PLuMCoTConfig, RolloutLookahead,
                     SampleAggregator, SelfCertaintyScorer, StageBundle, TopKTopPFilter,
                     VLLMBackend, calibrate_entropy_threshold, llama_guard_scorer,
                     policy_judge_scorer, qwen3guard_scorer, resolve_num_logprobs)
 
JUDGE_DEFAULTS = {
    "llamaguard": "meta-llama/Llama-Guard-3-1B",
    "qwen3guard": "Qwen/Qwen3Guard-Gen-0.6B",
    "guard+loglik": "Qwen/Qwen3Guard-Gen-0.6B",
}
 
 
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--prompt", default="Explain why the sky is blue in two sentences.")
    ap.add_argument("--no-chat", action="store_true", help="raw prompt, no chat template")
    ap.add_argument("--scorer", default="loglik",
                    choices=["loglik", "certainty", "llamaguard", "qwen3guard", "policy",
                             "guard+loglik"])
    ap.add_argument("--judge-model", default=None,
                    help="judge model; default depends on --scorer ('policy' uses the generator)")
    ap.add_argument("--lookahead-tokens", type=int, default=16)
    ap.add_argument("--lookahead-samples", type=int, default=1,
                    help="more than 1 requires --lookahead-temp > 0")
    ap.add_argument("--lookahead-temp", type=float, default=0.0)
    ap.add_argument("--threshold", type=float, default=0.15)
    ap.add_argument("--judge-weight", type=float, default=0.5,
                    help="w in P(c) ∝ p(c)^(1-w)·exp(w·score); 0 = plain decoding, 1 = score only")
    ap.add_argument("--aggregate", default=None, choices=["mean", "max", "min", "logmeanexp"],
                    help="combine lookahead samples per candidate (default: scorer's kind)")
    ap.add_argument("--aggregate-temp", type=float, default=1.0,
                    help="tau for logmeanexp; negative = pessimistic soft-min")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="base decoding temperature; 0 (greedy) requires --judge-weight 1")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--num-logprobs", type=_num_logprobs, default="auto",
                    help='"auto" (what the stages need), -1 (full vocabulary), or an int')
    ap.add_argument("--gpu-mem", type=float, default=None)
    ap.add_argument("--dtype", default="auto",
                    help='model dtype for vLLM, e.g. "half" on GPUs without bfloat16 (T4)')
    ap.add_argument("--max-model-len", type=int, default=None,
                    help="cap the context length (and so the KV cache vLLM reserves)")
    ap.add_argument("--calibrate", type=float, default=None, metavar="RATE")
    return ap.parse_args()
 
 
def _num_logprobs(s: str):
    return s if s == "auto" else int(s)
 
 
def main():
    args = parse_args()
    from vllm import LLM
 
    cfg = PLuMCoTConfig(
        decoding=DecodingConfig(temperature=args.temperature, top_k=args.top_k,
                                top_p=args.top_p),
        max_new_tokens=args.max_new_tokens, seed=args.seed,
    )
    cfg.validate()
    trigger = EntropyTrigger(threshold=args.threshold)
    parts = dict(
        filter=TopKTopPFilter(),   # the base decoding's top-k and top-p
        lookahead=RolloutLookahead(max_tokens=args.lookahead_tokens,
                                   num_samples=args.lookahead_samples,
                                   temperature=args.lookahead_temp),
        aggregate=SampleAggregator(mode=args.aggregate, temperature=args.aggregate_temp),
        select=BlendSelect(judge_weight=args.judge_weight),
    )
    num_logprobs = resolve_num_logprobs(args.num_logprobs, cfg, [trigger, *parts.values()])
 
    judge_name = args.judge_model or JUDGE_DEFAULTS.get(args.scorer)
    separate_judge = judge_name is not None and judge_name != args.model
    gen_mem = args.gpu_mem or (0.45 if separate_judge else 0.85)
 
    llm_kw = dict(dtype=args.dtype, max_model_len=args.max_model_len)
    gen_llm = LLM(args.model, max_logprobs=num_logprobs,
                  enable_prefix_caching=True, gpu_memory_utilization=gen_mem, **llm_kw)
    gen = VLLMBackend(gen_llm, num_logprobs=num_logprobs, model_name=args.model)
 
    if args.calibrate is not None:
        prompt_ids = PLuMCoT(gen, trigger=trigger, stages=StageBundle(LogLikelihoodScorer(), **parts),
                             config=cfg).build_prompt(args.prompt, chat=not args.no_chat)
        thr, hs = calibrate_entropy_threshold(gen, [prompt_ids] * 8, target_rate=args.calibrate,
                                              decoding=cfg.decoding, tail=trigger.tail)
        print(f"{hs.size} tokens; threshold for {args.calibrate:.0%} firing rate: {thr:.4f}")
        qs = np.quantile(hs, [0.5, 0.8, 0.9, 0.95, 0.99])
        print("quantiles 50/80/90/95/99:", [round(float(x), 4) for x in qs])
        return
 
    judge = gen
    if separate_judge:
        # Two vLLM engines can share one process; if this fails on your
        # setup, serve the judge separately and wrap it in your own Backend.
        judge_llm = LLM(judge_name, max_logprobs=20, enable_prefix_caching=True,
                        gpu_memory_utilization=0.4, **llm_kw)
        judge = VLLMBackend(judge_llm, num_logprobs=20, model_name=judge_name)
 
    if args.scorer == "loglik":
        scorer = LogLikelihoodScorer()
    elif args.scorer == "certainty":
        scorer = SelfCertaintyScorer()
    elif args.scorer == "llamaguard":
        scorer = llama_guard_scorer(judge)
    elif args.scorer == "qwen3guard":
        scorer = qwen3guard_scorer(judge)
    elif args.scorer == "policy":
        scorer = policy_judge_scorer(judge)
    else:
        scorer = CompositeScorer([(qwen3guard_scorer(judge), 1.0), (LogLikelihoodScorer(), 0.5)])
 
    engine = PLuMCoT(gen, trigger=trigger, stages=StageBundle(scorer, **parts), config=cfg)
    result = engine.generate(args.prompt, chat=not args.no_chat)
 
    print("=" * 80)
    print(result.text)
    print("=" * 80)
    print("stats:", result.stats)
    for step in [s for s in result.trace if s.searched][:10]:
        top = sorted(range(len(step.candidate_ids)), key=lambda i: -step.select_probs[i])[:5]
        h = step.trigger_info.get("norm_entropy", float("nan"))
        print(f"\n@{step.position}  H/log|V|={h:.3f}  "
              f"tail={step.tail_mass:.3f}  chose {step.candidate_texts[step.chosen_index]!r}")
        for i in top:
            print(f"   {step.candidate_texts[i]!r:>14}  prior={step.prior_logprobs[i]:6.2f}  "
                  f"score={step.scores[i]:7.3f}  P_sel={step.select_probs[i]:.3f}  "
                  f"-> {step.lookahead_texts[i][0]!r}")
    if hasattr(scorer, "n_failures"):
        print(f"\njudge calls={scorer.n_calls}  unparsed verdicts={scorer.n_failures}")
 
 
if __name__ == "__main__":
    main()
