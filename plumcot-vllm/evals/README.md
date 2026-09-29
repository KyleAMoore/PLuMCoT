## Safety evaluation

`evals/safety_eval.py` asks whether a model inside PLuMCoT produces fewer
harmful responses than the same model decoding normally, without refusing
more safe prompts.

```
python evals/safety_eval.py all --model Qwen/Qwen2.5-7B-Instruct \
    --scorer qwen3guard --limit 50 --out runs/qwen7b
# open runs/qwen7b/report.html
```

**Benchmarks** (downloaded from their GitHub releases and cached):

| Subset                                         | Prompts | Goal                 |
| ---------------------------------------------- | ------- | -------------------- |
| HarmBench (test split, standard + contextual)  | 240     | no harmful responses |
| StrongREJECT (`--strongreject-small` for 60) | 313     | no harmful responses |
| XSTest unsafe                                  | 200     | no harmful responses |
| XSTest safe                                    | 250     | no refusals          |

**Conditions.** Both use the same model, chat template and sampling settings
(temperature 1, top-k 50, top-p 0.9 by default). "Normal decoding" is PLuMCoT
with the Trigger off, batched.

**Evaluator.** WildGuard (`allenai/wildguard`, gated) labels every response as
harmful or not and refusal or not. `--evaluators wildguard harmbench` adds
HarmBench's official classifier for HarmBench prompts. `keyword` is a
refusal-only check with no model. The judge stage warns if the evaluator is
the same model PLuMCoT scored with, since that would grade PLuMCoT against
the judge it optimized for.

**Stages.** `generate` → `judge` → `report`, each resumable (reruns skip work
already on disk). `all` runs them as separate processes so each stage's
models are freed before the next loads.

**Report.** `report.html` is one self-contained file: headline changes, a
safety-vs-over-refusal chart, outcome shares per benchmark, paired outcome
transitions, a statistics table (Wilson intervals, paired-bootstrap
intervals on differences, McNemar exact tests), cost, and examples of prompts
whose outcome changed. `summary.json` and `summary.csv` hold the same numbers.

**Runtime.** PLuMCoT generates one prompt at a time, so the full ~1,000
prompts take hours. Start with `--limit 50` (per subset).

`python tests/synthetic_run.py DIR` followed by
`python evals/safety_eval.py report --out DIR` builds a report from synthetic
labels, to preview the layout without a GPU.
