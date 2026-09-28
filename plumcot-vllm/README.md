# PLuMCoT on vLLM (proof-of-concept)
 
Sketch of the PLuMCoT loop (dissertation §7.4.2, Algorithm 1) on vLLM. Each
stage is a pluggable object; the engine only sequences them.
 
```
pip install vllm numpy        # tests need only numpy + pytest
python examples/run_demo.py --scorer loglik
pytest tests/
```
 
## Layout
 
| File | Contents |
|---|---|
| `plumcot/config.py` | Loop-level settings (`PLuMCoTConfig`, `DecodingConfig`); logprob-count resolution |
| `plumcot/stages.py` | Stage base classes and default implementations; `StageBundle`, `Configure` |
| `plumcot/scorers.py` | Score functions (below) and NaN handling |
| `plumcot/engine.py` | `PLuMCoT`: the loop |
| `plumcot/dist.py` | Top-N distribution, entropy estimate, top-k/top-p filter |
| `plumcot/backend.py` | `VLLMBackend`: token-id-level `generate()` with logprobs |
| `plumcot/base.py` | `Stage` hooks shared by every stage |
| `examples/run_demo.py` | CLI demo |
| `evals/` | Safety evaluation: base model vs the same model in PLuMCoT |
| `tests/` | Fake backend and tests that run without a GPU |
 
## Building an engine
 
```python
from plumcot import *
 
cfg = PLuMCoTConfig(decoding=DecodingConfig(temperature=1.0, top_k=50, top_p=0.9),
                    max_new_tokens=512, seed=0)
trigger = EntropyTrigger(threshold=0.15)
stages = StageBundle(
    scorer=qwen3guard_scorer(judge),
    filter=TopKTopPFilter(),                       # top-k/top-p of the base decoding
    lookahead=RolloutLookahead(max_tokens=16),
    aggregate=SampleAggregator(),                  # by the scorer's kind
    select=BlendSelect(judge_weight=0.5),
)
n = resolve_num_logprobs("auto", cfg, [trigger, *stages.stages()])   # 50
llm = LLM(model, max_logprobs=n, enable_prefix_caching=True)
engine = PLuMCoT(VLLMBackend(llm, num_logprobs=n), trigger=trigger, stages=stages, config=cfg)
result = engine.generate("How do I ...?")
```
 
Unspecified `StageBundle` members use their defaults. `engine.describe()`
returns a JSON-serializable description of the loop and every stage.
 
## The loop and its stages
 
| Stage | Where it is set | Default |
|---|---|---|
| Base decoding | `PLuMCoTConfig.decoding` | temperature 1, top-k 50, top-p 0.9 |
| Stopping | `PLuMCoTConfig.max_new_tokens` + stop tokens | 512 |
| 1 Trigger | `PLuMCoT(trigger=...)` | `EntropyTrigger`: raw normalized entropy H/log\|V\| ≥ 0.15 |
| 1.5 Configure | `PLuMCoT(stages=...)` | a single `StageBundle` (`FixedConfigure`) |
| 2 Filter | bundle | `TopKTopPFilter`: the base decoding's top-k, then top-p |
| 3 Lookahead | bundle | `RolloutLookahead`: 16 tokens, greedy, 1 sample |
| 4 Score | bundle | a `Scorer`, then `SampleAggregator` |
| 5 Select | bundle | `BlendSelect`: sample from p_T(c)^(1−w) · exp(w · score), w = 0.5 |
 
**Fixed across a generation**: base decoding, the stopping criterion, the
Trigger, chunking and the main-loop seed. Stopping is deliberately invisible
to Configure, so a per-step choice cannot change when generation ends.
 
**Configure** runs only on triggered steps. A `Configure` subclass declares
every bundle it may use up front (`options`) and returns an index from
`choose(step)`; all options are validated, including their logprob
requirements, when `PLuMCoT` is constructed. A `StageBundle` passed directly
is wrapped in `FixedConfigure`.
 
**Chunked decoding.** Untriggered tokens are decoded in chunks of
`chunk_size` (32) and each chunk is cut at the first position where the
Trigger fires, so vLLM is not called once per token. This is exact, which is
why a Trigger may depend only on one position's distribution and the tokens
before it.
 
**Custom stages.** Subclass `Trigger`, `Filter`, `Lookahead`, `Scorer` or
`Select` and implement `__call__` (`score` for scorers). Optional hooks from
`Stage`: `required_logprobs(config)`, `validate(ctx)` and `to_dict()`.
Settings are validated when a stage is constructed; unknown option strings
raise rather than falling back to a default.
 
## How many logprobs per position
 
`VLLMBackend(num_logprobs=...)` controls how much of each next-token
distribution the Trigger, Filter and entropy-based scorers see:
 
| Value | Meaning |
|---|---|
| `"auto"` (default) | the largest count any stage declares (the default Filter's top-k); resolved when `PLuMCoT` is constructed |
| `-1` | full vocabulary (exact entropy, much slower) |
| `n >= 1` | the top `n`; must cover every stage's requirement |
 
A named sentinel is used because vLLM already gives `None` (no logprobs) and
`-1` (full vocabulary) their own meanings. `PLuMCoT` resolves `"auto"` on the
backend in place, checks the value against the LLM's `max_logprobs`, and
warns if it is below 20. Judge backends are not resolved, so give them an
int. Each triggered step's trace records `tail_mass`, the probability outside
the returned tokens.
 
## Scorers
 
| Scorer | Extra model | Kind | Notes |
|---|---|---|---|
| `LogLikelihoodScorer` | none | utility | Baseline; mean or sum lookahead log-probability |
| `SelfCertaintyScorer` | none | utility | Negative entropy (default), max-logprob, or approximate KL(U‖p) |
| `llama_guard_scorer` | Llama Guard 3/4 | logprob | score = log P(safe) |
| `qwen3guard_scorer` | Qwen3Guard-Gen | logprob | "controversial" counted as unsafe by default |
| `policy_judge_scorer` | any instruct model (the generator works) | logprob | natural-language policy, Yes/No |
| `CompositeScorer` | — | utility | weighted sum of the above, each times its `scale` |
 
Judges are `logprob` with the default `transform="logprob"` and `utility`
with `"prob"` or `"logit"`.
 
All judges go through `VerdictJudgeScorer`, which finds the first label token
in the judge's greedy output and returns P(pos)/(P(pos)+P(neg)) from the
logprobs at that position. An unparsed verdict is NaN by default
(`on_unparsed="raise"` makes it an error).
 
**Missing scores.** A scorer returns NaN for "no information" (e.g. an empty
lookahead after a stop-token candidate). NaNs are filled with a neutral value
from the other branches at the same step, averaged by the scorer's kind: the
log of the mean probability for `logprob` scorers, the arithmetic mean for
utilities. If every branch is NaN, the scores become 0 and Select uses the
prior alone; `SampleAggregator(on_all_nan=...)` warns (default), raises, or
ignores this.
 
## Select: the model/judge trade-off
 
`BlendSelect` samples candidate c with probability
 
    P(c) ∝ p_T(c)^(1−w) · exp(w · score(c)),   w = judge_weight
 
where p_T is the candidate's probability under base decoding (decoding
temperature T applied), renormalized over the Filter's pool. The softmax is
at temperature 1. This is a log-linear pool of the model and the judge, the
minimizer of (1−w)·KL(q‖p_T) + w·KL(q‖q_S) with q_S ∝ exp(score).
 
| w | Behavior |
|---|---|
| 0 | exactly base decoding over the filtered candidates |
| 0.5 | with a `logprob` scorer: exact Bayesian MAP in argmax mode; the posterior at temperature 2 when sampling |
| 1 | score only, uniform prior over the candidates (APriCoT-style "agnostic") |
 
With a `logprob` scorer (score = log P(ok | c)), the rule is a tempered Bayesian
update, [p_T(c) · P(ok | c)^k]^(1−w) with evidence weight k = w/(1−w).
 
**Temperature.** The Filter's pool and Select's prior both follow the base
decoding temperature, so a triggered step sees the same policy as an
untriggered one. The Trigger's entropy always uses the raw (T = 1)
distribution. At T = 0 (greedy) the prior is a point mass on the top token,
so `BlendSelect` requires w = 1 there (and raises otherwise); the pool is
then computed on the raw distribution.
 
Two settings make the scores mean the same thing across scorers:
 
* **`Scorer.scale`**: nats per unit of score, applied before aggregation.
  Log-probabilities are already in nats. Calibrate utilities on other scales
  once per scorer; don't normalize within a candidate group, which turns
  near-ties into noise.
* **Aggregation across lookahead samples** (`SampleAggregator(mode=None)`)
  follows the scorer's kind: `logmeanexp` at τ = 1 for `logprob` scorers
  (the log of the mean probability, i.e. log V̂) and `mean` for `utility`
  scorers. An explicit mode overrides it. `temperature` τ sets risk attitude
  for `logmeanexp`: τ > 0 is optimistic (toward `max`), τ < 0 pessimistic
  (toward `min`), and |τ| → ∞ gives the mean.
Command-line flags: `--judge-weight`, `--aggregate`, `--aggregate-temp`,
`--temperature`, `--top-k`, `--top-p` (in `examples/run_demo.py` and
`evals/safety_eval.py`).
 
## Trace and stats
 
`PLuMCoTResult.trace` has one `StepRecord` per triggered step, including
steps where the Filter left a single candidate (`searched=False`, committed
without lookahead). `stats` counts `fired` (Trigger fired), `searched`
(lookahead ran), `single_candidate`, `chunks`, `branches` and
`lookahead_tokens`.
 
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
 
| Subset | Prompts | Goal |
|---|---|---|
| HarmBench (test split, standard + contextual) | 240 | no harmful responses |
| StrongREJECT (`--strongreject-small` for 60) | 313 | no harmful responses |
| XSTest unsafe | 200 | no harmful responses |
| XSTest safe | 250 | no refusals |
 
**Conditions.** Both use the same model, chat template and base decoding
settings (temperature 1, top-k 50, top-p 0.9 by default). "Normal decoding"
samples from the same policy as PLuMCoT with a Trigger that never fires,
batched.
 
**Evaluator.** WildGuard (`allenai/wildguard`, gated) labels every response as
harmful or not and refusal or not. `--evaluators wildguard harmbench` adds
HarmBench's official classifier for HarmBench prompts. `keyword` is a
refusal-only check with no model. The judge stage warns if the evaluator is
the same model PLuMCoT scored with, since that would grade PLuMCoT against
the judge it optimized for.
 
**Stages.** `generate` → `judge` → `report`, each resumable (reruns skip work
already on disk). `all` runs them as separate processes so each stage's
models are freed before the next loads. A run directory records the full
stage description; resuming with different settings is refused. Runs made
before the stage refactor record a different description, so start them in
a new directory.
 
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
 
## Known limitations
 
* **Entropy is estimated** unless `num_logprobs=-1`. With a top-N slice, the
  tail's contribution is bounded rather than exact (`EntropyTrigger.tail`).
  Pick the threshold with `calibrate_entropy_threshold` rather than trusting
  the default.
* **Judge label logprobs.** vLLM 0.30's `SamplingParams.logprob_token_ids`
  returns logprobs for chosen token ids. Passing the label tokens there would
  make `VerdictJudgeScorer` exact even when a label falls outside the top-N.
* **Guard models see short prefixes.** They were trained on complete
  responses; expect false negatives early in a response.
* **Filter limits alignment.** Select can only choose among admitted
  candidates; an unaligned base model may admit no refusal-like token.
* **One prompt at a time.** Batching across prompts would need the async engine.
* **Model IDs and APIs** were checked against vLLM 0.30.0's source (no GPU run);
  confirm the guard-model IDs on Hugging Face.
