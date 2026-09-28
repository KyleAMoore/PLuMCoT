# PLuMCoT on vLLM (proof-of-concept)

Sketch of the PLuMCoT loop (dissertation §7.4.2, Algorithm 1) on vLLM, with
pluggable **Score** and **Lookahead** and fixed defaults for the other stages.

```
pip install vllm numpy        # tests need only numpy + pytest
python examples/run_demo.py --scorer loglik
pytest tests/
```

## Layout

| File                     | Contents                                                        |
| ------------------------ | --------------------------------------------------------------- |
| `plumcot/config.py`    | One dataclass per choice point, with defaults                   |
| `plumcot/dist.py`      | Top-N distribution, entropy estimate, top-k/top-p filter        |
| `plumcot/backend.py`   | `VLLMBackend`: token-id-level `generate()` with logprobs    |
| `plumcot/scorers.py`   | Score functions (below)                                         |
| `plumcot/engine.py`    | `PLuMCoT` loop, one method per stage; `calibrate_threshold` |
| `examples/run_demo.py` | CLI demo                                                        |
| `evals/`               | Safety evaluation: base model vs the same model in PLuMCoT      |
| `tests/`               | Fake backend and tests that run without a GPU                   |

## Stage defaults

| Stage     | Default                                                               |
| --------- | --------------------------------------------------------------------- |
| Trigger   | normalized entropy H/log\|V\| ≥ 0.15 (calibrate per model)           |
| Configure | constant;`PLuMCoT.configure()` returns `self.config`              |
| Filter    | top-k = 50, then top-p = 0.9                                          |
| Lookahead | 16 tokens, greedy, 1 sample; stops at EOS                             |
| Select    | sample from p(c)^(1−w) · exp(w · score), w =`judge_weight` = 0.5 |

Untriggered tokens are decoded in chunks of 32 and each chunk is cut at the
first position where the trigger fires, so vLLM is not called once per token.

## How many logprobs per position

`VLLMBackend(num_logprobs=...)` controls how much of each next-token
distribution the Trigger, Filter and entropy-based scorers see:

| Value                 | Meaning                                                         |
| --------------------- | --------------------------------------------------------------- |
| `"top_k"` (default) | match`filter.top_k`; resolved when `PLuMCoT` is constructed |
| `-1`                | full vocabulary (exact entropy, much slower)                    |
| `n >= 1`            | the top`n`, independent of `filter.top_k` (must be ≥ it)   |

A named sentinel is used because vLLM already gives `None` (no logprobs) and
`-1` (full vocabulary) their own meanings. Build the LLM with a matching cap:

```python
cfg = PLuMCoTConfig()
n = resolve_num_logprobs("top_k", cfg)          # 50
llm = LLM(model, max_logprobs=n, enable_prefix_caching=True)
engine = PLuMCoT(VLLMBackend(llm, num_logprobs=n), scorer, cfg)
```

`PLuMCoT` checks the value against the LLM's `max_logprobs` at startup and
warns once, at startup, if it is below 20. Judge backends have no Filter, so
give them an int. Each triggered step's trace records `tail_mass`, the
probability outside the returned tokens, for after-the-fact inspection.

## Scorers

| Scorer                  | Extra model                              | Kind    | Notes                                                            |
| ----------------------- | ---------------------------------------- | ------- | ---------------------------------------------------------------- |
| `LogLikelihoodScorer` | none                                     | utility | Baseline; mean or sum lookahead log-probability                  |
| `SelfCertaintyScorer` | none                                     | utility | Negative entropy (default), max-logprob, or approximate KL(U‖p) |
| `llama_guard_scorer`  | Llama Guard 3/4                          | logprob | score = log P(safe)                                              |
| `qwen3guard_scorer`   | Qwen3Guard-Gen                           | logprob | "controversial" counted as unsafe by default                     |
| `policy_judge_scorer` | any instruct model (the generator works) | logprob | natural-language policy, Yes/No                                  |
| `CompositeScorer`     | —                                       | utility | weighted sum of the above, each times its`scale`               |

Judges are `logprob` with the default `transform="logprob"` and `utility`
with `"prob"` or `"logit"`.

All judges go through `VerdictJudgeScorer`, which finds the first label token
in the judge's greedy output and returns P(pos)/(P(pos)+P(neg)) from the
logprobs at that position.

To add a scorer, subclass `Scorer`, set `needs_logprobs` and `kind`, and
return one float per `Branch` (NaN means "no information").

## Select: the model/judge trade-off

Select samples candidate c with probability

    P(c) ∝ p(c)^(1−w) · exp(w · score(c)),   w = SelectConfig.judge_weight

This is a log-linear pool of the model and the judge, the minimizer of
(1−w)·KL(q‖p) + w·KL(q‖q_S) with q_S ∝ exp(score). Equivalently, it is a
Boltzmann choice over the blended utility (1−w)·log p(c) + w·score(c), which
weights the token's immediate plausibility against its lookahead's value.

| w   | Behavior                                                                                                  |
| --- | --------------------------------------------------------------------------------------------------------- |
| 0   | plain decoding over the filtered candidates                                                               |
| 0.5 | with a`logprob` scorer: exact Bayesian MAP in argmax mode; the posterior at temperature 2 when sampling |
| 1   | score only, uniform prior over the candidates (APriCoT-style "agnostic")                                  |

With a `logprob` scorer (score = log P(ok | c)), the rule is a tempered Bayesian
update, [p(c) · P(ok | c)^k]^(1−w) with evidence weight k = w/(1−w).

Two settings make the scores mean the same thing across scorers:

* **`Scorer.scale`**: nats per unit of score, applied before aggregation.
  Log-probabilities are already in nats. Calibrate utilities on other scales
  once per scorer; don't normalize within a candidate group, which turns
  near-ties into noise.
* **Aggregation across lookahead samples** (`LookaheadConfig.aggregate`,
  default `None`) follows the scorer's kind: `logmeanexp` at τ = 1 for
  `logprob` scorers (the log of the mean probability, i.e. log V̂) and `mean`
  for `utility` scorers. An explicit setting overrides it.
  `aggregate_temperature` τ sets risk attitude for `logmeanexp`: τ > 0 is
  optimistic (toward `max`), τ < 0 pessimistic (toward `min`), and |τ| → ∞
  gives the mean.

Command-line flags: `--judge-weight`, `--aggregate`, `--aggregate-temp`
(in `examples/run_demo.py` and `evals/safety_eval.py`).

## Known limitations

* **Entropy is estimated** unless `num_logprobs=-1`. With a top-N slice, the
  tail's contribution is bounded rather than exact (`TriggerConfig.tail`).
  Recommended to pick the threshold with `calibrate_threshold` rather
  than trusting the default.
* **Judge label logprobs.** vLLM 0.30's `SamplingParams.logprob_token_ids`
  returns logprobs for chosen token ids. Passing the label tokens there would
  make `VerdictJudgeScorer` exact even when a label falls outside the top-N.
* **Guard models see short prefixes.** They were trained on complete
  responses; may result in more false negatives early in a response.
* **Filter limits alignment.** Select can only choose among admitted
  candidates; an unaligned base model may admit no refusal-like token.
* **One prompt at a time.** Batching across prompts would need the async engine.
* **Model IDs and APIs** were checked against vLLM 0.30.0's source (no GPU run);
  confirm the guard-model IDs on Hugging Face.
