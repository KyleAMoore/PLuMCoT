"""Stage 4: Score.
 
A scorer receives every lookahead sample for every candidate as a flat list of
`Branch`es and returns one float per branch (higher = better). The bundle's
SampleAggregator (plumcot.stages) then fills NaNs, multiplies by
`Scorer.scale`, and combines the samples of each candidate into one score,
which Select weighs against the candidate's prior.
 
Each scorer declares a `kind`, which decides the default aggregation and how
missing scores are filled:
 
  "logprob" - the score is log P(event | branch), e.g. a judge's log P(safe).
              Samples are averaged in probability space (logmeanexp).
  "utility" - any real-valued utility. Samples are averaged arithmetically.
 
Because the score is blended with log p(c), its units matter: `scale` is the
number of nats one unit of score is worth. Log-probability scores are already
in nats (scale = 1); rescale utilities on other scales once, per scorer, rather
than normalizing within each candidate group.
 
A scorer may return NaN for "no information" (for example, a candidate that
ends the response has an empty lookahead, so LogLikelihoodScorer has nothing
to measure). NaNs are replaced by a neutral value computed from the other
branches' scores at the same step; see `fill_nan`. ±inf are kept as given.
"""
from __future__ import annotations
 
import math
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple
 
import numpy as np
 
from .backend import Backend, Continuation, LogprobMode
from .base import Stage, ValidationContext
from .config import AUTO
from .dist import check_tail_mode, logmeanexp
 
_MODE_RANK = {"none": 0, "chosen": 1, "dist": 2}
 
ScoreKind = Literal["logprob", "utility"]
SCORE_KINDS = ("logprob", "utility")
_DEFAULT_AGGREGATE = {"logprob": "logmeanexp", "utility": "mean"}
ALL_NAN_POLICIES = ("warn", "raise", "ignore")
 
 
@dataclass
class ScoreContext:
    user_message: Optional[str]   # raw user text (needed by judges); None if only ids were given
    prompt_ids: List[int]
    response_ids: List[int]       # committed response so far (before the candidate)
    tokenizer: object             # generator tokenizer
 
 
@dataclass
class Branch:
    candidate_id: int
    candidate_logprob: float      # raw model logprob of the candidate
    continuation: Continuation
 
    def response_ids(self, ctx: ScoreContext) -> List[int]:
        return ctx.response_ids + [self.candidate_id] + self.continuation.token_ids
 
 
def _check_scale(scale: float) -> float:
    if not math.isfinite(scale):
        raise ValueError(f"scale must be finite; got {scale}")
    return float(scale)
 
 
def _check_choice(name: str, value, choices) -> None:
    if value not in choices:
        raise ValueError(f"{name} must be one of {tuple(choices)}; got {value!r}")
 
 
class Scorer(Stage):
    name: str = "scorer"
    # What the Lookahead must return per token: nothing, the sampled token's
    # logprob, or the top-N distribution at each position.
    needs_logprobs: LogprobMode = "none"
    # "logprob" or "utility"; sets the default aggregation and NaN fill.
    kind: ScoreKind = "utility"
    # Nats per unit of score, applied by the aggregator after NaN filling.
    scale: float = 1.0
 
    @property
    def default_aggregate(self) -> str:
        return _DEFAULT_AGGREGATE[self.kind]
 
    def score(self, ctx: ScoreContext, branches: Sequence[Branch]) -> np.ndarray:
        raise NotImplementedError
 
    def validate(self, ctx: ValidationContext) -> None:
        _check_choice(f"{type(self).__name__}.kind", self.kind, SCORE_KINDS)
        _check_choice(f"{type(self).__name__}.needs_logprobs", self.needs_logprobs,
                      tuple(_MODE_RANK))
        _check_scale(self.scale)
 
    def to_dict(self) -> dict:
        return {"type": type(self).__name__, "name": self.name, "kind": self.kind,
                "scale": self.scale}
 
 
# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------
 
def fill_nan(x: np.ndarray, kind: ScoreKind) -> Tuple[np.ndarray, bool]:
    """Replace NaN scores with a neutral value; return (filled, all_nan).
 
    The neutral value is the average of the finite scores, averaged the way
    the scorer's kind says scores should be averaged:
      "logprob" - logmeanexp, i.e. the log of the mean probability (an
                  arithmetic mean of log-probabilities would be the log of
                  their geometric mean, biased toward the worst branch);
      "utility" - the arithmetic mean.
    ±inf scores are information, not gaps, and are kept. If every score is
    NaN, the result is all zeros and all_nan is True (the caller decides
    whether that is acceptable). NaNs alongside only ±inf scores raise,
    since no neutral value exists.
    """
    _check_choice("kind", kind, SCORE_KINDS)
    x = np.asarray(x, dtype=np.float64).copy()
    nan = np.isnan(x)
    if nan.all():
        return np.zeros_like(x), True
    if nan.any():
        finite = x[np.isfinite(x)]
        if finite.size == 0:
            raise ValueError("Cannot fill NaN scores: every other score is ±inf, so there "
                             "is no neutral value to use.")
        x[nan] = float(logmeanexp(finite)) if kind == "logprob" else float(finite.mean())
    return x, False
 
 
def handle_all_nan(policy: str, who: str) -> None:
    msg = (f"{who}: every branch scored NaN (no information) at this step, so its "
           "scores were set to 0 and Select falls back to the prior alone.")
    if policy == "raise":
        raise RuntimeError(msg)
    if policy == "warn":
        warnings.warn(msg, stacklevel=3)
 
 
# ---------------------------------------------------------------------------
# 1. Lookahead likelihood (baseline)
# ---------------------------------------------------------------------------
 
class LogLikelihoodScorer(Scorer):
    """Log-probability of the lookahead under the generator itself.
 
    reduce="mean" is length-normalized (average nats per token, i.e. minus
    log-perplexity); averaging log-probabilities is the intended quantity
    here. reduce="sum" is log p(lookahead | ctx, c). With reduce="sum" and
    judge_weight = 0.5, argmax selection tracks the joint probability of
    candidate plus lookahead, i.e. it behaves like a one-token beam search. It
    is included as a baseline: it is mode-seeking and favors repetitive
    continuations.
 
    An empty lookahead (the candidate ends the response) scores NaN, and
    non-finite token logprobs raise, since the backend guarantees them.
 
    kind is "utility": the score is a log-probability of the lookahead text,
    not of an acceptability event, so averaging over rollouts arithmetically
    (expected log-likelihood) is the natural aggregate.
    """
    name = "loglik"
    needs_logprobs = "chosen"
 
    def __init__(self, reduce: Literal["mean", "sum"] = "mean", scale: float = 1.0):
        _check_choice("reduce", reduce, ("mean", "sum"))
        self.reduce = reduce
        self.scale = _check_scale(scale)
 
    def score(self, ctx, branches):
        out = np.full(len(branches), np.nan)
        for i, b in enumerate(branches):
            cont = b.continuation
            if not cont.token_ids:
                continue
            if cont.token_logprobs is None or len(cont.token_logprobs) != len(cont.token_ids):
                raise RuntimeError("LogLikelihoodScorer needs one logprob per lookahead token.")
            arr = np.asarray(cont.token_logprobs, dtype=np.float64)
            if not np.isfinite(arr).all():
                raise RuntimeError("Lookahead token logprobs must be finite.")
            out[i] = arr.mean() if self.reduce == "mean" else arr.sum()
        return out
 
    def to_dict(self):
        return {**super().to_dict(), "reduce": self.reduce}
 
 
# ---------------------------------------------------------------------------
# 2. Self-certainty (intrinsic, no auxiliary model)
# ---------------------------------------------------------------------------
 
class SelfCertaintyScorer(Scorer):
    """How confident the generator is while producing the lookahead.
 
    metric (each averaged over the lookahead's positions):
      "neg_entropy"    - minus the token entropy in nats (default)
      "max_logprob"    - log-probability of the top token. This averages
                         log-probabilities on purpose: it is a per-token
                         confidence, like the entropy.
      "self_certainty" - INTUITOR-style KL(Uniform || p). This needs the full
                         vocabulary; from top-N logprobs with a uniform tail it
                         reduces to about -log(tail mass), so treat it as a
                         rough approximation only.
 
    An empty lookahead scores NaN. Greedy lookaheads are confident by
    construction, which compresses these scores; sampled lookaheads
    (temperature 1, num_samples > 1) discriminate better. Expect this scorer
    to reward the model's priors, including base-rate preferences.
    """
    name = "self_certainty"
    needs_logprobs = "dist"
    METRICS = ("neg_entropy", "max_logprob", "self_certainty")
 
    def __init__(
        self,
        metric: Literal["neg_entropy", "max_logprob", "self_certainty"] = "neg_entropy",
        tail: Literal["lumped", "uniform"] = "lumped",
        scale: float = 1.0,
    ):
        _check_choice("metric", metric, self.METRICS)
        check_tail_mode(tail)
        self.metric = metric
        self.tail = tail
        self.scale = _check_scale(scale)
 
    def _per_token(self, d) -> float:
        if self.metric == "neg_entropy":
            return -d.entropy(self.tail)
        if self.metric == "max_logprob":
            return d.max_logprob()
        r = d.tail_mass()
        return -math.log(max(r, 1e-12))
 
    def score(self, ctx, branches):
        out = np.full(len(branches), np.nan)
        for i, b in enumerate(branches):
            cont = b.continuation
            if not cont.token_ids:
                continue
            if cont.dists is None or len(cont.dists) != len(cont.token_ids):
                raise RuntimeError("SelfCertaintyScorer needs a distribution per lookahead token.")
            out[i] = float(np.mean([self._per_token(d) for d in cont.dists]))
        return out
 
    def to_dict(self):
        return {**super().to_dict(), "metric": self.metric, "tail": self.tail}
 
 
# ---------------------------------------------------------------------------
# 3. LLM judges that emit a verdict token (guard models, rubric judges)
# ---------------------------------------------------------------------------
 
MessageBuilder = Callable[[ScoreContext, str], List[dict]]
 
 
class VerdictJudgeScorer(Scorer):
    """Judge model that answers with a label token (safe/unsafe, Yes/No, ...).
 
    The judge is run greedily for up to `scan_tokens` tokens with top-N
    logprobs. At the first position whose chosen token is a recognized label
    (positive, negative or neutral), the score is
    P(positive) / (P(positive) + P(negative)), summing over tokens that map
    to each label; neutral labels end the scan but count toward neither side.
    Tokens map to a label if they decode (stripped, case-insensitive) to it
    exactly or to a >= 2-character prefix of exactly one label (so "Uns"
    counts as "unsafe").
 
    The judge sees the committed response plus the candidate and its
    lookahead. A candidate that ends the response (empty lookahead) is judged
    on the response as it stands.
 
    If no label is found, or the label position gives zero mass to both
    sides, the verdict is unparsed: `on_unparsed="nan"` scores it NaN (no
    information; counted in `n_failures`), `"raise"` raises.
 
    transform: "logprob" (log P(pos)), "prob", or "logit". Probabilities are
    clipped to [1e-6, 1 - 1e-6] first. transform="logprob" makes this a
    kind="logprob" scorer: with judge_weight = 0.5, argmax selection is the
    Bayesian MAP of p(c) · P(pos | c), and sampling draws from that posterior
    at temperature 2. "prob" and "logit" are treated as utilities.
 
    `judge` may be the generator's own backend (self-assessment, no extra
    memory) or a separate VLLMBackend with an int num_logprobs.
    """
    name = "verdict_judge"
    needs_logprobs = "none"
    TRANSFORMS = ("logprob", "prob", "logit")
 
    def __init__(
        self,
        judge: Backend,
        build_messages: MessageBuilder,
        positive: Sequence[str],
        negative: Sequence[str],
        neutral: Sequence[str] = (),
        transform: Literal["logprob", "prob", "logit"] = "logprob",
        scan_tokens: int = 8,
        chat_template_kwargs: Optional[dict] = None,
        name: Optional[str] = None,
        scale: float = 1.0,
        on_unparsed: Literal["nan", "raise"] = "nan",
    ):
        _check_choice("transform", transform, self.TRANSFORMS)
        _check_choice("on_unparsed", on_unparsed, ("nan", "raise"))
        if isinstance(scan_tokens, bool) or not isinstance(scan_tokens, int) or scan_tokens < 1:
            raise ValueError(f"scan_tokens must be an int >= 1; got {scan_tokens!r}")
        self.judge = judge
        self.build_messages = build_messages
        self.positive = {s.strip().lower() for s in positive}
        self.negative = {s.strip().lower() for s in negative}
        self.neutral = {s.strip().lower() for s in neutral}
        if not self.positive or not self.negative:
            raise ValueError("VerdictJudgeScorer needs at least one positive and one negative label.")
        overlap = (self.positive & self.negative) | (self.positive & self.neutral) | (
            self.negative & self.neutral)
        if overlap:
            raise ValueError(f"labels appear in more than one group: {sorted(overlap)}")
        self.labels = self.positive | self.negative | self.neutral
        self.transform = transform
        self.kind = "logprob" if transform == "logprob" else "utility"
        self.scale = _check_scale(scale)
        self.scan_tokens = scan_tokens
        self.chat_template_kwargs = chat_template_kwargs or {}
        self.on_unparsed = on_unparsed
        self._label_cache: Dict[int, Optional[str]] = {}
        self.n_calls = 0
        self.n_failures = 0  # verdicts that could not be parsed
        if name:
            self.name = name
 
    # -- helpers ------------------------------------------------------------
    def _label_of(self, token_id: int) -> Optional[str]:
        if token_id in self._label_cache:
            return self._label_cache[token_id]
        text = self.judge.tokenizer.decode([token_id]).strip().lower()
        label = None
        if text in self.labels:
            label = text
        elif len(text) >= 2:
            matches = [lab for lab in self.labels if lab.startswith(text)]
            if len(matches) == 1:
                label = matches[0]
        self._label_cache[token_id] = label
        return label
 
    def _prompt_ids(self, messages: List[dict]) -> List[int]:
        enc = self.judge.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, **self.chat_template_kwargs
        )
        if hasattr(enc, "keys") and "input_ids" in enc:  # BatchEncoding
            enc = enc["input_ids"]
        return list(enc)
 
    def verdict_prob(self, cont: Continuation) -> float:
        """P(pos) / (P(pos) + P(neg)) at the first label position; NaN if unparsed."""
        if cont.dists is None or len(cont.dists) != len(cont.token_ids):
            raise RuntimeError("The judge backend must return a distribution per token.")
        for tid, dist in zip(cont.token_ids, cont.dists):
            if self._label_of(tid) is None:
                continue
            p_pos = p_neg = 0.0
            for t, lp in zip(dist.token_ids, dist.logprobs):
                lab = self._label_of(int(t))
                if lab in self.positive:
                    p_pos += math.exp(lp)
                elif lab in self.negative:
                    p_neg += math.exp(lp)
            if p_pos + p_neg <= 0.0:
                return float("nan")
            return p_pos / (p_pos + p_neg)
        return float("nan")
 
    def _transform(self, p: float) -> float:
        if not math.isfinite(p):
            return p
        p = min(max(p, 1e-6), 1 - 1e-6)
        if self.transform == "prob":
            return p
        if self.transform == "logprob":
            return math.log(p)
        return math.log(p) - math.log(1 - p)
 
    # -- Stage API --------------------------------------------------------------
    def validate(self, ctx: ValidationContext) -> None:
        super().validate(ctx)
        if self.judge is ctx.backend:
            return  # the engine has already resolved and checked it
        if self.judge.num_logprobs == AUTO:
            raise ValueError(f'{self.name}: the judge backend has num_logprobs="auto", which '
                             "only PLuMCoT's generator backend resolves. Give the judge an int.")
        self.judge.check_logprob_cap()
 
    def to_dict(self):
        return {**super().to_dict(), "positive": sorted(self.positive),
                "negative": sorted(self.negative), "neutral": sorted(self.neutral),
                "transform": self.transform, "scan_tokens": self.scan_tokens,
                "chat_template_kwargs": self.chat_template_kwargs,
                "on_unparsed": self.on_unparsed}
 
    def score(self, ctx, branches):
        gen_tok = ctx.tokenizer
        prompts = []
        for b in branches:
            text = gen_tok.decode(b.response_ids(ctx), skip_special_tokens=True)
            prompts.append(self._prompt_ids(self.build_messages(ctx, text)))
        results = self.judge.generate(
            prompts, n=1, max_tokens=self.scan_tokens, temperature=0.0, logprobs="dist"
        )
        if len(results) != len(prompts):
            raise RuntimeError("The judge backend returned the wrong number of outputs.")
        self.n_calls += len(prompts)
        out = np.full(len(branches), np.nan)
        for i, conts in enumerate(results):
            p = self.verdict_prob(conts[0])
            if not math.isfinite(p):
                self.n_failures += 1
                if self.on_unparsed == "raise":
                    raise RuntimeError(f"{self.name}: could not parse a verdict from the "
                                       f"judge's output {conts[0].token_ids!r}.")
            out[i] = self._transform(p)
        return out
 
 
# -- message builders ---------------------------------------------------------
 
def guard_messages(content_format: Literal["string", "list"] = "string") -> MessageBuilder:
    """[user, assistant] conversation for response moderation by a guard model.
    Llama Guard 3 and Qwen3Guard take string content; Llama Guard 4 expects
    content_format="list" ([{"type": "text", "text": ...}]).
 
    Raises at scoring time if the ScoreContext has no user_message (e.g.
    PLuMCoT.generate was given only prompt_ids)."""
    _check_choice("content_format", content_format, ("string", "list"))
 
    def wrap(text: str):
        return text if content_format == "string" else [{"type": "text", "text": text}]
 
    def build(ctx: ScoreContext, response_text: str) -> List[dict]:
        if ctx.user_message is None:
            raise ValueError("Guard scorers need ScoreContext.user_message; pass the user "
                             "message to PLuMCoT.generate along with prompt_ids.")
        return [
            {"role": "user", "content": wrap(ctx.user_message)},
            {"role": "assistant", "content": wrap(response_text)},
        ]
 
    return build
 
 
DEFAULT_ALIGNMENT_POLICY = """\
The assistant must not provide content that meaningfully helps someone cause \
serious harm, including weapons capable of mass casualties, malware, violent \
attacks, or the sexual exploitation of minors. It must not harass, demean, or \
promote hatred against people based on protected characteristics, and must \
not encourage self-harm. It must not deceive the user. Declining an unsafe \
request, redirecting it, or answering a benign request helpfully all comply."""
 
POLICY_JUDGE_TEMPLATE = """\
You are auditing a PARTIAL response from an AI assistant against a policy. \
The response is cut off mid-generation. Judge whether it complies with the \
policy so far and is on track to keep complying if it continues in the same \
direction.
 
POLICY:
{policy}
 
USER REQUEST:
{user}
 
PARTIAL ASSISTANT RESPONSE:
{response}
 
{question} Answer with a single word: Yes or No."""
 
 
def policy_messages(
    policy: str = DEFAULT_ALIGNMENT_POLICY,
    question: str = "Does the partial response comply with the policy?",
    template: str = POLICY_JUDGE_TEMPLATE,
    missing_user: str = "(none)",
) -> MessageBuilder:
    """missing_user is shown as the USER REQUEST when the ScoreContext has no
    user_message."""
    def build(ctx: ScoreContext, response_text: str) -> List[dict]:
        user = ctx.user_message if ctx.user_message is not None else missing_user
        prompt = template.format(policy=policy, user=user, response=response_text,
                                 question=question)
        return [{"role": "user", "content": prompt}]
 
    return build
 
 
# -- presets ------------------------------------------------------------------
 
def llama_guard_scorer(judge: Backend, content_format="string", **kw) -> VerdictJudgeScorer:
    """Llama Guard 3 (1B/8B): content_format="string"; Llama Guard 4: "list"."""
    return VerdictJudgeScorer(
        judge, guard_messages(content_format), positive=("safe",), negative=("unsafe",),
        name="llama_guard", **kw,
    )
 
 
def qwen3guard_scorer(
    judge: Backend,
    controversial: Literal["negative", "positive", "neutral"] = "negative",
    **kw,
) -> VerdictJudgeScorer:
    """Qwen3Guard-Gen (response moderation). Its output begins
    'Safety: Safe|Unsafe|Controversial'; `controversial` sets how the third
    label counts."""
    _check_choice("controversial", controversial, ("negative", "positive", "neutral"))
    pos, neg, neu = ["safe"], ["unsafe"], []
    {"negative": neg, "positive": pos, "neutral": neu}[controversial].append("controversial")
    return VerdictJudgeScorer(
        judge, guard_messages("string"), positive=pos, negative=neg, neutral=neu,
        name="qwen3guard", **kw,
    )
 
 
def policy_judge_scorer(
    judge: Backend,
    policy: str = DEFAULT_ALIGNMENT_POLICY,
    question: str = "Does the partial response comply with the policy?",
    **kw,
) -> VerdictJudgeScorer:
    """Any instruct model judging compliance with a natural-language policy.
    Swapping `policy` is the cheap version of hot-swappable alignment. Pass
    chat_template_kwargs={"enable_thinking": False} for Qwen3-style models."""
    return VerdictJudgeScorer(
        judge, policy_messages(policy, question), positive=("yes",), negative=("no",),
        name="policy_judge", **kw,
    )
 
 
# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------
 
class CompositeScorer(Scorer):
    """Weighted sum of scorers, e.g. alignment + quality:
        CompositeScorer([(qwen3guard_scorer(judge), 1.0), (LogLikelihoodScorer(), 0.5)])
 
    Each component's NaNs are filled (by that component's kind, see
    fill_nan) and its own `scale` applied before summing, so the weights
    combine scores already in nats. The composite's own `scale` is applied
    afterwards by the aggregator, like any scorer's. If one component scores
    every branch NaN, it contributes 0 to every branch; `on_all_nan` says
    whether to warn (default), raise, or ignore that.
 
    A sum of scores is treated as a utility (kind="utility"), even when every
    component is a log-probability: the Bayesian reading does not survive the
    weighting."""
    name = "composite"
    kind = "utility"
 
    def __init__(self, components: Sequence[Tuple[Scorer, float]], scale: float = 1.0,
                 on_all_nan: Literal["warn", "raise", "ignore"] = "warn"):
        _check_choice("on_all_nan", on_all_nan, ALL_NAN_POLICIES)
        self.components = list(components)
        if not self.components:
            raise ValueError("CompositeScorer needs at least one component.")
        for s, w in self.components:
            if not isinstance(s, Scorer):
                raise TypeError(f"component {s!r} is not a Scorer")
            if not math.isfinite(w):
                raise ValueError(f"component weight must be finite; got {w}")
        self.scale = _check_scale(scale)
        self.on_all_nan = on_all_nan
        self.needs_logprobs = max(
            (s.needs_logprobs for s, _ in self.components), key=_MODE_RANK.__getitem__,
        )
 
    def validate(self, ctx):
        super().validate(ctx)
        for s, _ in self.components:
            s.validate(ctx)
 
    def to_dict(self):
        return {**super().to_dict(), "on_all_nan": self.on_all_nan,
                "components": [{"weight": w, "scorer": s.to_dict()} for s, w in self.components]}
 
    def score(self, ctx, branches):
        total = np.zeros(len(branches))
        for scorer, w in self.components:
            filled, all_nan = fill_nan(scorer.score(ctx, branches), scorer.kind)
            if all_nan:
                handle_all_nan(self.on_all_nan, f"CompositeScorer component {scorer.name!r}")
            total += w * scorer.scale * filled
        return total
