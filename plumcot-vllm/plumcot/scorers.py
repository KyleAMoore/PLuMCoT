"""Stage 4: Score.

A scorer receives every lookahead sample for every candidate as a flat list of
`Branch`es and returns one float per branch (higher = better). The engine
multiplies by `Scorer.scale`, aggregates samples per candidate, and passes the
result to Select, where it enters as w · score against (1 - w) · log p(c).

Each scorer declares a `kind`, which decides the default aggregation:

  "logprob" - the score is log P(event | branch), e.g. a judge's log P(safe).
              Default aggregation is logmeanexp at tau = 1 (log of the mean
              probability), and Select has a tempered-Bayes reading.
  "utility" - any real-valued utility. Default aggregation is the mean
              (risk-neutral), and Select is a Boltzmann choice over
              (1 - w)·log p + w·score.

Because the score is blended with log p(c), its units matter: `scale` is the
number of nats one unit of score is worth. Log-probability scores are already
in nats (scale = 1); rescale utilities on other scales once, per scorer, rather
than normalizing within each candidate group.

A scorer may return NaN for "no information" (for example, a candidate that
is itself end-of-sequence has an empty lookahead). The engine replaces NaN
with the mean of the finite scores, which makes that branch neutral.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np

from .backend import Backend, Continuation, LogprobMode

_MODE_RANK = {"none": 0, "chosen": 1, "dist": 2}

ScoreKind = Literal["logprob", "utility"]
_DEFAULT_AGGREGATE = {"logprob": "logmeanexp", "utility": "mean"}


@dataclass
class ScoreContext:
    user_message: Optional[str]   # raw user text (needed by judges)
    prompt_ids: List[int]
    response_ids: List[int]       # committed response so far (before the candidate)
    tokenizer: object             # generator tokenizer


@dataclass
class Branch:
    candidate_id: int
    candidate_logprob: float
    continuation: Continuation

    def response_ids(self, ctx: ScoreContext) -> List[int]:
        return ctx.response_ids + [self.candidate_id] + self.continuation.token_ids


class Scorer:
    name: str = "scorer"
    # What the Lookahead must return per token: nothing, the sampled token's
    # logprob, or the top-N distribution at each position.
    needs_logprobs: LogprobMode = "none"
    # "logprob" or "utility"; sets the default aggregation (see module docstring).
    kind: ScoreKind = "utility"
    # Nats per unit of score, applied by the engine before aggregation.
    scale: float = 1.0

    @property
    def default_aggregate(self) -> str:
        return _DEFAULT_AGGREGATE[self.kind]

    def score(self, ctx: ScoreContext, branches: Sequence[Branch]) -> np.ndarray:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 1. Lookahead likelihood (baseline)
# ---------------------------------------------------------------------------

class LogLikelihoodScorer(Scorer):
    """Log-probability of the lookahead under the generator itself.

    reduce="mean" is length-normalized (average nats per token);
    reduce="sum" is log p(lookahead | ctx, c). With reduce="sum" and
    judge_weight = 0.5, argmax selection tracks the joint probability of
    candidate plus lookahead, i.e. it behaves like a one-token beam search. It
    is included as a baseline: it is mode-seeking and favors repetitive
    continuations.

    kind is "utility": the score is a log-probability of the lookahead text,
    not of an acceptability event, so averaging over rollouts is the natural
    aggregate.
    """
    name = "loglik"
    needs_logprobs = "chosen"

    def __init__(self, reduce: Literal["mean", "sum"] = "mean", scale: float = 1.0):
        self.reduce = reduce
        self.scale = scale

    def score(self, ctx, branches):
        out = np.full(len(branches), np.nan)
        for i, b in enumerate(branches):
            lps = b.continuation.token_logprobs
            if not lps:
                continue
            arr = np.asarray(lps, dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            if arr.size:
                out[i] = arr.mean() if self.reduce == "mean" else arr.sum()
        return out


# ---------------------------------------------------------------------------
# 2. Self-certainty (intrinsic, no auxiliary model)
# ---------------------------------------------------------------------------

class SelfCertaintyScorer(Scorer):
    """How confident the generator is while producing the lookahead.

    metric:
      "neg_entropy"    - minus mean token entropy in nats (default)
      "max_logprob"    - mean log-probability of the top token at each step
      "self_certainty" - INTUITOR-style KL(Uniform || p). This needs the full
                         vocabulary; from top-N logprobs with a uniform tail it
                         reduces to about -log(tail mass), so treat it as a
                         rough approximation only.

    Greedy lookaheads are confident by construction, which compresses these
    scores; sampled lookaheads (temperature 1, num_samples > 1) discriminate
    better. Expect this scorer to reward the model's priors, including
    base-rate preferences.
    """
    name = "self_certainty"
    needs_logprobs = "dist"

    def __init__(
        self,
        metric: Literal["neg_entropy", "max_logprob", "self_certainty"] = "neg_entropy",
        tail: Literal["lumped", "uniform"] = "lumped",
        scale: float = 1.0,
    ):
        self.metric = metric
        self.tail = tail
        self.scale = scale

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
            dists = b.continuation.dists
            if dists:
                out[i] = float(np.mean([self._per_token(d) for d in dists]))
        return out


# ---------------------------------------------------------------------------
# 3. LLM judges that emit a verdict token (guard models, rubric judges)
# ---------------------------------------------------------------------------

MessageBuilder = Callable[[ScoreContext, str], List[dict]]


class VerdictJudgeScorer(Scorer):
    """Judge model that answers with a label token (safe/unsafe, Yes/No, ...).

    The judge is run greedily for a few tokens with top-N logprobs. At the
    first position whose chosen token is a recognized label, the score is
    P(positive) / (P(positive) + P(negative)), summing over tokens that map
    to each label. Tokens map to a label if they decode to it exactly or to a
    >= 2-character prefix of exactly one label (so "Uns" counts as "unsafe").

    transform: "logprob" (default, log P(pos)), "prob", or "logit".
    transform="logprob" makes this a kind="logprob" scorer: with
    judge_weight = 0.5, argmax selection is the Bayesian MAP of
    p(c) · P(pos | c), and sampling draws from that posterior at temperature 2.
    "prob" and "logit" are treated as utilities.

    `judge` may be the generator's own backend (self-assessment, no extra
    memory) or a separate VLLMBackend.
    """
    name = "verdict_judge"
    needs_logprobs = "none"

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
    ):
        self.judge = judge
        self.build_messages = build_messages
        self.positive = {s.lower() for s in positive}
        self.negative = {s.lower() for s in negative}
        self.neutral = {s.lower() for s in neutral}
        self.labels = self.positive | self.negative | self.neutral
        self.transform = transform
        self.kind = "logprob" if transform == "logprob" else "utility"
        self.scale = scale
        self.scan_tokens = scan_tokens
        self.chat_template_kwargs = chat_template_kwargs or {}
        self._label_cache: Dict[int, Optional[str]] = {}
        self.n_calls = 0
        self.n_failures = 0  # verdicts that could not be parsed
        if name:
            self.name = name

    # -- helpers ------------------------------------------------------------
    def _label_of(self, token_id: int) -> Optional[str]:
        if token_id in self._label_cache:
            return self._label_cache[token_id]
        text = self.judge.tokenizer.decode([token_id]).strip().lower()  # pyright: ignore[reportAttributeAccessIssue]
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
        enc = self.judge.tokenizer.apply_chat_template(  # pyright: ignore[reportAttributeAccessIssue]
            messages, tokenize=True, add_generation_prompt=True, **self.chat_template_kwargs
        )
        if hasattr(enc, "keys") and "input_ids" in enc:
            enc = enc["input_ids"]
        return list(enc)

    def verdict_prob(self, cont: Continuation) -> float:
        if not cont.dists:
            return float("nan")
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

    # -- Scorer API -----------------------------------------------------------
    def score(self, ctx, branches):
        gen_tok = ctx.tokenizer
        prompts = []
        for b in branches:
            text = gen_tok.decode(b.response_ids(ctx), skip_special_tokens=True) # pyright: ignore[reportAttributeAccessIssue]
            prompts.append(self._prompt_ids(self.build_messages(ctx, text)))
        results = self.judge.generate(
            prompts, n=1, max_tokens=self.scan_tokens, temperature=0.0, logprobs="dist"
        )
        self.n_calls += len(prompts)
        out = np.full(len(branches), np.nan)
        for i, conts in enumerate(results):
            p = self.verdict_prob(conts[0])
            if not math.isfinite(p):
                self.n_failures += 1
            out[i] = self._transform(p)
        return out


# -- message builders ---------------------------------------------------------

def guard_messages(content_format: Literal["string", "list"] = "string") -> MessageBuilder:
    """[user, assistant] conversation for response moderation by a guard model.
    Llama Guard 3 and Qwen3Guard take string content; Llama Guard 4 expects
    content_format="list" ([{"type": "text", "text": ...}])."""

    def wrap(text: str):
        return text if content_format == "string" else [{"type": "text", "text": text}]

    def build(ctx: ScoreContext, response_text: str) -> List[dict]:
        if ctx.user_message is None:
            raise ValueError("Guard scorers need ScoreContext.user_message.")
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
) -> MessageBuilder:
    def build(ctx: ScoreContext, response_text: str) -> List[dict]:
        prompt = template.format(
            policy=policy, user=ctx.user_message or "(none)", response=response_text,
            question=question,
        )
        return [{"role": "user", "content": prompt}]

    return build


# -- presets ------------------------------------------------------------------

def llama_guard_scorer(judge: Backend, content_format="string", **kw) -> VerdictJudgeScorer:
    """Llama Guard 3 (1B/8B): content_format="string"; Llama Guard 4: "list"."""
    return VerdictJudgeScorer(
        judge, guard_messages(content_format), positive=("safe",), negative=("unsafe",), # pyright: ignore[reportArgumentType]
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

def fill_nan(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).copy()
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x)
    x[~finite] = x[finite].mean()
    return x


class CompositeScorer(Scorer):
    """Weighted sum of scorers, e.g. alignment + quality:
        CompositeScorer([(qwen3guard_scorer(judge), 1.0), (LogLikelihoodScorer(), 0.5)])

    Each component's NaNs are filled and its own `scale` applied before
    summing, so the weights combine scores already in nats. A sum of scores is
    treated as a utility (kind="utility"), even when every component is a
    log-probability: the Bayesian reading does not survive the weighting."""
    name = "composite"
    kind = "utility"

    def __init__(self, components: Sequence[Tuple[Scorer, float]], scale: float = 1.0):
        self.components = list(components)
        self.scale = scale
        self.needs_logprobs = max(
            (s.needs_logprobs for s, _ in self.components), key=_MODE_RANK.__getitem__,
            default="none",
        ) # pyright: ignore[reportAttributeAccessIssue]

    def score(self, ctx, branches):
        total = np.zeros(len(branches))
        for scorer, w in self.components:
            total += w * scorer.scale * fill_nan(scorer.score(ctx, branches))
        return total
