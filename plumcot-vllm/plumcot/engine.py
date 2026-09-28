"""The PLuMCoT generation loop (dissertation Algorithm 1).

Stage map:
  1   Trigger    PLuMCoT.trigger    normalized-entropy gate
  1.5 Configure  PLuMCoT.configure  constant (returns self.config)
  2   Filter     PLuMCoT.filter     top-k then top-p
  3   Lookahead  PLuMCoT.lookahead  batched vLLM continuations per candidate
  4   Score      Scorer.score       pluggable (plumcot.scorers), times Scorer.scale
  5   Select     PLuMCoT.select     sample from p(c)^(1-w) · exp(w · score)

Each stage is a method, so a variation can be tried by subclassing and
overriding one method.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .backend import Backend, Continuation
from .config import MATCH_TOP_K, PLuMCoTConfig, resolve_num_logprobs
from .dist import TokenDist, logmeanexp, softmax, top_k_top_p
from .scorers import Branch, ScoreContext, Scorer, fill_nan


@dataclass
class StepRecord:
    """One triggered generation step, for analysis."""
    position: int                  # index in the response
    norm_entropy: float
    tail_mass: float               # probability outside the returned top-N logprobs
    candidate_ids: List[int]
    candidate_texts: List[str]
    candidate_logprobs: List[float]
    scores: List[float]            # aggregated per candidate
    select_probs: List[float]
    chosen_index: int
    lookahead_texts: List[List[str]]  # [candidate][sample]


@dataclass
class PLuMCoTResult:
    text: str
    response_ids: List[int]
    trace: List[StepRecord] = field(default_factory=list)
    stats: Dict[str, int] = field(default_factory=dict)


class PLuMCoT:
    def __init__(self, backend: Backend, scorer: Scorer, config: Optional[PLuMCoTConfig] = None):
        self.backend = backend
        self.scorer = scorer
        self.config = config or PLuMCoTConfig()
        # Resolve num_logprobs="top_k" (and validate any explicit value) here,
        # so problems surface at startup rather than mid-decoding.
        backend.num_logprobs = resolve_num_logprobs(backend.num_logprobs, self.config)
        check = getattr(backend, "check_logprob_cap", None)
        if check is not None:
            check()
        self.config.validate(backend.num_logprobs)

    # -- prompt construction ---------------------------------------------------
    def build_prompt(self, user_message: str, chat: bool = True,
                     system: Optional[str] = None) -> List[int]:
        tok = self.backend.tokenizer
        if chat and getattr(tok, "chat_template", None):
            msgs = ([{"role": "system", "content": system}] if system else []) + [
                {"role": "user", "content": user_message}
            ]
            enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
            if hasattr(enc, "keys") and "input_ids" in enc:
                enc = enc["input_ids"]
            return list(enc)
        return list(tok.encode(user_message))

    # -- Stage 1.5: Configure --------------------------------------------------
    def configure(self, prompt_ids: List[int], response_ids: List[int]) -> PLuMCoTConfig:
        return self.config

    # -- Stage 1: Trigger --------------------------------------------------------
    def trigger(self, dist: TokenDist, cfg: PLuMCoTConfig) -> Tuple[bool, float]:
        h = dist.normalized_entropy(cfg.trigger.tail)
        return (cfg.trigger.enabled and h >= cfg.trigger.threshold), h

    # -- Stage 2: Filter ---------------------------------------------------------
    def filter(self, dist: TokenDist, cfg: PLuMCoTConfig) -> Tuple[np.ndarray, np.ndarray]:
        f = cfg.filter
        return top_k_top_p(dist, f.top_k, f.top_p, f.min_candidates)

    # -- Stage 3: Lookahead ------------------------------------------------------
    def lookahead(self, context: List[int], cand_ids: np.ndarray, cand_lps: np.ndarray,
                  cfg: PLuMCoTConfig, seed: Optional[int]) -> List[List[Branch]]:
        la = cfg.lookahead
        n = 1 if la.temperature == 0.0 else la.num_samples
        stop_ids = self.backend.stop_token_ids
        live = [i for i, c in enumerate(cand_ids) if int(c) not in stop_ids]
        conts = self.backend.generate(
            [context + [int(cand_ids[i])] for i in live],
            n=n, max_tokens=la.max_tokens, temperature=la.temperature,
            top_k=la.top_k, top_p=la.top_p, stop=la.stop,
            logprobs=self.scorer.needs_logprobs, seed=seed,
        )
        by_cand: Dict[int, List[Continuation]] = dict(zip(live, conts))
        branches = []
        for i, (c, lp) in enumerate(zip(cand_ids, cand_lps)):
            # A stop-token candidate ends the response: nothing to roll out.
            cs = by_cand.get(i) or [Continuation([], [], [], "candidate_is_stop")] * n
            branches.append([Branch(int(c), float(lp), cont) for cont in cs])
        return branches

    # -- Stage 4 (aggregation of the Scorer's per-sample output) -----------------
    def aggregate_mode(self, cfg: PLuMCoTConfig) -> str:
        """The configured mode, or the scorer's default for its kind."""
        return cfg.lookahead.aggregate or self.scorer.default_aggregate

    def aggregate(self, raw: np.ndarray, cfg: PLuMCoTConfig) -> np.ndarray:
        """raw: (num_candidates, num_samples) -> (num_candidates,)"""
        raw = fill_nan(raw.ravel()).reshape(raw.shape)
        mode = self.aggregate_mode(cfg)
        if mode == "mean":
            return raw.mean(axis=1)
        if mode == "max":
            return raw.max(axis=1)
        if mode == "min":
            return raw.min(axis=1)
        tau = cfg.lookahead.aggregate_temperature
        return tau * logmeanexp(raw / tau, axis=1)

    # -- Stage 5: Select -----------------------------------------------------------
    def select(self, cand_lps: np.ndarray, scores: np.ndarray, cfg: PLuMCoTConfig,
               rng: np.random.Generator) -> Tuple[int, np.ndarray]:
        """P(c) ∝ p(c)^(1 - w) · exp(w · score(c)), w = cfg.select.judge_weight."""
        s = cfg.select
        w = s.judge_weight
        probs = softmax((1 - w) * cand_lps + w * scores)
        idx = int(np.argmax(probs)) if s.mode == "argmax" else int(rng.choice(len(probs), p=probs))
        return idx, probs

    # -- main loop -----------------------------------------------------------------
    def generate(
        self,
        user_message: Optional[str] = None,
        *,
        prompt_ids: Optional[List[int]] = None,
        chat: bool = True,
        system: Optional[str] = None,
        keep_trace: bool = True,
    ) -> PLuMCoTResult:
        if prompt_ids is None:
            if user_message is None:
                raise ValueError("Provide user_message or prompt_ids.")
            prompt_ids = self.build_prompt(user_message, chat=chat, system=system)
        tok = self.backend.tokenizer
        stop_ids = self.backend.stop_token_ids
        rng = np.random.default_rng(self.config.seed)

        def next_seed() -> Optional[int]:
            return None if self.config.seed is None else int(rng.integers(2**31 - 1))

        response: List[int] = []
        trace: List[StepRecord] = []
        stats = dict(tokens=0, triggered=0, single_candidate=0, chunks=0,
                     lookahead_tokens=0, branches=0)

        while len(response) < self.config.max_new_tokens:
            cfg = self.configure(prompt_ids, response)
            remaining = cfg.max_new_tokens - len(response)
            chunk = self.backend.generate(
                [prompt_ids + response], n=1, max_tokens=min(cfg.chunk_size, remaining),
                temperature=cfg.temperature, top_k=cfg.filter.top_k, top_p=cfg.filter.top_p,
                logprobs="dist", seed=next_seed(),
            )[0][0]
            stats["chunks"] += 1
            if not chunk.token_ids:
                break

            # Stage 1 on each position of the chunk; stop at the first firing.
            fire_at, fire_h = None, 0.0
            for j, d in enumerate(chunk.dists or []):
                fired, h = self.trigger(d, cfg)
                if fired:
                    fire_at, fire_h = j, h
                    break

            if fire_at is None:
                response += chunk.token_ids
                if chunk.finish_reason == "stop":
                    break
                continue

            response += chunk.token_ids[:fire_at]
            dist = chunk.dists[fire_at]
            cand_ids, cand_lps = self.filter(dist, cfg)

            if len(cand_ids) == 1:
                stats["single_candidate"] += 1
                chosen = int(cand_ids[0])
            else:
                stats["triggered"] += 1
                context = prompt_ids + response
                branches = self.lookahead(context, cand_ids, cand_lps, cfg, next_seed())
                flat = [b for bs in branches for b in bs]
                ctx = ScoreContext(user_message, list(prompt_ids), list(response), tok)
                raw = np.asarray(self.scorer.score(ctx, flat), dtype=np.float64)
                raw = raw.reshape(len(branches), -1) * self.scorer.scale
                scores = self.aggregate(raw, cfg)
                idx, probs = self.select(cand_lps, scores, cfg, rng)
                chosen = int(cand_ids[idx])
                stats["branches"] += len(flat)
                stats["lookahead_tokens"] += sum(len(b.continuation.token_ids) for b in flat)
                if keep_trace:
                    trace.append(StepRecord(
                        position=len(response),
                        norm_entropy=fire_h,
                        tail_mass=dist.tail_mass(),
                        candidate_ids=[int(c) for c in cand_ids],
                        candidate_texts=[tok.decode([int(c)]) for c in cand_ids],
                        candidate_logprobs=[float(x) for x in cand_lps],
                        scores=[float(x) for x in scores],
                        select_probs=[float(x) for x in probs],
                        chosen_index=idx,
                        lookahead_texts=[[tok.decode(b.continuation.token_ids) for b in bs]
                                         for bs in branches],
                    ))

            response.append(chosen)
            if chosen in stop_ids:
                break

        stats["tokens"] = len(response)
        text = tok.decode(response, skip_special_tokens=True)
        return PLuMCoTResult(text, response, trace, stats)


def calibrate_threshold(
    backend: Backend,
    prompts: Sequence[List[int]],
    target_rate: float = 0.1,
    max_tokens: int = 256,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
    tail: str = "lumped",
) -> Tuple[float, np.ndarray]:
    """Decode normally and return the normalized-entropy threshold at which
    the Trigger would fire on roughly `target_rate` of tokens, plus all
    observed normalized entropies (useful for plotting)."""
    if backend.num_logprobs == MATCH_TOP_K:
        backend.num_logprobs = top_k
    outs = backend.generate(prompts, n=1, max_tokens=max_tokens, temperature=temperature,
                            top_k=top_k, top_p=top_p, logprobs="dist")
    hs = np.array([d.normalized_entropy(tail) for req in outs for d in (req[0].dists or [])])
    if hs.size == 0:
        raise RuntimeError("No tokens generated during calibration.")
    return float(np.quantile(hs, 1.0 - target_rate)), hs
