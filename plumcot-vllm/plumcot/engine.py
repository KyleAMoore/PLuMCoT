"""The PLuMCoT generation loop (dissertation Algorithm 1).
 
    for each generation step:
        Trigger (fixed)            -> fire?
        if fired:
            Configure              -> which StageBundle
            Filter                 -> candidates
            Lookahead              -> rollouts per candidate
            Scorer + aggregator    -> one score per candidate
            Select                 -> the token to commit
        else:
            base decoding
 
The stages are objects (plumcot.stages); this module only sequences them.
Untriggered tokens are decoded in chunks, and each chunk is truncated at the
first position where the Trigger fires (see PLuMCoTConfig.chunk_size).
"""
from __future__ import annotations
 
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
 
import numpy as np
 
from .backend import Backend
from .base import ValidationContext
from .config import PLuMCoTConfig, resolve_num_logprobs, warn_if_few_logprobs
from .scorers import ScoreContext
from .stages import (Configure, FixedConfigure, StageBundle, StepContext, Trigger,
                     _is_int)
 
 
@dataclass
class StepRecord:
    """One triggered step, for analysis.
 
    searched is False when the Filter left a single candidate; that
    candidate is committed without lookahead, and the score fields are None.
    """
    position: int                    # index in the response
    option: int                      # index of the Configure option used
    trigger_info: Dict[str, float]
    tail_mass: float                 # probability outside the returned top-N logprobs
    candidate_ids: List[int]
    candidate_texts: List[str]
    candidate_logprobs: List[float]  # raw model logprobs
    prior_logprobs: List[float]      # base-policy prior, renormalized over candidates
    chosen_index: int
    searched: bool
    scores: Optional[List[float]] = None               # per candidate, after aggregation
    sample_scores: Optional[List[List[float]]] = None  # [candidate][sample], filled and scaled
    select_probs: Optional[List[float]] = None
    lookahead_texts: Optional[List[List[str]]] = None  # [candidate][sample]
 
 
@dataclass
class PLuMCoTResult:
    text: str
    response_ids: List[int]
    trace: List[StepRecord] = field(default_factory=list)
    stats: Dict[str, int] = field(default_factory=dict)
 
 
class PLuMCoT:
    """Run the PLuMCoT loop on a backend.
 
    trigger: Stage 1, fixed for the whole generation.
    stages:  a StageBundle (used at every triggered step) or a Configure that
             chooses among declared bundles per triggered step.
    config:  loop-level settings; None means PLuMCoTConfig() with its
             documented defaults.
 
    Construction resolves backend.num_logprobs="auto" in place (to the
    largest count any stage needs) and validates every stage against the
    configuration, so problems surface here rather than mid-generation.
    """
 
    def __init__(self, backend: Backend, *, trigger: Trigger,
                 stages: Union[StageBundle, Configure],
                 config: Optional[PLuMCoTConfig] = None):
        if not isinstance(trigger, Trigger):
            raise TypeError(f"trigger must be a Trigger; got {type(trigger).__name__}")
        if isinstance(stages, StageBundle):
            stages = FixedConfigure(stages)
        if not isinstance(stages, Configure):
            raise TypeError(f"stages must be a StageBundle or a Configure; "
                            f"got {type(stages).__name__}")
        self.backend = backend
        self.trigger = trigger
        self.configure = stages
        self.config = config if config is not None else PLuMCoTConfig()
        self.config.validate()
 
        all_stages = [trigger] + [s for b in stages.options for s in b.stages()]
        backend.num_logprobs = resolve_num_logprobs(backend.num_logprobs, self.config, all_stages)
        backend.check_logprob_cap()
        warn_if_few_logprobs(backend.num_logprobs)
        ctx = ValidationContext(self.config, backend.num_logprobs, backend)
        for s in [trigger, stages] + all_stages[1:]:
            s.validate(ctx)
 
    def describe(self) -> dict:
        """JSON-serializable description of the loop and every stage."""
        return {"loop": self.config.to_dict(), "trigger": self.trigger.to_dict(),
                "configure": self.configure.to_dict()}
 
    # -- prompt construction ---------------------------------------------------
    def build_prompt(self, user_message: str, chat: bool = True,
                     system: Optional[str] = None) -> List[int]:
        """chat=True applies the tokenizer's chat template (and raises if it
        has none); chat=False encodes the raw text."""
        tok = self.backend.tokenizer
        if not chat:
            if system is not None:
                raise ValueError("A system message needs chat=True.")
            return list(tok.encode(user_message))
        if not getattr(tok, "chat_template", None):
            raise ValueError("The tokenizer has no chat template; pass chat=False to "
                             "encode the raw prompt.")
        msgs = ([{"role": "system", "content": system}] if system is not None else []) + [
            {"role": "user", "content": user_message}
        ]
        enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
        if hasattr(enc, "keys") and "input_ids" in enc:  # BatchEncoding
            enc = enc["input_ids"]
        return list(enc)
 
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
        """Give user_message (the prompt is built from it) or prompt_ids, or
        both: scorers that judge the conversation read user_message, so pass
        it along with prompt_ids when using one."""
        if prompt_ids is None:
            if user_message is None:
                raise ValueError("Provide user_message or prompt_ids.")
            prompt_ids = self.build_prompt(user_message, chat=chat, system=system)
        prompt_ids = list(prompt_ids)
        cfg = self.config
        dec = cfg.decoding
        tok = self.backend.tokenizer
        stop_ids = self.backend.stop_token_ids
        rng = np.random.default_rng(cfg.seed)
 
        def next_seed() -> Optional[int]:
            return None if cfg.seed is None else int(rng.integers(2**31 - 1))
 
        response: List[int] = []
        trace: List[StepRecord] = []
        stats = dict(tokens=0, chunks=0, fired=0, searched=0, single_candidate=0,
                     branches=0, lookahead_tokens=0)
 
        while len(response) < cfg.max_new_tokens:
            remaining = cfg.max_new_tokens - len(response)
            outs = self.backend.generate(
                [prompt_ids + response], n=1, max_tokens=min(cfg.chunk_size, remaining),
                temperature=dec.temperature, top_k=dec.top_k, top_p=dec.top_p,
                logprobs="dist", seed=next_seed(),
            )
            chunk = outs[0][0]
            stats["chunks"] += 1
            if not chunk.token_ids:
                raise RuntimeError("The backend returned an empty chunk.")
            if chunk.dists is None or len(chunk.dists) != len(chunk.token_ids):
                raise RuntimeError("The backend must return a distribution for every "
                                   "position of a chunk (logprobs='dist').")
 
            # Stage 1 on each position of the chunk; stop at the first firing.
            fire_at, info = None, {}
            for j, d in enumerate(chunk.dists):
                fired, info = self.trigger(d, prompt_ids, response + chunk.token_ids[:j])
                if fired:
                    fire_at = j
                    break
 
            if fire_at is None:
                response += chunk.token_ids
                if chunk.finish_reason == "stop":
                    break
                continue
 
            response += chunk.token_ids[:fire_at]
            stats["fired"] += 1
            chosen, record = self._step(user_message, prompt_ids, response,
                                        chunk.dists[fire_at], info, rng, next_seed, stats)
            if keep_trace:
                trace.append(record)
            response.append(chosen)
            if chosen in stop_ids:
                break
 
        stats["tokens"] = len(response)
        text = tok.decode(response, skip_special_tokens=True)
        return PLuMCoTResult(text, response, trace, stats)
 
    def _step(self, user_message, prompt_ids, response, dist, trigger_info, rng, next_seed,
              stats):
        """Stages 1.5-5 at one triggered position. Returns (token, StepRecord)."""
        tok = self.backend.tokenizer
        opts = self.configure.options
        i = self.configure.choose(StepContext(list(prompt_ids), list(response), dist,
                                              dict(trigger_info)))
        if not _is_int(i) or not 0 <= i < len(opts):
            raise ValueError(f"Configure.choose returned {i!r}; expected an index into its "
                             f"{len(opts)} options.")
        i = int(i)
        b = opts[i]
        cands = b.filter(dist, self.config.decoding)
        rec = StepRecord(
            position=len(response), option=i, trigger_info=dict(trigger_info),
            tail_mass=dist.tail_mass(),
            candidate_ids=[int(c) for c in cands.token_ids],
            candidate_texts=[tok.decode([int(c)]) for c in cands.token_ids],
            candidate_logprobs=[float(x) for x in cands.logprobs],
            prior_logprobs=[float(x) for x in cands.prior_logprobs],
            chosen_index=0, searched=False,
        )
        if len(cands.token_ids) == 1:
            stats["single_candidate"] += 1
            return int(cands.token_ids[0]), rec
 
        stats["searched"] += 1
        branches = b.lookahead(self.backend, prompt_ids + response, cands,
                               b.scorer.needs_logprobs, next_seed())
        n_c = len(cands.token_ids)
        if len(branches) != n_c or not branches[0] or any(
                len(bs) != len(branches[0]) for bs in branches):
            raise RuntimeError("Lookahead must return the same number (>= 1) of branches "
                               "for every candidate.")
        flat = [br for bs in branches for br in bs]
        sctx = ScoreContext(user_message, list(prompt_ids), list(response), tok)
        raw = np.asarray(b.scorer.score(sctx, flat), dtype=np.float64)
        if raw.shape != (len(flat),):
            raise RuntimeError(f"Scorer {b.scorer.name!r} returned shape {raw.shape}; "
                               f"expected ({len(flat)},).")
        scores, sample_scores = b.aggregate(raw.reshape(n_c, -1), b.scorer)
        idx, probs = b.select(cands, scores, rng)
        if not _is_int(idx) or not 0 <= idx < n_c:
            raise ValueError(f"Select returned {idx!r}; expected an index into {n_c} candidates.")
        idx = int(idx)
        stats["branches"] += len(flat)
        stats["lookahead_tokens"] += sum(len(br.continuation.token_ids) for br in flat)
        rec.chosen_index = idx
        rec.searched = True
        rec.scores = [float(x) for x in scores]
        rec.sample_scores = [[float(x) for x in row] for row in sample_scores]
        rec.select_probs = [float(x) for x in probs]
        rec.lookahead_texts = [[tok.decode(br.continuation.token_ids) for br in bs]
                               for bs in branches]
        return int(cands.token_ids[idx]), rec
