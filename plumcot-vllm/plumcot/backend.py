"""Thin wrapper around a vLLM `LLM` that works entirely in token-id space.
 
Everything PLuMCoT needs from the model goes through `generate()`:
ordinary chunked decoding, candidate lookaheads, and judge verdicts.
 
vLLM notes
----------
* `num_logprobs` is how many logprobs are requested per position:
    "auto" (default) - resolved by PLuMCoT to the largest count its stages
                       need (for the default Filter, its top-k)
    -1               - the full vocabulary (exact entropy; slow)
    n >= 1           - the top n
  Judge backends are not resolved by PLuMCoT, so give them an int.
* Construct the LLM with `max_logprobs` at least that large (vLLM's default
  is 20; -1 removes the cap). `config.resolve_num_logprobs` gives the value.
* Returned logprobs are the model's raw distribution (vLLM V1's default
  `logprobs_mode="raw_logprobs"`), i.e. before temperature/top-k/top-p.
  The Trigger's entropy is computed on these raw values; the Filter applies
  the base-decoding temperature itself.
* Prefix caching is what makes lookahead affordable: all k candidate
  continuations share the context's KV cache.
"""
from __future__ import annotations
 
import warnings
from dataclasses import dataclass
from typing import Iterable, List, Literal, Optional, Protocol, Sequence
 
from .config import AUTO, FULL_VOCAB, NumLogprobs
from .dist import TokenDist
 
LogprobMode = Literal["none", "chosen", "dist"]
LOGPROB_MODES = ("none", "chosen", "dist")
 
 
@dataclass
class Continuation:
    token_ids: List[int]
    token_logprobs: Optional[List[float]] = None  # logprob of each generated token
    dists: Optional[List[TokenDist]] = None       # top-N distribution at each position
    finish_reason: Optional[str] = None           # "stop", "length", or a PLuMCoT marker
 
 
class Backend(Protocol):
    """What the engine and scorers need; VLLMBackend implements it and the
    test suite provides a fake.
 
    generate() must return, for each context, exactly `n` Continuations, and
    must populate `token_logprobs` for logprobs="chosen" and both
    `token_logprobs` and `dists` for logprobs="dist".
    """
    tokenizer: object
    vocab_size: int
    num_logprobs: NumLogprobs
    stop_token_ids: set
 
    def check_logprob_cap(self) -> None:
        """Raise if the resolved num_logprobs exceeds what the model serves."""
        ...
 
    def generate(
        self,
        contexts: Sequence[Sequence[int]],
        *,
        n: int = 1,
        max_tokens: int,
        temperature: float,
        top_k: Optional[int] = None,
        top_p: float = 1.0,
        stop: Sequence[str] = (),
        logprobs: LogprobMode = "none",
        seed: Optional[int] = None,
    ) -> List[List[Continuation]]: ...
 
 
class VLLMBackend:
    def __init__(
        self,
        llm,
        num_logprobs: NumLogprobs = AUTO,
        stop_token_ids: Optional[Iterable[int]] = None,
        model_name: Optional[str] = None,
    ):
        """stop_token_ids: None reads them from the tokenizer's eos token plus
        the model's generation_config.json (warning if that file cannot be
        read); pass them explicitly to skip the lookup."""
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
 
        self._SamplingParams = SamplingParams
        self._TokensPrompt = TokensPrompt
        self.llm = llm
        self.num_logprobs = num_logprobs
        self.tokenizer = llm.get_tokenizer()
        self.vocab_size = len(self.tokenizer)
        self.stop_token_ids = (
            set(stop_token_ids) if stop_token_ids is not None
            else self._default_stop_ids(model_name)
        )
        if not self.stop_token_ids:
            raise ValueError("No stop token ids found; pass stop_token_ids explicitly.")
 
    def _default_stop_ids(self, model_name: Optional[str]) -> set:
        ids = set()
        if getattr(self.tokenizer, "eos_token_id", None) is not None:
            ids.add(self.tokenizer.eos_token_id)
        # Chat models often end turns with a token other than eos (e.g.
        # <|eot_id|>, <|im_end|>); generation_config.json lists them.
        try:
            from transformers import GenerationConfig
 
            name = model_name or self.llm.model_config.model
            eos = GenerationConfig.from_pretrained(name).eos_token_id
            if eos is not None:
                ids.update(eos if isinstance(eos, (list, tuple)) else [eos])
        except Exception as e:  # noqa: BLE001 - reported, not swallowed
            warnings.warn(
                f"Could not read generation_config.json ({type(e).__name__}: {e}); "
                f"stopping only on the tokenizer's eos token {sorted(ids)}. Chat models "
                "that end turns with another token will run to max_new_tokens; pass "
                "stop_token_ids explicitly.",
                stacklevel=3,
            )
        return ids
 
    def check_logprob_cap(self) -> None:
        """Fail early, with the fix, if the LLM's max_logprobs is too small."""
        n = self.num_logprobs
        if n == AUTO:
            raise ValueError('num_logprobs="auto" has not been resolved; PLuMCoT resolves '
                             "it for the generator. Give judge backends an int.")
        cap = self.llm.model_config.max_logprobs
        if cap == FULL_VOCAB:
            return
        if n == FULL_VOCAB or n > cap:
            raise ValueError(
                f"num_logprobs={n} but the LLM was built with max_logprobs={cap}; "
                f"construct it with LLM(..., max_logprobs={n})."
            )
 
    def generate(
        self,
        contexts,
        *,
        n=1,
        max_tokens,
        temperature,
        top_k=None,
        top_p=1.0,
        stop=(),
        logprobs: LogprobMode = "none",
        seed=None,
    ):
        if logprobs not in LOGPROB_MODES:
            raise ValueError(f"logprobs must be one of {LOGPROB_MODES}; got {logprobs!r}")
        if not contexts:
            return []
        kwargs = dict(
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_token_ids=sorted(self.stop_token_ids),
            seed=seed,
        )
        if top_k is not None:
            if top_k < 1:
                raise ValueError(f"top_k must be None (disabled) or >= 1; got {top_k}")
            kwargs["top_k"] = top_k
        if stop:
            kwargs["stop"] = list(stop)
        if logprobs != "none":
            if logprobs == "dist" and self.num_logprobs == AUTO:
                raise RuntimeError(
                    'num_logprobs="auto" has not been resolved. Pass this backend to '
                    "PLuMCoT (which resolves it) or give it an int."
                )
            # logprobs=0 returns only the sampled token's logprob; -1 returns
            # the full vocabulary.
            kwargs["logprobs"] = self.num_logprobs if logprobs == "dist" else 0
        params = self._SamplingParams(**kwargs)
        prompts = [self._TokensPrompt(prompt_token_ids=list(c)) for c in contexts]
        outputs = self.llm.generate(prompts, params, use_tqdm=False)
        return [[self._convert(o, logprobs) for o in req.outputs] for req in outputs]
 
    def _convert(self, out, mode: LogprobMode) -> Continuation:
        ids = list(out.token_ids)
        token_lps = dists = None
        if mode != "none":
            if out.logprobs is None or len(out.logprobs) != len(ids):
                raise RuntimeError(
                    f"logprobs={mode!r} was requested but vLLM returned "
                    f"{'none' if out.logprobs is None else len(out.logprobs)} positions "
                    f"for {len(ids)} tokens."
                )
            token_lps = []
            dists = [] if mode == "dist" else None
            for tid, pos in zip(ids, out.logprobs):
                lp = pos.get(tid) if pos else None
                if lp is None:
                    # vLLM always includes the sampled token; its absence
                    # means the output is not what this wrapper expects.
                    raise RuntimeError(f"vLLM omitted the sampled token {tid}'s logprob.")
                token_lps.append(float(getattr(lp, "logprob", lp)))
                if dists is not None:
                    dists.append(TokenDist.from_vllm(pos, self.vocab_size))
        return Continuation(ids, token_lps, dists, out.finish_reason)
