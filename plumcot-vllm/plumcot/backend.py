"""Thin wrapper around a vLLM `LLM` that works entirely in token-id space.

Everything PLuMCoT needs from the model goes through `generate()`:
ordinary chunked decoding, candidate lookaheads, and judge verdicts.

vLLM notes
----------
* `num_logprobs` is how many logprobs are requested per position:
    "top_k" (default) - resolved by PLuMCoT to its config's filter.top_k
    -1                - the full vocabulary (exact entropy; slow)
    n >= 1            - the top n
  Judge backends have no Filter, so give them an int.
* Construct the LLM with `max_logprobs` at least that large (vLLM's default
  is 20; -1 removes the cap). `config.resolve_num_logprobs` gives the value:
      LLM(model, max_logprobs=resolve_num_logprobs("top_k", cfg))
* Returned logprobs are the model's raw distribution (vLLM V1's default
  `logprobs_mode="raw_logprobs"`), i.e. before temperature/top-k/top-p.
  That is what the Trigger's entropy and the Filter should see.
* Prefix caching is what makes lookahead affordable: all k candidate
  continuations share the context's KV cache.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Literal, Optional, Protocol, Sequence

from .config import FULL_VOCAB, MATCH_TOP_K, NumLogprobs
from .dist import TokenDist

LogprobMode = Literal["none", "chosen", "dist"]


@dataclass
class Continuation:
    token_ids: List[int]
    token_logprobs: Optional[List[float]] = None  # logprob of each generated token
    dists: Optional[List[TokenDist]] = None       # top-N distribution at each position
    finish_reason: Optional[str] = None           # "stop", "length", or a PLuMCoT marker


class Backend(Protocol):
    """What the engine and scorers need; VLLMBackend implements it and the
    test suite provides a fake."""
    tokenizer: object
    vocab_size: int
    num_logprobs: NumLogprobs
    stop_token_ids: set

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
        num_logprobs: NumLogprobs = MATCH_TOP_K,
        stop_token_ids: Optional[Iterable[int]] = None,
        model_name: Optional[str] = None,
    ):
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

    def _default_stop_ids(self, model_name: Optional[str]) -> set:
        ids = set()
        if getattr(self.tokenizer, "eos_token_id", None) is not None:
            ids.add(self.tokenizer.eos_token_id)
        # Chat models often end turns with a token other than eos (e.g.
        # <|eot_id|>, <|im_end|>); generation_config.json lists them.
        try:
            from transformers import GenerationConfig

            name = model_name or getattr(self.llm, "model_config", None)
            if name is None:
                raise ValueError("Model name could not be resolved. EOS IDs unknown.")
            elif not isinstance(name, str):
                name = name.model

            eos = GenerationConfig.from_pretrained(name).eos_token_id
            if eos is not None:
                ids.update(eos if isinstance(eos, (list, tuple)) else [eos])
        except Exception:
            pass
        return ids

    def check_logprob_cap(self) -> None:
        """Fail early, with the fix, if the LLM's max_logprobs is too small."""
        cap = getattr(getattr(self.llm, "model_config", None), "max_logprobs", None)
        n = self.num_logprobs
        if cap is None or cap == FULL_VOCAB or n == MATCH_TOP_K:
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
        if top_k is not None and top_k > 0:
            kwargs["top_k"] = top_k
        if stop:
            kwargs["stop"] = list(stop)
        if logprobs != "none":
            if logprobs == "dist" and self.num_logprobs == MATCH_TOP_K:
                raise RuntimeError(
                    'num_logprobs="top_k" has not been resolved. Pass this backend to '
                    "PLuMCoT (which resolves it) or give it an int."
                )
            # logprobs=0 returns only the sampled token's logprob; -1 returns
            # the full vocabulary.
            kwargs["logprobs"] = self.num_logprobs if logprobs == "dist" else 0
        params = self._SamplingParams(**kwargs) # type: ignore
        prompts = [self._TokensPrompt(prompt_token_ids=list(c)) for c in contexts]
        outputs = self.llm.generate(prompts, params, use_tqdm=False)
        return [[self._convert(o, logprobs) for o in req.outputs] for req in outputs]

    def _convert(self, out, mode: LogprobMode) -> Continuation:
        ids = list(out.token_ids)
        token_lps = dists = None
        if mode != "none" and out.logprobs is not None:
            token_lps = []
            dists = [] if mode == "dist" else None
            for tid, pos in zip(ids, out.logprobs):
                lp = pos.get(tid) if pos else None
                token_lps.append(
                    float(getattr(lp, "logprob", lp)) if lp is not None else float("nan")
                )
                if dists is not None:
                    dists.append(TokenDist.from_vllm(pos, self.vocab_size))
        return Continuation(ids, token_lps, dists, out.finish_reason)
