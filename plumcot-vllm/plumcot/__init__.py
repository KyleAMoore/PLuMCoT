"""PLuMCoT: Primed Lookahead using Multiple Chains of Thought (vLLM sketch)."""
from .backend import Backend, Continuation, VLLMBackend
from .config import (FULL_VOCAB, MATCH_TOP_K, FilterConfig, LookaheadConfig, PLuMCoTConfig,
                     SelectConfig, TriggerConfig, resolve_num_logprobs)
from .dist import TokenDist, top_k_top_p
from .engine import PLuMCoT, PLuMCoTResult, StepRecord, calibrate_threshold
from .scorers import (DEFAULT_ALIGNMENT_POLICY, Branch, CompositeScorer,
                      LogLikelihoodScorer, ScoreContext, Scorer, SelfCertaintyScorer,
                      VerdictJudgeScorer, guard_messages, llama_guard_scorer,
                      policy_judge_scorer, policy_messages, qwen3guard_scorer)

__all__ = [
    "Backend", "Continuation", "VLLMBackend",
    "FilterConfig", "LookaheadConfig", "PLuMCoTConfig", "SelectConfig", "TriggerConfig",
    "FULL_VOCAB", "MATCH_TOP_K", "resolve_num_logprobs",
    "TokenDist", "top_k_top_p",
    "PLuMCoT", "PLuMCoTResult", "StepRecord", "calibrate_threshold",
    "DEFAULT_ALIGNMENT_POLICY", "Branch", "CompositeScorer", "LogLikelihoodScorer",
    "ScoreContext", "Scorer", "SelfCertaintyScorer", "VerdictJudgeScorer",
    "guard_messages", "llama_guard_scorer", "policy_judge_scorer", "policy_messages",
    "qwen3guard_scorer",
]
