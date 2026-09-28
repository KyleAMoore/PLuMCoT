"""PLuMCoT: Primed Lookahead using Multiple Chains of Thought (vLLM sketch)."""
from .backend import Backend, Continuation, VLLMBackend
from .base import Stage, ValidationContext
from .config import (AUTO, FULL_VOCAB, DecodingConfig, PLuMCoTConfig,
                     resolve_num_logprobs)
from .dist import TokenDist, top_k_top_p
from .engine import PLuMCoT, PLuMCoTResult, StepRecord
from .scorers import (DEFAULT_ALIGNMENT_POLICY, Branch, CompositeScorer,
                      LogLikelihoodScorer, ScoreContext, Scorer, SelfCertaintyScorer,
                      VerdictJudgeScorer, fill_nan, guard_messages, llama_guard_scorer,
                      policy_judge_scorer, policy_messages, qwen3guard_scorer)
from .stages import (MATCH_DECODING, AlwaysTrigger, BlendSelect, Candidates, Configure,
                     EntropyTrigger, Filter, FixedConfigure, Lookahead, NeverTrigger,
                     RolloutLookahead, SampleAggregator, Select, StageBundle, StepContext,
                     TopKTopPFilter, Trigger, base_policy_prior, calibrate_entropy_threshold)
 
__all__ = [
    "Backend", "Continuation", "VLLMBackend",
    "Stage", "ValidationContext",
    "AUTO", "FULL_VOCAB", "DecodingConfig", "PLuMCoTConfig", "resolve_num_logprobs",
    "TokenDist", "top_k_top_p",
    "PLuMCoT", "PLuMCoTResult", "StepRecord",
    "DEFAULT_ALIGNMENT_POLICY", "Branch", "CompositeScorer", "LogLikelihoodScorer",
    "ScoreContext", "Scorer", "SelfCertaintyScorer", "VerdictJudgeScorer", "fill_nan",
    "guard_messages", "llama_guard_scorer", "policy_judge_scorer", "policy_messages",
    "qwen3guard_scorer",
    "MATCH_DECODING", "AlwaysTrigger", "BlendSelect", "Candidates", "Configure",
    "EntropyTrigger", "Filter", "FixedConfigure", "Lookahead", "NeverTrigger",
    "RolloutLookahead", "SampleAggregator", "Select", "StageBundle", "StepContext",
    "TopKTopPFilter", "Trigger", "base_policy_prior", "calibrate_entropy_threshold",
]
