# PLuMCoT
A harness implementation of the lookahead-based **PLuMCoT** (Primed Lookahead using Multiple Chains-of-Thought) LLM autoregressive generation framework.

## Framework Definition
PLuMCoT is defined primarily by its inclusion of lookahead and score processes at many text generation iterations. On iterations that use PLuMCoT, a set of candidate tokens are filtered (typically but not necessarily via top-k and/or top-p) from the full vocabulary. Each of these is subjected to a Monte Carlo free generation in parallel to obtain a set of lookaheads. Each of these lookaheads are provided a score by some scoring mechanism. The selected token is chosen (or, optionally, sampled) based on the lookahead scores (optionally in concert with the candidates' initial predicted probabilities).

## Pseudocode
```text
FUNCTION plumcot(llm, context):
    response <- []
    WHILE True DO
        IF stop_criteria(context, response)
            RETURN response
        ELSE
            IF trigger?(context) THEN
                logits <- llm.forward(context)
                filtered <- filter(logits)
                scores <- {}
                FOR EACH candidante IN filtered DO          // in parallel
                    L <- llm_generate(context + candidate)  // normal (non-PLuMCoT) multi-step generation loop
                    s <- score(context + candidate + L)     // + is concatenation
                    scores.add(s, candidate)
                selected <- select(scores)
            ELSE
                selected <- llm_gen(context)                // normal (non-PLuMCoT) generation step
        response <- response + selected
        context <- context + selected
```

## Predicted Benefits
### Flexibility/customizability
The (partial) outcome-driven nature of PLuMCoT suggests that different behaviors can be induced in a given model without additional training by selecting a score function that rewards or punishes those behaviors. This may include benefits like custom alignment protocols to suit the specific use-case and user. In principle, the chosen score function can be contextually determined and variable per iteration, meaning that differing behaviors can be induced at different points in the model's response. This might allow intelligently variable reward structures rewarding congeniality and helpfulness when generating chat-like text and instead rewarding syntactic correctness when generating code-like text. Story generation may benefit by allowing character-specific writing style verifiers that ensure that different characters have identifiable and consistent personalities.

### Training-free Alignment
By using a harmful output detection based scoring function, this process may steer models away from likely harmful output multiple iterations in advance. This is hypothesized to reduce model harmful behavior without substantial loss in helpfulness, as the model can simultaneously rewarded for helpful output while hard-gating against harmful responses. In addition, this improved alignment compliance, if successful, would be achieved with zero alignment fine-tuning, providing at least two benefits. First, successful alignment scorers can be cheaply transferred across models, allowing the encoded protocols to be enforced on subsequent models without redundant training. Second, alignment protocols can be "hot-swapped", allowing iterative or context-specific change in model allowed behavior based on the task in question or as user alignment expectations evolve.

# Source Publications and Citations
* PLuMCoT was first proposed and described in my dissertation, the relevant portion of which has been extracted and uploaded [here](PLuMCoT_dissertation.pdf). This link will be replaced once the dissertation is publicly published.
  * Moore, K. (2026). Human-Analogous Uncertainty Behavior: Cognitive Studies of Large Language Models [Doctoral dissertation, Vanderbilt University]. Link TBA
* PLuMCoT was first publicly published in the proceedings of the [Advances in Cognitive Systems (ACS) 2026 conference](http://cogsys.org/conference/2026/). A link to this publication will be added once proceedings have been finalized.
  * Fisher, D., Moore, K., & Roberts, J. (2026). State-Space Search and Transformer-Based AI Hybridizations. In *Thirteenth Annual Conference on Advances in Cognitive Systems*.
