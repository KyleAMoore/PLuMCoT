"""A GPU-free stand-in for VLLMBackend: deterministic toy distributions keyed
on the last few context tokens, plus a toy judge."""
from __future__ import annotations
 
import numpy as np
 
from plumcot.backend import Continuation
from plumcot.dist import TokenDist
 
V = 40
EOS = 0
HARM = 13  # the fake judge considers any response containing this token unsafe
 
 
class FakeTokenizer:
    SPECIAL = {0: "<eos>", 1: "safe", 2: "unsafe", 3: "Yes", 4: "No", 5: "Safety", 6: ":",
               7: "\n\n"}
    eos_token_id = EOS
    chat_template = "fake"
 
    def __len__(self):
        return V
 
    def decode(self, ids, skip_special_tokens=False):
        out = []
        for i in ids:
            if skip_special_tokens and i == EOS:
                continue
            out.append(self.SPECIAL.get(int(i), f" w{int(i)}"))
        return "".join(out)
 
    def encode(self, text):
        return [8 + (ord(c) % 30) for c in text][:24]
 
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, **kw):
        text = "|".join(
            m["content"] if isinstance(m["content"], str) else m["content"][0]["text"]
            for m in messages
        )
        if not tokenize:
            return text
        ids = self.encode(text[-24:])
        if add_generation_prompt:
            ids.append(9)
        return ids
 
 
def _raw_logits(ctx):
    rng = np.random.default_rng(hash(tuple(int(t) for t in ctx[-3:])) % (2**32))
    scale = rng.choice([0.3, 1.5, 4.0])  # varies entropy step to step
    logits = rng.normal(size=V) * scale
    logits[EOS] -= 2.0
    return logits
 
 
def _logprobs(logits):
    z = logits - logits.max()
    return z - np.log(np.exp(z).sum())
 
 
class FakeBackend:
    def __init__(self, num_logprobs=20):
        self.tokenizer = FakeTokenizer()
        self.vocab_size = V
        self.num_logprobs = num_logprobs
        self.stop_token_ids = {EOS}
        self.calls = 0
 
    def check_logprob_cap(self):
        """The fake serves any number of logprobs."""
 
    def _sample(self, lps, temperature, top_k, top_p, rng):
        if temperature == 0.0:
            return int(np.argmax(lps))
        order = np.argsort(-lps)
        if top_k:
            order = order[:top_k]
        p = np.exp(lps[order] / temperature)
        p /= p.sum()
        if top_p < 1.0:
            n = int(np.searchsorted(np.cumsum(p), top_p - 1e-9)) + 1
            order, p = order[:n], p[:n] / p[:n].sum()
        return int(rng.choice(order, p=p))
 
    def _dist(self, lps, chosen):
        n = len(lps) if self.num_logprobs == -1 else self.num_logprobs
        top = list(np.argsort(-lps)[:n])
        if chosen not in top:
            top.append(chosen)
        return TokenDist.from_items(((int(t), float(lps[t])) for t in top), V)
 
    def generate(self, contexts, *, n=1, max_tokens, temperature, top_k=None, top_p=1.0,
                 stop=(), logprobs="none", seed=None):
        self.calls += 1
        rng = np.random.default_rng(seed)
        results = []
        for ctx in contexts:
            outs = []
            for _ in range(n):
                seq, tlps, dists, reason = [], [], [], "length"
                cur = list(ctx)
                for _ in range(max_tokens):
                    lps = _logprobs(_raw_logits(cur))
                    t = self._sample(lps, temperature, top_k, top_p, rng)
                    seq.append(t)
                    tlps.append(float(lps[t]))
                    dists.append(self._dist(lps, t))
                    cur.append(t)
                    if t in self.stop_token_ids:
                        reason = "stop"
                        break
                outs.append(Continuation(
                    seq,
                    tlps if logprobs != "none" else None,
                    dists if logprobs == "dist" else None,
                    reason,
                ))
            results.append(outs)
        return results
 
 
class FakeGuard(FakeBackend):
    """Emits '\\n\\n' then 'safe'/'unsafe'. P(unsafe) is high iff the response
    text (which the scorer embeds in the prompt) contained the HARM token.
    The tokenizer's fake chat template keeps only the last 24 chars, so the
    guard is told directly via `last_texts`."""
 
    def __init__(self):
        super().__init__(num_logprobs=5)
        self.last_texts = []
 
    def generate(self, contexts, **kw):
        self.calls += 1
        out = []
        for text in self.pop_texts(len(contexts)):
            p_unsafe = 0.95 if f"w{HARM}" in text else 0.05
            d0 = TokenDist.from_items([(7, 0.0)], V)
            d1 = TokenDist.from_items([(1, np.log(1 - p_unsafe)), (2, np.log(p_unsafe))], V)
            tok = 2 if p_unsafe > 0.5 else 1
            out.append([Continuation([7, tok], None, [d0, d1], "stop")])
        return out
 
    def pop_texts(self, n):
        texts, self.last_texts = self.last_texts[:n], self.last_texts[n:]
        return texts
