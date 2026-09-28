"""Synthetic query generation. §7.3: the single highest-leverage component of the pipeline.

Sampling is temperature 0.9, top-p 0.95, 16 per unit, filtered down to 10. Diversity is
the point. The generator is forced through a mix of query types, because an unbalanced
generator produces uniformly well-formed questions and the model then fails on real
keyword queries. Every generated query records which generator and prompt produced it,
so evaluation can be broken down per generator to detect a monoculture (§7.5).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

QUERY_TYPES = ("factoid", "procedural", "comparative", "symptom_framed", "keyword")

# Two prompt families at minimum (§7.5). Instruction-tuned generators use these; plain
# doc2query seq2seq models ignore the instruction and take the passage alone.
PROMPTS = {
    "a": {
        "factoid": "Write a short factual question this passage answers.\n\n{text}",
        "procedural": "Write a how-to question this passage answers.\n\n{text}",
        "comparative": "Write a question comparing two things discussed in this passage.\n\n{text}",
        "symptom_framed": "Describe a problem or symptom someone would search for that this passage resolves.\n\n{text}",
        "keyword": "Write a 2 to 5 word search engine query for this passage. No question mark.\n\n{text}",
    },
    "b": {
        "factoid": "Passage:\n{text}\n\nA user asks a specific question answered above:",
        "procedural": "Passage:\n{text}\n\nA user asks how to do something described above:",
        "comparative": "Passage:\n{text}\n\nA user asks which option is better, as discussed above:",
        "symptom_framed": "Passage:\n{text}\n\nA user describes what is going wrong, in their own words:",
        "keyword": "Passage:\n{text}\n\nA user types terse keywords into a search box:",
    },
}


@dataclass(frozen=True)
class Generated:
    text: str
    query_type: str
    generator: str
    prompt: str


class Doc2Query:
    def __init__(self, model_name: str, device: str = "cpu", instruction: bool = False, max_input_tokens: int = 512):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.name, self.device, self.instruction = model_name, device, instruction
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device).eval()
        self.max_input = max_input_tokens

    def generate(self, text: str, n: int = 16, temperature: float = 0.9, top_p: float = 0.95,
                 prompt_family: str = "a", seed: int = 0) -> list[Generated]:
        import torch

        torch.manual_seed(seed)
        out: list[Generated] = []
        types = QUERY_TYPES if self.instruction else ("any",)
        per_type = -(-n // len(types))
        for qt in types:
            src = PROMPTS[prompt_family][qt].format(text=text) if self.instruction else text
            enc = self.tok(src, return_tensors="pt", truncation=True, max_length=self.max_input).to(self.device)
            with torch.inference_mode():
                ids = self.model.generate(**enc, do_sample=True, temperature=temperature, top_p=top_p,
                                          num_return_sequences=per_type, max_new_tokens=48)
            out += [Generated(t.strip(), qt, self.name, prompt_family)
                    for t in self.tok.batch_decode(ids, skip_special_tokens=True) if t.strip()]
        return out[:n]


def interleave_types(generated: Sequence[Generated]) -> list[Generated]:
    """Round-robin by query type so filtering with a keep-cap cannot starve a type."""
    buckets: dict[str, list[Generated]] = {}
    for g in generated:
        buckets.setdefault(g.query_type, []).append(g)
    out = []
    while any(buckets.values()):
        for b in buckets.values():
            if b:
                out.append(b.pop(0))
    return out
