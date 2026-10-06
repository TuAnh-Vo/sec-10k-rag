"""Turn retrieved chunk IDs into a cited answer (notebook sections 14-16).

question -> (1) retrieve -> (2) numbered excerpts -> (3) prompt -> (4) generate
"""
from .config import GENERATOR_NAME, MAX_NEW_TOKENS, TOP_K_CONTEXT

SENTINEL = "INSUFFICIENT_CONTEXT"

SYSTEM_PROMPT = f"""You answer questions about SEC 10-K filings using ONLY the numbered excerpts provided.

Rules:
1. End every factual sentence with its source marker, like [2].
2. Use no information beyond the excerpts, even if you know it.
3. If the excerpts do not answer the question, reply with exactly: {SENTINEL}
4. Copy figures exactly as written. Do not calculate, convert, or estimate.

Answer in 2-4 sentences."""

USER_TEMPLATE = """Excerpts:
{context}

Question: {question}
Answer:"""


class LocalLLM:
    """A Qwen instruct model with greedy decoding (same question -> same answer).

    quantize_4bit=True  : NF4 via bitsandbytes, ~6 GB VRAM (Kaggle T4 / P100).
    quantize_4bit=False : bf16, ~17 GB VRAM (the Hugging Face ZeroGPU demo).
    """

    def __init__(self, model_name=GENERATOR_NAME, quantize_4bit=False, device="cuda"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = model_name
        self.tok = AutoTokenizer.from_pretrained(model_name)
        if quantize_4bit:
            from transformers import BitsAndBytesConfig
            q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=torch.float16,
                                   bnb_4bit_use_double_quant=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, quantization_config=q, device_map="auto").eval()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, dtype=torch.bfloat16).to(device).eval()

    def complete(self, messages, max_new_tokens=MAX_NEW_TOKENS):
        import torch

        prompt = self.tok.apply_chat_template(messages, tokenize=False,
                                              add_generation_prompt=True,
                                              enable_thinking=False)   # no <think> block
        enc = self.tok(prompt, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            out = self.model.generate(**enc, max_new_tokens=max_new_tokens,
                                      do_sample=False,
                                      pad_token_id=self.tok.eos_token_id)
        new_tokens = out[0][enc["input_ids"].shape[1]:]
        text = self.tok.decode(new_tokens, skip_special_tokens=True).strip()
        return text.split("</think>")[-1].strip()


def build_context(retriever, chunk_ids):
    """Numbered excerpts [1]..[k], each headed with ticker and fiscal year."""
    m = retriever.meta
    return "\n\n".join(
        f"[{i}] {m[c]['ticker']} · FY{m[c]['fiscal_year']} · {m[c]['company']}\n"
        f"{retriever.text[c]}"
        for i, c in enumerate(chunk_ids, 1))


def answer(retriever, llm, question, k=TOP_K_CONTEXT):
    ids = retriever.search(question, top_n=k)          # hybrid: dense + BM25 + RRF
    text = llm.complete([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(
            context=build_context(retriever, ids), question=question)},
    ])
    return {"question": question, "answer": text, "chunk_ids": ids,   # [i] == ids[i-1]
            "abstained": SENTINEL in text.strip().upper()[:60]}


def format_sources(retriever, res, markdown=False):
    m = retriever.meta
    lines = []
    for i, c in enumerate(res["chunk_ids"], 1):
        label = f"[{i}] {m[c]['ticker']} FY{m[c]['fiscal_year']} · {c}"
        lines.append(f"{label} · [10-K on EDGAR]({m[c]['source_url']})" if markdown
                     else f"{label} · {m[c]['source_url']}")
    return "\n".join(lines)
