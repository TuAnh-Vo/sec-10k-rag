"""Public demo: ask a question, get an answer cited to SEC 10-K excerpts.

Runs on a Hugging Face Space with ZeroGPU hardware. ZeroGPU rules this file follows:
  * models are placed on "cuda" at MODULE level (ZeroGPU emulates CUDA here),
  * the work that needs a real GPU lives inside a function decorated @spaces.GPU,
    which borrows a GPU for that call only and gives it back afterwards.
@spaces.GPU does nothing outside ZeroGPU, so the same file also runs on your own
GPU machine. On a CPU-only laptop, set GENERATOR=Qwen/Qwen3-1.7B to test the UI.
"""
import os

import gradio as gr
import spaces
import torch

from rag import store
from rag.config import GENERATOR_NAME
from rag.evaluation import load_questions
from rag.generation import LocalLLM, answer, format_sources
from rag.retrieval import Retriever, load_embedder

ON_ZEROGPU = os.environ.get("SPACES_ZERO_GPU") is not None
DEVICE = "cuda" if (ON_ZEROGPU or torch.cuda.is_available()) else "cpu"
GENERATOR = os.environ.get("GENERATOR", GENERATOR_NAME)

# ---- load everything once, at startup ---------------------------------------
chunks, vectors = store.load()
retriever = Retriever(chunks, vectors, load_embedder(DEVICE))
llm = LocalLLM(GENERATOR, quantize_4bit=False, device=DEVICE)
print(f"ready: {len(chunks):,} chunks | generator {GENERATOR} | device {DEVICE}")


@spaces.GPU(duration=60)            # a short duration gets better queue priority
def respond(message, history):      # Gradio always passes history; each question stands alone
    if not message or not message.strip():
        return "Please type a question about the filings."
    res = answer(retriever, llm, message.strip())
    sources = format_sources(retriever, res, markdown=True).replace("\n", "  \n")
    if res["abstained"]:
        return ("The retrieved filing excerpts do not answer this question, so I won't guess.\n\n"
                f"---\n**Excerpts that were checked**  \n{sources}")
    return f"{res['answer']}\n\n---\n**Sources**  \n{sources}"


RESULTS_MD = """
| Configuration | hit@1 | hit@3 | hit@5 | hit@10 | MRR | p50 latency |
|---|---|---|---|---|---|---|
| 1 · dense only | 0.30 | 0.50 | 0.80 | 0.80 | 0.470 | 14 ms |
| **2 · dense + BM25 (RRF) — used here** | **0.50** | **0.90** | **0.90** | **1.00** | **0.694** | 24 ms |
| 3 · + MMR | 0.50 | 0.90 | 0.90 | 1.00 | 0.694 | 26 ms |
| 4 · + cross-encoder rerank | 0.50 | 0.70 | 0.70 | 0.90 | 0.617 | 114 ms |
| 5 · + MMR + rerank | 0.50 | 0.70 | 0.70 | 0.90 | 0.617 | 116 ms |

Measured on 10 hand-written questions with one gold chunk each. That is a small set:
treat the numbers as directional, not final.
"""

ABOUT_MD = f"""
**Corpus:** the 10-K annual reports of Apple, Microsoft, NVIDIA, Alphabet and Amazon,
3 fiscal years each (15 filings, {len(chunks):,} chunks).

**How an answer is made:** your question is searched two ways — by meaning
(`bge-base-en-v1.5` embeddings) and by exact words (BM25). The two ranked lists are
merged with Reciprocal Rank Fusion, the top 5 excerpts are numbered `[1]`–`[5]`, and
`{GENERATOR}` writes a 2–4 sentence answer that must cite an excerpt after every fact.
If the excerpts don't contain the answer, the model is told to refuse instead of guessing.

**Tips:** ask complete questions that name the company (follow-ups like
"and Microsoft?" are not understood). Always check the cited excerpt before trusting a number.
"""

examples = [
    "What are the main risk factors related to supply chain disruption?",
    load_questions()[0]["q"],
    "What will Apple's share price be at the end of next year?",
]

with gr.Blocks(title="SEC 10-K RAG") as demo:
    gr.Markdown("# SEC 10-K RAG\nAsk about the annual reports of AAPL, MSFT, NVDA, "
                "GOOGL and AMZN. Every answer cites the filing excerpts it used.")
    gr.ChatInterface(fn=respond, examples=examples, cache_examples=False)
    with gr.Accordion("How it works", open=False):
        gr.Markdown(ABOUT_MD)
    with gr.Accordion("Retrieval results (ablation)", open=False):
        gr.Markdown(RESULTS_MD)

if __name__ == "__main__":
    demo.launch()
