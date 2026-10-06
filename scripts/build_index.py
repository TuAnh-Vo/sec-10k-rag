"""Rebuild artifacts/ from scratch: download -> chunk -> filter -> embed -> save.

You only need this to reproduce the index yourself. The repo already ships the
finished artifacts/ folder, so the demo and evaluate.py work without it.

    export SEC_USER_AGENT="Your Name your.email@example.com"
    python scripts/build_index.py            # GPU strongly recommended (~1-2 min)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from rag import store
from rag.chunking import build_chunks
from rag.edgar import load_filings
from rag.retrieval import embed_passages, load_embedder

EXPECTED_CHUNKS = 5089      # what the notebook produced; see README "Reproducibility"

if __name__ == "__main__":
    filings = load_filings()
    print(f"{len(filings)} filings loaded")

    chunks = build_chunks(filings)
    if len(chunks) != EXPECTED_CHUNKS:
        print(f"WARNING: {len(chunks):,} chunks, expected {EXPECTED_CHUNKS:,}. "
              "Gold IDs in data/eval_questions.json may no longer match. "
              "Check library versions against requirements.txt.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = load_embedder(device)
    vectors = embed_passages(embedder, [c["text"] for c in chunks])
    store.save(chunks, vectors)
