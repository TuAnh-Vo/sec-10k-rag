"""Save and load the two files that ARE the index.

    artifacts/chunks.jsonl     one line per chunk: chunk_id, text, meta
    artifacts/embeddings.npy   float32 [n_chunks, 768], unit-length rows,
                               row i belongs to line i of chunks.jsonl

Why not save the LangChain FAISS folder? Loading it needs pickle
(allow_dangerous_deserialization=True), which runs code from a file. Plain
JSON + NumPy needs no trust, opens anywhere, and is easy to inspect.
"""
import json

import numpy as np

from .config import ARTIFACTS, EMBEDDING_MODEL_NAME


def save(chunks: list[dict], vectors: np.ndarray, out_dir=ARTIFACTS) -> None:
    assert len(chunks) == len(vectors), "chunks and vectors desync"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps({"chunk_id": c["chunk_id"], "text": c["text"],
                                "meta": c["meta"]}, ensure_ascii=False) + "\n")
    np.save(out_dir / "embeddings.npy", vectors.astype("float32"))
    (out_dir / "info.json").write_text(json.dumps(
        {"n_chunks": len(chunks), "dim": int(vectors.shape[1]),
         "embedding_model": EMBEDDING_MODEL_NAME, "normalized": True}, indent=2))
    print(f"saved {len(chunks):,} chunks -> {out_dir}")


def load(in_dir=ARTIFACTS) -> tuple[list[dict], np.ndarray]:
    with open(in_dir / "chunks.jsonl", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f if line.strip()]
    vectors = np.load(in_dir / "embeddings.npy")

    # The same guard as notebook section 8: every store must agree.
    assert len(chunks) == len(vectors), (
        f"DESYNC: {len(chunks)} chunks vs {len(vectors)} vectors")
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids)), "duplicate chunk_id in artifacts"
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3), "vectors are not unit length"
    return chunks, vectors
