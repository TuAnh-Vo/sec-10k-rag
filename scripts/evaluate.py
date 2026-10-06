"""Reproduce the retrieval ablation table (notebook section 13).

    python scripts/evaluate.py      # writes results/results_validation.csv
"""
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from rag import store
from rag.config import MMR_LAMBDA, ROOT
from rag.evaluation import gold_ranks, load_questions, metrics
from rag.retrieval import Retriever, load_embedder

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    chunks, vectors = store.load()
    r = Retriever(chunks, vectors, load_embedder(device))
    questions = load_questions()

    missing = [x["gold"] for x in questions if x["gold"] not in r.text]
    assert not missing, f"gold ids not in corpus: {missing}"

    configs = {
        "1_dense_only":  lambda q: r.search(q, use_bm25=False),
        "2_+bm25_rrf":   lambda q: r.search(q),
        "3_+mmr":        lambda q: r.search(q, use_mmr=True, lam=MMR_LAMBDA),
        "4_+rerank":     lambda q: r.search(q, use_rerank=True),
        "5_+mmr+rerank": lambda q: r.search(q, use_mmr=True, use_rerank=True, lam=MMR_LAMBDA),
    }

    rows = []
    for name, fn in configs.items():
        fn(questions[0]["q"])                          # warm-up (loads reranker once)
        lat = []
        for x in questions:
            t0 = time.perf_counter(); fn(x["q"])
            lat.append((time.perf_counter() - t0) * 1000)
        ranks = gold_ranks(fn, questions)
        m = metrics(ranks)
        rows.append({"config": name, **m, "p50_ms": round(np.median(lat)),
                     "p95_ms": round(np.percentile(lat, 95))})
        print(f"{name:<15} " + "  ".join(f"{k}={v:.3f}" for k, v in m.items() if k != "n")
              + f"   ranks={ranks}")

    out = ROOT / "results" / "results_validation.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out}  (latency depends on your hardware)")
