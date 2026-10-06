"""Retrieval metrics: hit@k and MRR against one gold chunk per question."""
import json

from .config import ROOT

EVAL_PATH = ROOT / "data" / "eval_questions.json"


def load_questions():
    return json.loads(EVAL_PATH.read_text(encoding="utf-8"))


def gold_ranks(search_fn, questions):
    ranks = []
    for x in questions:
        got = search_fn(x["q"])
        ranks.append(got.index(x["gold"]) + 1 if x["gold"] in got else None)
    return ranks


def metrics(ranks, k_values=(1, 3, 5, 10)):
    n = len(ranks)
    out = {"n": n}
    for k in k_values:
        # hit@k is the CEILING on the generator: an unretrieved chunk cannot be cited.
        out[f"hit@{k}"] = sum(1 for r in ranks if r and r <= k) / n
    # MRR: rank 1 -> 1.0, rank 2 -> 0.5, miss -> 0. The metric a reranker moves.
    out["mrr"] = sum(1 / r for r in ranks if r) / n
    return out
