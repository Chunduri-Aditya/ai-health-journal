#!/usr/bin/env python3
"""Precision@k / recall@k / MRR retrieval eval over a labeled journal set.

Exercises the real local Chroma path (ONNX MiniLM embeddings) end to end in an
ephemeral persist dir, so it measures retrieval QUALITY that unit tests with
fakes cannot: does a query surface the right past entries, and rank them first?
Verification only — it does not touch app logic or the real ./storage/chroma.

Run:
    PYTHONPATH=. RETRIEVAL_ENABLED=true VECTOR_BACKEND=chroma \
        python evals/rag_retrieval_eval.py
or: make rag-eval

Exits non-zero if aggregate P@1 or MRR falls below the floors (a regression
gate). Tune floors via RAG_EVAL_P1_FLOOR / RAG_EVAL_MRR_FLOOR.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

# Repo root on path so `vector_store` imports whether run from evals/ or root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CASES_PATH = Path(__file__).resolve().parent / "rag_retrieval_cases.json"
P_AT_1_FLOOR = float(os.getenv("RAG_EVAL_P1_FLOOR", "0.80"))
MRR_FLOOR = float(os.getenv("RAG_EVAL_MRR_FLOOR", "0.80"))


def _avg(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def main() -> int:
    # Force the local path regardless of ambient env.
    os.environ["RETRIEVAL_ENABLED"] = "true"
    os.environ["VECTOR_BACKEND"] = "chroma"

    with CASES_PATH.open(encoding="utf-8") as handle:
        cases = json.load(handle)
    documents = cases["documents"]
    queries = cases["queries"]
    k = int(cases.get("k", 2))

    # Ephemeral store so the eval never pollutes real journal data.
    tmp_dir = tempfile.mkdtemp(prefix="rag_eval_")
    os.environ["CHROMA_PERSIST_DIR"] = tmp_dir
    namespace = "rag_eval"

    from vector_store.chroma_store import ChromaStore

    store = ChromaStore(default_namespace=namespace)
    try:
        for doc in documents:
            store.add_entry(entry_id=doc["id"], text=doc["text"], namespace=namespace)

        p1s: list[float] = []
        pks: list[float] = []
        recalls: list[float] = []
        rrs: list[float] = []
        rows: list[tuple] = []

        for case in queries:
            relevant = set(case["relevant_ids"])
            hits = store.query(case["query"], top_k=k, namespace=namespace)
            ids = [h.id for h in hits]

            hit_in_topk = [i for i in ids[:k] if i in relevant]
            p_at_1 = 1.0 if ids[:1] and ids[0] in relevant else 0.0
            p_at_k = len(hit_in_topk) / max(1, len(ids[:k]))
            recall_at_k = len(hit_in_topk) / max(1, len(relevant))
            reciprocal_rank = 0.0
            for rank, _id in enumerate(ids, start=1):
                if _id in relevant:
                    reciprocal_rank = 1.0 / rank
                    break

            p1s.append(p_at_1)
            pks.append(p_at_k)
            recalls.append(recall_at_k)
            rrs.append(reciprocal_rank)
            rows.append((p_at_1, p_at_k, recall_at_k, reciprocal_rank, case["query"], ids[:k]))
    finally:
        store.clear_namespace(namespace)
        shutil.rmtree(tmp_dir, ignore_errors=True)

    mean_p1, mean_pk, mean_recall, mean_mrr = _avg(p1s), _avg(pks), _avg(recalls), _avg(rrs)

    print(f"=== RAG retrieval eval (k={k}, {len(documents)} docs, {len(queries)} queries) ===")
    for p_at_1, p_at_k, recall_at_k, rr, query, ids in rows:
        print(
            f"  P@1={p_at_1:.2f}  P@{k}={p_at_k:.2f}  R@{k}={recall_at_k:.2f}  "
            f"RR={rr:.2f}  {query!r} -> {ids}"
        )
    print("--- aggregate ---")
    print(
        f"  P@1={mean_p1:.3f}  P@{k}={mean_pk:.3f}  "
        f"Recall@{k}={mean_recall:.3f}  MRR={mean_mrr:.3f}"
    )
    print(f"  floors: P@1>={P_AT_1_FLOOR}  MRR>={MRR_FLOOR}")

    passed = mean_p1 >= P_AT_1_FLOOR and mean_mrr >= MRR_FLOOR
    print("PASS ✅" if passed else "FAIL ❌")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
