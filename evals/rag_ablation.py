#!/usr/bin/env python3
"""Retrieval ablation: dense vs BM25 vs hybrid (RRF), same corpus, same metrics.

Why this exists
---------------
`rag_retrieval_eval.py` answers "is retrieval above a floor?". It cannot answer
"is this the best retrieval we could be running?", because it measures exactly
one configuration. A floor that the incumbent already clears is not evidence
that the incumbent is good -- and on this corpus the incumbent scores P@1=1.000
against a floor of 0.80, so the gate is incapable of failing.

This harness measures several configurations against each other on identical
data, which is the only way a change to retrieval can be justified as an
improvement rather than asserted as one.

Reporting only: it never gates CI. A measurement tool that can fail a build
invites tuning the measurement instead of the system.

Run:
    PYTHONPATH=. python evals/rag_ablation.py
    PYTHONPATH=. python evals/rag_ablation.py --cases evals/rag_retrieval_cases.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.retrieval_strategies import (  # noqa: E402
    BM25,
    Doc,
    Strategy,
    bm25_strategy,
    dense_strategy,
    hybrid_strategy,
    ollama_embedder_strategy,
    valence_aware_strategy,
    with_valence_rerank,
)

DEFAULT_CASES = Path(__file__).resolve().parent / "rag_retrieval_cases.json"


def _avg(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def score_strategy(
    strategy: Strategy,
    queries: Sequence[dict],
    k: int,
) -> Dict[str, object]:
    """Compute P@1, P@k, Recall@k and MRR for one strategy.

    Metric definitions are kept identical to rag_retrieval_eval.py so the two
    scripts' numbers can be read side by side.
    """
    p1s: List[float] = []
    pks: List[float] = []
    recalls: List[float] = []
    rrs: List[float] = []
    per_query: List[dict] = []

    for case in queries:
        relevant = set(case["relevant_ids"])
        ids = strategy(case["query"], k)

        hit_in_topk = [i for i in ids[:k] if i in relevant]
        p_at_1 = 1.0 if ids[:1] and ids[0] in relevant else 0.0
        p_at_k = len(hit_in_topk) / max(1, len(ids[:k]))
        recall_at_k = len(hit_in_topk) / max(1, len(relevant))

        reciprocal_rank = 0.0
        for rank, doc_id in enumerate(ids, start=1):
            if doc_id in relevant:
                reciprocal_rank = 1.0 / rank
                break

        p1s.append(p_at_1)
        pks.append(p_at_k)
        recalls.append(recall_at_k)
        rrs.append(reciprocal_rank)
        per_query.append(
            {
                "query": case["query"],
                "trap": case.get("trap", "untagged"),
                "returned": ids[:k],
                "relevant": sorted(relevant),
                "p_at_1": p_at_1,
                "p_at_k": p_at_k,
                "recall_at_k": recall_at_k,
                "rr": reciprocal_rank,
            }
        )

    return {
        "p_at_1": _avg(p1s),
        "p_at_k": _avg(pks),
        "recall_at_k": _avg(recalls),
        "mrr": _avg(rrs),
        "per_query": per_query,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--k", type=int, default=None, help="override k from the cases file")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument(
        "--embed-model",
        default="nomic-embed-text",
        help="Ollama embedding model to compare against Chroma's default MiniLM",
    )
    parser.add_argument(
        "--no-ollama",
        action="store_true",
        help="skip the Ollama embedder comparison (offline runs)",
    )
    args = parser.parse_args()

    os.environ["RETRIEVAL_ENABLED"] = "true"
    os.environ["VECTOR_BACKEND"] = "chroma"

    with args.cases.open(encoding="utf-8") as handle:
        cases = json.load(handle)
    documents = cases["documents"]
    queries = cases["queries"]
    k = args.k or int(cases.get("k", 2))

    # Ephemeral store so the ablation never touches real journal data.
    tmp_dir = tempfile.mkdtemp(prefix="rag_ablation_")
    os.environ["CHROMA_PERSIST_DIR"] = tmp_dir
    namespace = "rag_ablation"

    from vector_store.chroma_store import ChromaStore

    store = ChromaStore(default_namespace=namespace)
    results: Dict[str, Dict[str, object]] = {}

    try:
        for doc in documents:
            store.add_entry(entry_id=doc["id"], text=doc["text"], namespace=namespace)

        bm25 = BM25([Doc(id=d["id"], text=d["text"]) for d in documents])

        strategies = {
            "dense (current)": dense_strategy(store, namespace),
            "bm25 (control)": bm25_strategy(bm25),
            "hybrid rrf": hybrid_strategy(store, namespace, bm25),
            "dense+valence": valence_aware_strategy(store, namespace),
        }

        # Alternative embedder, skipped cleanly when the Ollama daemon or the
        # embedding model is unavailable so the rest of the ablation still runs.
        if not args.no_ollama:
            docs = [Doc(id=d["id"], text=d["text"]) for d in documents]
            try:
                strategies[f"{args.embed_model} raw"] = ollama_embedder_strategy(
                    docs, model=args.embed_model
                )
                strategies[f"{args.embed_model} prefixed"] = ollama_embedder_strategy(
                    docs,
                    model=args.embed_model,
                    doc_prefix="search_document: ",
                    query_prefix="search_query: ",
                )
                # Does the valence reranker still add anything once the
                # underlying retrieval is strong, or was it compensating for a
                # weak embedder?
                strategies[f"{args.embed_model} + valence"] = with_valence_rerank(
                    ollama_embedder_strategy(docs, model=args.embed_model), docs
                )
            except Exception as exc:  # noqa: BLE001 - reported, not swallowed
                print(
                    f"[skip] {args.embed_model} unavailable "
                    f"({type(exc).__name__}); run `ollama pull {args.embed_model}`\n"
                )

        for name, strategy in strategies.items():
            results[name] = score_strategy(strategy, queries, k)
    finally:
        store.clear_namespace(namespace)
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # ── Report ────────────────────────────────────────────────────────────────
    print(
        f"=== Retrieval ablation "
        f"(k={k}, {len(documents)} docs, {len(queries)} queries) ===\n"
    )
    header = f"{'strategy':<18} {'P@1':>7} {f'P@{k}':>7} {f'R@{k}':>7} {'MRR':>7}"
    print(header)
    print("-" * len(header))
    for name, res in results.items():
        print(
            f"{name:<18} {res['p_at_1']:>7.3f} {res['p_at_k']:>7.3f} "
            f"{res['recall_at_k']:>7.3f} {res['mrr']:>7.3f}"
        )

    # Per-trap breakdown. An aggregate hides *which kind* of confusion a
    # strategy fails on, and that is the only part that tells you what to fix.
    traps = []
    for entry in results["dense (current)"]["per_query"]:
        if entry["trap"] not in traps:
            traps.append(entry["trap"])

    if traps != ["untagged"]:
        print(f"\n--- Recall@{k} by trap category ---")
        trap_header = f"{'strategy':<18}" + "".join(f"{t[:15]:>17}" for t in traps)
        print(trap_header)
        print("-" * len(trap_header))
        for name, res in results.items():
            cells = ""
            for trap in traps:
                vals = [
                    q["recall_at_k"] for q in res["per_query"] if q["trap"] == trap
                ]
                cells += f"{_avg(vals):>17.3f}"
            print(f"{name:<18}{cells}")
        counts = "".join(
            f"{sum(1 for q in results['dense (current)']['per_query'] if q['trap'] == t):>17}"
            for t in traps
        )
        print(f"{'(n queries)':<18}{counts}")

    # Per-query diff against the incumbent, so a headline average that hides a
    # regression on individual queries is still visible.
    baseline = results["dense (current)"]["per_query"]
    print("\n--- per-query vs dense (current) ---")
    for name, res in results.items():
        if name == "dense (current)":
            continue
        print(f"\n  {name}")
        for base, cand in zip(baseline, res["per_query"]):
            delta = cand["recall_at_k"] - base["recall_at_k"]
            marker = "  " if delta == 0 else ("↑ " if delta > 0 else "↓ ")
            print(
                f"   {marker}R@{k} {base['recall_at_k']:.2f} -> {cand['recall_at_k']:.2f}  "
                f"{base['returned']} -> {cand['returned']}   {cand['query'][:44]!r}"
            )

    best = max(results.items(), key=lambda kv: (kv[1]["recall_at_k"], kv[1]["mrr"]))
    print(f"\nbest by Recall@{k}, MRR tiebreak: {best[0]}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("w", encoding="utf-8") as handle:
            json.dump(
                {"k": k, "n_docs": len(documents), "n_queries": len(queries), "results": results},
                handle,
                indent=2,
            )
        print(f"wrote {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
