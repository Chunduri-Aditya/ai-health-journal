#!/usr/bin/env python3
"""Sweep CHUNK_SIZE settings and report recall@k on a held-out query set.

Usage (requires running Postgres with pgvector):
  DATABASE_URL=... VECTOR_BACKEND=pgvector EMBEDDING_BACKEND=hash \\
    python scripts/chunk_sweep.py --queries evals/data/retrieval_queries.jsonl

Writes results to .runtime/chunk_sweep_results.json — paste measured numbers
into the resume bullet only after this has been run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vector_store.chunking import ChunkingConfig
from src.vector_store.embedders import build_embedder
from src.vector_store.pgvector_store import PgVectorStore


def load_queries(path: Path):
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--sizes", default="200,400,800")
    ap.add_argument("--overlap", type=int, default=80)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--out", type=Path, default=Path(".runtime/chunk_sweep_results.json"))
    args = ap.parse_args()

    url = os.environ.get("DATABASE_URL", "")
    if not url:
        print("DATABASE_URL required", file=sys.stderr)
        return 2

    queries = load_queries(args.queries)
    # Expected schema per line: {"query": "...", "relevant_ids": ["e1", ...], "corpus": [{"id","text"}, ...]}
    sizes = [int(x) for x in args.sizes.split(",")]
    results = []

    from types import SimpleNamespace

    os.environ.setdefault("ALLOW_HASH_EMBEDDER", "true")
    cfg = SimpleNamespace(embedding_backend=os.getenv("EMBEDDING_BACKEND", "hash"), embedding_dimension=int(os.getenv("EMBEDDING_DIMENSION", "384") or 384))
    embedder = build_embedder(cfg)

    for size in sizes:
        ns = f"sweep-{size}"
        store = PgVectorStore(
            database_url=url,
            embedder=embedder,
            chunking=ChunkingConfig(chunk_size=size, chunk_overlap=min(args.overlap, size - 1)),
            default_namespace=ns,
            allow_cloud=os.getenv("ALLOW_CLOUD_VECTORSTORE", "false").lower() == "true",
        )
        store.clear_namespace(ns)
        # Index first query set's corpus if present; else index synthetic from relevant texts
        corpus = queries[0].get("corpus") if queries else None
        if not corpus:
            print("queries file needs a corpus field on the first row", file=sys.stderr)
            return 2
        for doc in corpus:
            store.add_entry(doc["id"], doc["text"], namespace=ns)

        hits_at_k = 0
        total = 0
        for row in queries:
            total += 1
            hits = store.query(row["query"], top_k=args.k, namespace=ns)
            got = {h.id for h in hits}
            relevant = set(row.get("relevant_ids") or [])
            if got & relevant:
                hits_at_k += 1
        recall = hits_at_k / max(total, 1)
        results.append({"chunk_size": size, "overlap": args.overlap, "recall_at_k": recall, "k": args.k, "n": total})
        print(f"chunk_size={size} recall@{args.k}={recall:.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
