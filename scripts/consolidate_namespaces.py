#!/usr/bin/env python3
"""Merge per-session Chroma namespaces into a single fixed namespace.

Why
---
With RAG_NAMESPACE_MODE=session, every browser session got its own Chroma
collection, and the session id lives in a cookie signed with a SECRET_KEY that
regenerates on every boot when unset. Each restart minted a fresh namespace.

Measured on a real store before this ran: 331 collections holding 280 entries,
229 of them alone in their own namespace and 90 empty. Because _retrieve_hits
excludes the current entry by id, an entry alone in a namespace retrieves from a
pool of zero. The journal history was being written and never read.

This consolidates the scattered entries into one namespace so history actually
accumulates. Pair it with RAG_NAMESPACE_MODE=fixed (now the default).

Safety model (identical to scripts/migrate_embeddings.py)
---------------------------------------------------------
  * DRY RUN BY DEFAULT. Without --apply nothing on disk changes.
  * THE SOURCE STORE IS NEVER WRITTEN TO. A new directory is always built.
  * VERIFY BEFORE SWAP. Every source id must be present in the target with
    byte-identical text; any id collision or loss aborts before the swap.
  * NOTHING IS DELETED. --apply renames the original aside with a timestamp.

Usage
-----
    python scripts/consolidate_namespaces.py                 # preview
    python scripts/consolidate_namespaces.py --apply         # merge for real
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("consolidate")

BATCH = 64
_COLLECTION_PREFIX = "ns__"


def _client(path: str):
    import chromadb
    from chromadb.config import Settings

    return chromadb.PersistentClient(
        path=path, settings=Settings(anonymized_telemetry=False)
    )


def _collection_name(namespace: str) -> str:
    """Must match vector_store.chroma_store._collection_name exactly."""
    import re

    safe = re.sub(r"[^a-zA-Z0-9._-]", "_", namespace or "default")
    return f"{_COLLECTION_PREFIX}{safe}"[:63]


def _build_ef(cfg):
    if (cfg.embedding_backend or "default").lower() != "ollama":
        return None
    from chromadb.utils import embedding_functions as chroma_ef

    return chroma_ef.OllamaEmbeddingFunction(
        url=cfg.ollama_embed_url, model_name=cfg.ollama_embed_model
    )


def _read_all(source_dir: str, ef) -> Tuple[Dict[str, Dict[str, Any]], List[str], int]:
    """Return (entries_by_id, source_collection_names, empty_count)."""
    src = _client(source_dir)
    names = [c.name for c in src.list_collections()]
    entries: Dict[str, Dict[str, Any]] = {}
    collisions: List[str] = []
    empty = 0

    for name in names:
        kwargs: Dict[str, Any] = {"name": name}
        if ef is not None:
            kwargs["embedding_function"] = ef
        col = src.get_collection(**kwargs)
        got = col.get(include=["documents", "metadatas"])
        ids = got.get("ids") or []
        if not ids:
            empty += 1
            continue
        docs = got.get("documents") or []
        metas = got.get("metadatas") or []
        for i, entry_id in enumerate(ids):
            if entry_id in entries:
                collisions.append(entry_id)
                continue
            entries[entry_id] = {
                "text": docs[i] if i < len(docs) else "",
                "metadata": dict(metas[i] or {}) if i < len(metas) else {},
                "source_collection": name,
            }

    if collisions:
        raise RuntimeError(
            f"id collisions across namespaces: {collisions[:5]} "
            f"({len(collisions)} total). Consolidation would lose entries; aborting."
        )
    return entries, names, empty


def main() -> int:
    # load_dotenv BEFORE load_config, or .env is ignored and every setting
    # silently falls back to its built-in default. Caught the hard way: without
    # this the script read EMBEDDING_BACKEND as "default", re-embedded the merged
    # collection with Chroma's 384-dim MiniLM, and would have silently reverted
    # a completed nomic migration. Reads still worked, because Chroma falls back
    # to the embedding function recorded on each collection, so nothing looked
    # wrong until the write.
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")

    from config import load_config

    cfg = load_config()
    default_source = os.getenv("CHROMA_PERSIST_DIR", "./storage/chroma")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=default_source)
    parser.add_argument(
        "--target-namespace",
        default=cfg.rag_namespace_fixed,
        help="namespace all entries are merged into",
    )
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    source = Path(args.source)
    target_collection = _collection_name(args.target_namespace)

    log.info("=== Chroma namespace consolidation ===")
    log.info("  source          : %s", source)
    log.info("  target namespace: %s  (collection %s)", args.target_namespace, target_collection)
    log.info("  embedder        : %s", cfg.embedding_backend)
    log.info("  mode            : %s", "APPLY" if args.apply else "DRY RUN (nothing will change)")
    log.info("")

    if not source.exists():
        log.info("No store at %s. Nothing to consolidate.", source)
        return 0

    ef = _build_ef(cfg)
    try:
        entries, names, empty = _read_all(str(source), ef)
    except Exception as exc:  # noqa: BLE001 - surfaced verbatim, nothing changed
        log.error("Aborted before any change: %s: %s", type(exc).__name__, exc)
        return 1

    if not entries:
        log.info("No entries found. Nothing to consolidate.")
        return 0

    log.info(
        "Found %d entries across %d collections (%d empty).",
        len(entries), len(names), empty,
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    scratch = source.parent / f"{source.name}.consolidating_{stamp}"

    try:
        dst = _client(str(scratch))
        kwargs: Dict[str, Any] = {
            "name": target_collection,
            "metadata": {"description": f"journal_entries (ns={args.target_namespace})"},
        }
        if ef is not None:
            kwargs["embedding_function"] = ef
        collection = dst.get_or_create_collection(**kwargs)

        ids = list(entries.keys())
        for start in range(0, len(ids), BATCH):
            chunk = ids[start : start + BATCH]
            collection.add(
                ids=chunk,
                documents=[entries[i]["text"] for i in chunk],
                metadatas=[
                    {
                        **(entries[i]["metadata"] or {}),
                        "namespace": args.target_namespace,
                        # Kept so the pre-consolidation grouping stays auditable.
                        "origin_collection": entries[i]["source_collection"],
                    }
                    for i in chunk
                ],
            )
        log.info("  merged %d entries into %s", len(ids), target_collection)

        # ── Verify ───────────────────────────────────────────────────────────
        log.info("")
        log.info("Verifying...")
        check_kwargs: Dict[str, Any] = {"name": target_collection}
        if ef is not None:
            check_kwargs["embedding_function"] = ef
        got = dst.get_collection(**check_kwargs).get(include=["documents", "embeddings"])
        got_ids = set(got.get("ids") or [])
        got_docs = dict(zip(got.get("ids") or [], got.get("documents") or []))

        problems: List[str] = []

        # Embedding WIDTH is checked, not just ids and text. Comparing only text
        # made this verification blind to the failure that actually occurred:
        # the merged collection being rebuilt under the wrong embedder. Text
        # survives that perfectly, so a text-only check reports success on a
        # store whose vectors have all silently changed model.
        from vector_store.embeddings import expected_dimension

        want_dim = expected_dimension(
            (cfg.embedding_backend or "default").lower(), cfg.ollama_embed_model
        )
        embeddings = got.get("embeddings")
        actual_dim = len(embeddings[0]) if embeddings is not None and len(embeddings) else None
        log.info(
            "  embedding width: %s (expected %s)",
            actual_dim,
            want_dim if want_dim is not None else "unknown model, not checked",
        )
        if want_dim is not None and actual_dim is not None and actual_dim != want_dim:
            problems.append(
                f"embedding width {actual_dim} != expected {want_dim} for "
                f"backend={cfg.embedding_backend} model={cfg.ollama_embed_model}. "
                "The merged store would be under the wrong embedder."
            )
        missing = set(entries) - got_ids
        if missing:
            problems.append(f"{len(missing)} entries missing, e.g. {sorted(missing)[:3]}")
        for entry_id, original in entries.items():
            if entry_id in got_docs and got_docs[entry_id] != original["text"]:
                problems.append(f"{entry_id}: text changed")

        if problems:
            log.error("VERIFICATION FAILED. Original store untouched.")
            for p in problems[:10]:
                log.error("  - %s", p)
            shutil.rmtree(scratch, ignore_errors=True)
            return 1

        log.info("Verification passed: %d/%d entries, text identical.", len(got_ids), len(entries))
    except Exception as exc:  # noqa: BLE001
        log.error("Failed before any change: %s: %s", type(exc).__name__, exc)
        shutil.rmtree(scratch, ignore_errors=True)
        return 1

    if not args.apply:
        shutil.rmtree(scratch, ignore_errors=True)
        log.info("")
        log.info("DRY RUN complete. Nothing changed on disk.")
        log.info("Re-run with --apply to consolidate for real.")
        return 0

    retired = source.parent / f"{source.name}.pre_consolidate_{stamp}"
    source.rename(retired)
    scratch.rename(source)

    log.info("")
    log.info("Consolidation applied.")
    log.info("  live store: %s  (%d collections -> 1)", source, len(names))
    log.info("  previous  : %s  (kept, not deleted)", retired)
    log.info("")
    log.info("To revert:")
    log.info("    rm -rf %s && mv %s %s", source, retired, source)
    log.info("  then set RAG_NAMESPACE_MODE=session")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
