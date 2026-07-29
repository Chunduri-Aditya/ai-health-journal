#!/usr/bin/env python3
"""Re-embed an existing Chroma store under a different embedding backend.

Why a migration is needed at all
--------------------------------
Embedding backends produce different vector widths (Chroma's bundled MiniLM is
384-dim, nomic-embed-text is 768-dim) and Chroma records the embedding function
on each collection. Switching backends against an existing store therefore
raises rather than corrupting it, and the stored journal entries have to be
re-embedded under the new model before they are retrievable again.

Safety model
------------
This script touches real journal data, so it is built to be boring:

  * DRY RUN BY DEFAULT. Without --apply it reads, re-embeds into a scratch
    directory, verifies, reports, and deletes the scratch copy.
  * THE SOURCE STORE IS NEVER WRITTEN TO. Migration always builds a brand new
    directory; the original is only ever read.
  * VERIFY BEFORE SWAP. Collection names, entry ids, document text and metadata
    are compared exactly between source and migrated copy. Any mismatch aborts
    before anything is swapped.
  * NOTHING IS DELETED. --apply renames the original aside with a timestamp and
    moves the migrated copy into place. Reversing it is a rename.

Usage
-----
    python scripts/migrate_embeddings.py                     # preview only
    python scripts/migrate_embeddings.py --apply             # migrate for real
    python scripts/migrate_embeddings.py --to-backend default --apply   # revert
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("migrate")

# Chroma reads documents back in pages; keep batches modest so a large journal
# does not build one enormous embedding request.
BATCH = 64


def _client(path: str):
    import chromadb
    from chromadb.config import Settings

    return chromadb.PersistentClient(
        path=path, settings=Settings(anonymized_telemetry=False)
    )


def _read_collection(collection) -> Dict[str, Dict[str, Any]]:
    """Return {id: {"text":..., "metadata":...}} for every entry."""
    got = collection.get(include=["documents", "metadatas"])
    ids = got.get("ids") or []
    docs = got.get("documents") or []
    metas = got.get("metadatas") or []
    out: Dict[str, Dict[str, Any]] = {}
    for i, entry_id in enumerate(ids):
        out[entry_id] = {
            "text": docs[i] if i < len(docs) else "",
            "metadata": dict(metas[i] or {}) if i < len(metas) else {},
        }
    return out


def _build_target_ef(backend: str, model: str, url: str):
    if backend == "default":
        return None
    from chromadb.utils import embedding_functions as chroma_ef

    return chroma_ef.OllamaEmbeddingFunction(url=url, model_name=model)


def migrate(
    source_dir: str,
    target_dir: str,
    *,
    backend: str,
    model: str,
    url: str,
) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """Copy every collection from source_dir into target_dir, re-embedding.

    Returns (source_snapshot_by_collection, collection_names).
    """
    src = _client(source_dir)
    names = [c.name for c in src.list_collections()]
    if not names:
        return {}, []

    snapshot: Dict[str, Dict[str, Any]] = {}
    for name in names:
        snapshot[name] = _read_collection(src.get_collection(name))

    ef = _build_target_ef(backend, model, url)
    dst = _client(target_dir)

    for name in names:
        entries = snapshot[name]
        kwargs: Dict[str, Any] = {"name": name}
        if ef is not None:
            kwargs["embedding_function"] = ef
        collection = dst.get_or_create_collection(**kwargs)

        ids = list(entries.keys())
        for start in range(0, len(ids), BATCH):
            chunk = ids[start : start + BATCH]
            collection.add(
                ids=chunk,
                documents=[entries[i]["text"] for i in chunk],
                metadatas=[entries[i]["metadata"] or {"_": ""} for i in chunk],
            )
        log.info("  re-embedded %-40s %d entries", name, len(ids))

    return snapshot, names


def verify(
    snapshot: Dict[str, Dict[str, Any]],
    target_dir: str,
    *,
    backend: str,
    model: str,
    url: str,
) -> List[str]:
    """Compare the migrated store against the source snapshot. Returns problems."""
    problems: List[str] = []
    ef = _build_target_ef(backend, model, url)
    dst = _client(target_dir)

    migrated_names = {c.name for c in dst.list_collections()}
    missing = set(snapshot) - migrated_names
    if missing:
        problems.append(f"collections missing after migration: {sorted(missing)}")

    for name, entries in snapshot.items():
        if name not in migrated_names:
            continue
        kwargs: Dict[str, Any] = {"name": name}
        if ef is not None:
            kwargs["embedding_function"] = ef
        got = _read_collection(dst.get_collection(**kwargs))

        if set(got) != set(entries):
            lost = sorted(set(entries) - set(got))
            gained = sorted(set(got) - set(entries))
            problems.append(f"{name}: id mismatch (lost={lost[:5]} gained={gained[:5]})")
            continue

        for entry_id, original in entries.items():
            if got[entry_id]["text"] != original["text"]:
                problems.append(f"{name}/{entry_id}: document text changed")
            if got[entry_id]["metadata"] != (original["metadata"] or {"_": ""}):
                # Metadata is compared after the same empty-dict normalisation
                # applied on write, since Chroma rejects a literally empty dict.
                if original["metadata"]:
                    problems.append(f"{name}/{entry_id}: metadata changed")
    return problems


def main() -> int:
    # load_dotenv BEFORE load_config, or .env is ignored and every setting
    # silently falls back to its built-in default. See the same note in
    # scripts/consolidate_namespaces.py for what that cost.
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")

    from config import load_config

    cfg = load_config()
    default_source = os.getenv("CHROMA_PERSIST_DIR", "./storage/chroma")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=default_source)
    parser.add_argument(
        "--to-backend",
        default=None,
        choices=["default", "ollama"],
        help="target backend (defaults to the configured EMBEDDING_BACKEND)",
    )
    parser.add_argument("--model", default=cfg.ollama_embed_model)
    parser.add_argument("--url", default=cfg.ollama_embed_url)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="actually swap the store; without this nothing on disk changes",
    )
    args = parser.parse_args()

    backend = args.to_backend or cfg.embedding_backend
    source = Path(args.source)

    log.info("=== Chroma embedding migration ===")
    log.info("  source      : %s", source)
    log.info("  target model: %s", "chroma default MiniLM" if backend == "default" else args.model)
    log.info("  mode        : %s", "APPLY" if args.apply else "DRY RUN (nothing will change)")
    log.info("")

    if not source.exists():
        log.info("No store at %s. Nothing to migrate.", source)
        return 0

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    scratch = source.parent / f"{source.name}.migrating_{stamp}"

    try:
        snapshot, names = migrate(
            str(source), str(scratch), backend=backend, model=args.model, url=args.url
        )
    except Exception as exc:  # noqa: BLE001 - surfaced verbatim, nothing swapped
        log.error("Migration failed before any change was made: %s: %s", type(exc).__name__, exc)
        shutil.rmtree(scratch, ignore_errors=True)
        return 1

    if not names:
        log.info("Store has no collections. Nothing to migrate.")
        shutil.rmtree(scratch, ignore_errors=True)
        return 0

    total = sum(len(v) for v in snapshot.values())
    log.info("")
    log.info("Verifying %d entries across %d collection(s)...", total, len(names))
    problems = verify(snapshot, str(scratch), backend=backend, model=args.model, url=args.url)

    if problems:
        log.error("VERIFICATION FAILED. Original store untouched.")
        for p in problems[:20]:
            log.error("  - %s", p)
        shutil.rmtree(scratch, ignore_errors=True)
        return 1

    log.info("Verification passed: ids, document text and metadata all match.")

    if not args.apply:
        shutil.rmtree(scratch, ignore_errors=True)
        log.info("")
        log.info("DRY RUN complete. Nothing changed on disk.")
        log.info("Re-run with --apply to migrate for real.")
        return 0

    retired = source.parent / f"{source.name}.pre_{backend}_{stamp}"
    source.rename(retired)
    scratch.rename(source)

    log.info("")
    log.info("Migration applied.")
    log.info("  live store : %s", source)
    log.info("  previous   : %s  (kept, not deleted)", retired)
    log.info("")
    log.info("To revert:")
    log.info("    rm -rf %s && mv %s %s", source, retired, source)
    log.info("  then set EMBEDDING_BACKEND back to its previous value.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
