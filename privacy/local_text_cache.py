"""
Local text cache for privacy-aware Pinecone usage.

When PINECONE_STORE_TEXT=false, we still want interpretable grounding
locally. This module keeps a small JSONL cache mapping (namespace, id)
to redacted text + basic metadata.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

from config import load_config

_CFG = load_config()

_CACHE_PATH = _CFG.local_cache_path
_MAX_ITEMS = _CFG.local_cache_max_items
_TTL_DAYS = _CFG.local_cache_ttl_days
_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
_loaded = False
_skipped_lines = 0
_last_prune: Optional[str] = None


def _ensure_loaded() -> None:
    global _loaded, _skipped_lines
    if _loaded:
        return
    if not os.path.exists(_CACHE_PATH):
        _loaded = True
        return
    try:
        with open(_CACHE_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    _skipped_lines += 1
                    continue
                ns = rec.get("namespace")
                doc_id = rec.get("id")
                if ns and doc_id:
                    _cache[(ns, doc_id)] = rec
    except Exception:
        # Best-effort; cache can rebuild over time
        _cache.clear()
        _skipped_lines = 0
    _loaded = True


def _prune() -> None:
    """Prune cache based on TTL and max items, and rewrite file."""
    global _last_prune
    _ensure_loaded()
    if not _cache:
        return

    now = datetime.utcnow()
    ttl_cutoff = now - timedelta(days=_TTL_DAYS)

    # Drop old entries
    items = list(_cache.items())
    kept: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for key, rec in items:
        created_at = rec.get("created_at")
        try:
            created_dt = datetime.fromisoformat(created_at) if created_at else now
        except Exception:
            created_dt = now
        if created_dt >= ttl_cutoff:
            kept[key] = rec

    # Enforce max items (drop oldest)
    if len(kept) > _MAX_ITEMS:
        sorted_items = sorted(
            kept.items(),
            key=lambda kv: kv[1].get("created_at") or "",
        )
        # Keep the most recent _MAX_ITEMS
        for key, _rec in sorted_items[:-_MAX_ITEMS]:
            kept.pop(key, None)

    _cache.clear()
    _cache.update(kept)
    _last_prune = now.isoformat()

    # Rewrite cache file with cleaned content
    os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
    with open(_CACHE_PATH, "w", encoding="utf-8") as f:
        for _key, rec in _cache.items():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def put(namespace: str, doc_id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    """Persist redacted text locally for a given (namespace, id)."""
    _ensure_loaded()
    rec = {
        "namespace": namespace,
        "id": doc_id,
        "text": text,
        "metadata": metadata or {},
        "created_at": datetime.utcnow().isoformat(),
    }
    _cache[(namespace, doc_id)] = rec

    os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
    with open(_CACHE_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    _prune()


def get(namespace: str, doc_id: str) -> Optional[Dict[str, Any]]:
    """Lookup a locally cached record for (namespace, id)."""
    _ensure_loaded()
    return _cache.get((namespace, doc_id))


def health() -> Dict[str, Any]:
    """Return simple cache health diagnostics."""
    _ensure_loaded()
    return {
        "items": len(_cache),
        "skipped_lines": _skipped_lines,
        "last_prune": _last_prune,
        "path": _CACHE_PATH,
        "max_items": _MAX_ITEMS,
        "ttl_days": _TTL_DAYS,
    }


