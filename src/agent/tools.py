"""Agent tools: retrieve, metadata query, gated write.

These are plain callables used by the LangGraph nodes. They are intentionally
framework-light so unit tests can exercise them without spinning a graph.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from ..vector_store.base import VectorStore, format_hits_as_context
from ..vector_store.pgvector_store import PgVectorStore

from ..config import load_config
from ..privacy.redact import redact
from .confirmation import build_pending_write

logger = logging.getLogger(__name__)



def tool_retrieve(
    store: VectorStore,
    query: str,
    *,
    namespace: str,
    top_k: int = 3,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    hits = store.query(query, top_k=top_k, namespace=namespace)
    latency_ms = (time.perf_counter() - t0) * 1000
    return {
        "tool": "retrieve_journal",
        "query": query,
        "hits": [h.to_dict() for h in hits],
        "context": format_hits_as_context(hits),
        "latency_ms": round(latency_ms, 2),
    }


def tool_query_metadata(
    store: VectorStore,
    *,
    namespace: str,
    entry_id: Optional[str] = None,
    limit: int = 20,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    if hasattr(store, "query_entry_metadata"):
        rows = store.query_entry_metadata(
            namespace=namespace, entry_id=entry_id, limit=limit
        )
    elif isinstance(store, PgVectorStore):
        rows = store.query_entry_metadata(
            namespace=namespace, entry_id=entry_id, limit=limit
        )
    else:
        rows = []
        if entry_id:
            hits = store.query(entry_id, top_k=1, namespace=namespace)
            rows = [
                {
                    "entry_id": h.id,
                    "metadata": h.metadata,
                    "created_at": h.metadata.get("created_at"),
                    "preview": h.text[:200],
                }
                for h in hits
            ]
    latency_ms = (time.perf_counter() - t0) * 1000
    return {
        "tool": "query_entry_metadata",
        "entry_id": entry_id,
        "rows": rows,
        "latency_ms": round(latency_ms, 2),
    }


def tool_propose_write(
    *,
    text: str,
    namespace: str,
    entry_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Park a write for confirmation — does NOT mutate the store."""
    eid = entry_id or f"agent-{uuid.uuid4().hex[:12]}"
    pending = build_pending_write(
        entry_id=eid, text=text, metadata=metadata, namespace=namespace
    )
    return {
        "tool": "propose_write",
        "status": "awaiting_confirmation",
        "pending_write": pending,
        "message": (
            f"Proposed write to entry {eid!r}. "
            "Reply 'confirm' to apply or 'cancel' to discard."
        ),
    }


def tool_apply_write(store: VectorStore, pending: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.perf_counter()
    pending = pending or {}
    entry_id = pending.get("entry_id")
    text = pending.get("text")
    if not isinstance(entry_id, str) or not entry_id.strip() or not isinstance(text, str):
        return {
            "tool": "apply_write",
            "status": "failed",
            "reason": "invalid_pending_write",
            "entry_id": entry_id if isinstance(entry_id, str) else None,
            "latency_ms": 0.0,
        }
    enabled = bool(getattr(store, "enabled", True))
    if not enabled:
        return {
            "tool": "apply_write",
            "status": "failed",
            "reason": "retrieval_disabled",
            "entry_id": entry_id,
            "latency_ms": 0.0,
        }

    # Apply redaction when PRIVACY_MODE=strict
    cfg = load_config()
    if cfg.privacy_mode == "strict":
        text = redact(text)
        if not text or not text.strip():
            return {
                "tool": "apply_write",
                "status": "failed",
                "reason": "redacted_empty",
                "entry_id": pending.get("entry_id"),
                "latency_ms": (time.perf_counter() - t0) * 1000,
            }
    
    ok = store.add_entry(
        entry_id,
        text,
        pending.get("metadata") or {},
        namespace=pending.get("namespace"),
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    return {
        "tool": "apply_write",
        "status": "applied" if ok else "failed",
        "entry_id": pending.get("entry_id"),
        "latency_ms": round(latency_ms, 2),
    }
