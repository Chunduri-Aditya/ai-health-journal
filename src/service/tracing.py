# GateGuard: callers service/main.py, agent tests. User: Implement the plan as specified. Do NOT edit the plan file.
"""Langfuse / OpenTelemetry tracing hooks for agent runs.

Enabled when LANGFUSE_ENABLED=true and LANGFUSE_PUBLIC_KEY / SECRET_KEY are set.
When disabled, helpers are no-ops so local runs stay offline.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

logger = logging.getLogger(__name__)

_SENSITIVE_KEYS = frozenset({"text", "context", "pending_write"})

_LANGFUSE_CLIENT: Any = None
_LANGFUSE_CLIENT_CHECKED = False


class TraceCollector:
    """In-process span collector used when Langfuse is off (and as a fallback)."""

    def __init__(self) -> None:
        self.spans: list[Dict[str, Any]] = []

    def add(self, name: str, **fields: Any) -> None:
        self.spans.append({"name": name, **fields})


def _trace_include_text() -> bool:
    if os.getenv("TRACE_INCLUDE_TEXT", "false").lower() == "true":
        return True
    try:
        from ..config import load_config

        return load_config().trace_include_text
    except Exception:  # noqa: BLE001
        return False


def scrub_trace_payload(data: Any, *, include_text: bool = False) -> Any:
    """Remove sensitive journal text from trace metadata unless explicitly allowed."""
    if include_text:
        return data
    if isinstance(data, dict):
        scrubbed: Dict[str, Any] = {}
        for key, value in data.items():
            if key in _SENSITIVE_KEYS:
                scrubbed[key] = "[scrubbed]"
            elif key == "hits" and isinstance(value, list):
                scrubbed[key] = [
                    scrub_trace_payload(item, include_text=include_text)
                    for item in value
                ]
            else:
                scrubbed[key] = scrub_trace_payload(value, include_text=include_text)
        return scrubbed
    if isinstance(data, list):
        return [
            scrub_trace_payload(item, include_text=include_text) for item in data
        ]
    return data


def _get_langfuse_client() -> Any:
    """Return a module-level Langfuse client singleton (lazy, created once)."""
    global _LANGFUSE_CLIENT, _LANGFUSE_CLIENT_CHECKED

    if _LANGFUSE_CLIENT_CHECKED:
        return _LANGFUSE_CLIENT

    _LANGFUSE_CLIENT_CHECKED = True
    if os.getenv("LANGFUSE_ENABLED", "false").lower() != "true":
        return None

    public = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    secret = os.getenv("LANGFUSE_SECRET_KEY", "")
    if not public or not secret:
        logger.warning("LANGFUSE_ENABLED but keys missing; tracing disabled.")
        return None

    try:
        from langfuse import Langfuse
    except ImportError:
        logger.warning("langfuse package not installed; tracing disabled.")
        return None

    _LANGFUSE_CLIENT = Langfuse(
        public_key=public,
        secret_key=secret,
        host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
    )
    return _LANGFUSE_CLIENT


def _start_root_observation(
    client: Any,
    *,
    name: str,
    metadata: Optional[Dict[str, Any]],
):
    """Start a root trace/span using Langfuse v4-style APIs when available."""
    meta = metadata or {}
    if hasattr(client, "start_as_current_observation"):
        try:
            return client.start_as_current_observation(
                as_type="span",
                name=name,
                metadata=meta,
            )
        except TypeError:
            try:
                return client.start_as_current_observation(name=name, metadata=meta)
            except Exception as e:  # noqa: BLE001
                logger.debug("start_as_current_observation failed: %s", e)
        except Exception as e:  # noqa: BLE001
            logger.debug("start_as_current_observation failed: %s", e)

    if hasattr(client, "start_span"):
        try:
            return client.start_span(name=name, metadata=meta)
        except Exception as e:  # noqa: BLE001
            logger.debug("start_span failed: %s", e)

    try:
        return client.trace(name=name, metadata=meta)
    except Exception as e:  # noqa: BLE001
        logger.warning("Langfuse trace start failed: %s", e)
        return None


def _record_child_span(root: Any, *, name: str, metadata: Dict[str, Any]) -> None:
    if root is None:
        return

    if hasattr(root, "start_span"):
        try:
            with root.start_span(name=name, metadata=metadata):
                return
        except Exception as e:  # noqa: BLE001
            logger.debug("root.start_span failed: %s", e)

    if hasattr(root, "span"):
        try:
            root.span(name=name, metadata=metadata)
            return
        except Exception as e:  # noqa: BLE001
            logger.debug("root.span failed: %s", e)


def _finalize_root_observation(
    client: Any,
    root: Any,
    *,
    spans: list[Dict[str, Any]],
) -> None:
    if root is None:
        return

    scrubbed = scrub_trace_payload(spans, include_text=_trace_include_text())

    if hasattr(root, "update"):
        try:
            root.update(output={"spans": scrubbed})
        except Exception as e:  # noqa: BLE001
            logger.debug("root.update failed: %s", e)

    if hasattr(root, "end"):
        try:
            root.end()
        except Exception as e:  # noqa: BLE001
            logger.debug("root.end failed: %s", e)

    if hasattr(root, "__exit__"):
        try:
            root.__exit__(None, None, None)
        except Exception as e:  # noqa: BLE001
            logger.debug("root context exit failed: %s", e)

    if client is not None and hasattr(client, "flush"):
        try:
            client.flush()
        except Exception as e:  # noqa: BLE001
            logger.warning("Langfuse flush failed: %s", e)


@contextmanager
def trace_agent_run(
    name: str = "journal_agent",
    *,
    metadata: Optional[Dict[str, Any]] = None,
) -> Iterator[TraceCollector]:
    collector = TraceCollector()
    client = _get_langfuse_client()
    t0 = time.perf_counter()
    root = _start_root_observation(client, name=name, metadata=metadata) if client else None
    try:
        yield collector
    finally:
        total_ms = (time.perf_counter() - t0) * 1000
        collector.add("agent_run_total", latency_ms=round(total_ms, 2))
        include_text = _trace_include_text()
        for span in collector.spans:
            span_name = span.get("name", "span")
            span_meta = scrub_trace_payload(
                {k: v for k, v in span.items() if k != "name"},
                include_text=include_text,
            )
            _record_child_span(root, name=span_name, metadata=span_meta)
        _finalize_root_observation(client, root, spans=collector.spans)
