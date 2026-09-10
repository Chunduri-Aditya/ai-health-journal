"""FastAPI entrypoint for the deployed Journal Agent service.

Endpoints:
  GET  /healthz          — liveness (no dependency round-trips)
  GET  /readyz           — readiness (store + configured LLM)
  POST /v1/agent/invoke  — one agent turn (JSON)
  POST /v1/agent/stream  — SSE stream of tool/trace events + final answer
  POST /v1/ingest        — chunk+embed+upsert a journal entry

GateGuard facts: callers start.sh, tests/test_service_*.py, Dockerfile.service.
Existing file. Request schemas: IngestRequest(entry_id,text,metadata),
AgentInvokeRequest(message,session_id). User: Implement the plan as specified,
it is attached for your reference. Do NOT edit the plan file itself.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import threading
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

load_dotenv()

from ..config import load_config
from ..safety import redact
from ..providers.factory import get_llm_provider
from .tracing import trace_agent_run
from ..vector_store.factory import get_vector_store

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("journal_agent")

_MAX_TEXT_CHARS = 20_000

_store = None
_provider = None
_sessions: Dict[str, Dict[str, Any]] = {}
_sessions_lock = threading.Lock()
_SESSION_TTL_SEC = int(os.getenv("AGENT_SESSION_TTL_SEC", "3600"))
_SESSION_MAX = int(os.getenv("AGENT_SESSION_MAX", "500"))


def _runtime_env() -> str:
    return (os.getenv("ENV") or "dev").strip().lower()


def _get_store():
    global _store
    if _store is None:
        _store = get_vector_store()
    return _store


def _get_provider():
    global _provider
    if _provider is None:
        _provider = get_llm_provider(load_config(), strict=True)
    return _provider


def _fixed_namespace(req_namespace: Optional[str]) -> str:
    """Single-tenant: client namespace overrides are rejected."""
    cfg = load_config()
    fixed = cfg.rag_namespace_fixed
    if req_namespace and req_namespace != fixed:
        raise HTTPException(
            status_code=400,
            detail=(
                "namespace override not allowed in single-tenant mode; "
                f"server namespace is {fixed!r}"
            ),
        )
    return fixed


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build store + provider once; fail fast on backend misconfiguration."""
    global _store, _provider
    cfg = load_config()
    _store = get_vector_store()
    _provider = get_llm_provider(cfg, strict=True)
    logger.info(
        "Journal Agent ready (store=%s enabled=%s llm=%s env=%s)",
        getattr(_store, "backend_name", "?"),
        getattr(_store, "enabled", False),
        cfg.llm_backend,
        _runtime_env(),
    )
    yield


app = FastAPI(
    title="Journal Agent",
    description="Journal Agent — Agentic RAG service (pgvector + LangGraph).",
    version="0.1.0",
    lifespan=lifespan,
)


def _require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    """Shared-secret gate for the single-owner service.

    Outside ENV=dev/test the key is required (fail-closed). In local
    ENV=dev/test the gate opens when JOURNAL_AGENT_API_KEY is unset.
    """
    expected = os.getenv("JOURNAL_AGENT_API_KEY", "")
    env = _runtime_env()
    if not expected:
        if env not in ("dev", "test"):
            raise HTTPException(
                status_code=503,
                detail="JOURNAL_AGENT_API_KEY is required when ENV is not dev/test",
            )
        return
    if not x_api_key or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


def _durable_sessions_available(store) -> bool:
    return bool(
        store
        and getattr(store, "enabled", False)
        and hasattr(store, "load_session")
        and hasattr(store, "save_session")
    )


def _load_session(session_id: str) -> Dict[str, Any]:
    store = _store
    if _durable_sessions_available(store):
        try:
            store.purge_stale_sessions(_SESSION_TTL_SEC)  # type: ignore[attr-defined]
        except Exception:
            pass
        payload = store.load_session(session_id)  # type: ignore[attr-defined]
        # Drop expired pending writes (HITL TTL).
        pending = payload.get("pending_write") if isinstance(payload, dict) else None
        if isinstance(pending, dict) and pending.get("proposed_at"):
            age = time.time() - float(pending.get("proposed_at") or 0)
            if age > min(_SESSION_TTL_SEC, 600):
                payload = dict(payload)
                payload["pending_write"] = None
                payload["awaiting_confirmation"] = False
        return dict(payload or {})

    now = time.time()
    with _sessions_lock:
        stale = [
            sid for sid, payload in _sessions.items()
            if now - float(payload.get("_ts", 0)) > _SESSION_TTL_SEC
        ]
        for sid in stale:
            _sessions.pop(sid, None)
        return dict(_sessions.get(session_id) or {})


def _save_session(session_id: str, payload: Dict[str, Any]) -> None:
    store = _store
    data = dict(payload)
    data.pop("_ts", None)
    if _durable_sessions_available(store):
        store.save_session(session_id, data)  # type: ignore[attr-defined]
        return

    with _sessions_lock:
        if len(_sessions) >= _SESSION_MAX and session_id not in _sessions:
            oldest = sorted(_sessions.items(), key=lambda kv: float(kv[1].get("_ts", 0)))
            for sid, _ in oldest[: max(1, len(_sessions) - _SESSION_MAX + 1)]:
                _sessions.pop(sid, None)
        data["_ts"] = time.time()
        _sessions[session_id] = data


class IngestRequest(BaseModel):
    entry_id: str = Field(..., min_length=1, max_length=256)
    text: str = Field(..., min_length=1, max_length=_MAX_TEXT_CHARS)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    namespace: Optional[str] = Field(default=None, max_length=256)


class AgentInvokeRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=_MAX_TEXT_CHARS)
    session_id: str = Field(..., min_length=1, max_length=256)
    namespace: Optional[str] = Field(default=None, max_length=256)


class AgentInvokeResponse(BaseModel):
    response: str
    session_id: str
    last_tool: Optional[str] = None
    awaiting_confirmation: bool = False
    tool_trace: List[Dict[str, Any]] = Field(default_factory=list)
    provider: str = ""


_ROOT_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Journal Agent</title>
  <link rel="icon" href="/favicon.ico"/>
  <style>
    :root { color-scheme: light dark; --fg: #1a1a1a; --muted: #5c5c5c; --line: #d0d0d0; --bg: #f7f6f3; --card: #fff; --accent: #0b5fff; }
    @media (prefers-color-scheme: dark) {
      :root { --fg: #f2f2f2; --muted: #a8a8a8; --line: #333; --bg: #121212; --card: #1c1c1c; --accent: #6ea8fe; }
    }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif;
           background: var(--bg); color: var(--fg); line-height: 1.45; }
    main { max-width: 40rem; margin: 2.5rem auto; padding: 0 1.25rem; }
    h1 { font-size: 1.6rem; margin: 0 0 .35rem; letter-spacing: -0.02em; }
    p { color: var(--muted); margin: 0 0 1.25rem; }
    .links { display: flex; flex-wrap: wrap; gap: .6rem; margin-bottom: 1.5rem; }
    a.chip { text-decoration: none; color: var(--fg); border: 1px solid var(--line);
             border-radius: .5rem; padding: .4rem .7rem; background: var(--card); font-size: .9rem; }
    a.chip:hover { border-color: var(--accent); color: var(--accent); }
    .panel { background: var(--card); border: 1px solid var(--line); border-radius: .75rem; padding: 1rem; }
    label { display: block; font-size: .85rem; color: var(--muted); margin-bottom: .35rem; }
    input, textarea { width: 100%; font: inherit; padding: .55rem .65rem; border-radius: .45rem;
                      border: 1px solid var(--line); background: transparent; color: var(--fg); margin-bottom: .75rem; }
    textarea { min-height: 5.5rem; resize: vertical; }
    button { font: inherit; border: 0; border-radius: .45rem; padding: .55rem 1rem;
             background: var(--accent); color: #fff; cursor: pointer; }
    button:disabled { opacity: .6; cursor: wait; }
    pre { white-space: pre-wrap; word-break: break-word; font-size: .82rem;
          background: var(--bg); border-radius: .45rem; padding: .75rem; margin: .75rem 0 0;
          border: 1px solid var(--line); max-height: 22rem; overflow: auto; }
    .hint { font-size: .8rem; color: var(--muted); margin-top: 1rem; }
  </style>
</head>
<body>
<main>
  <h1>Journal Agent</h1>
  <p>FastAPI service is running. Use the form below or open Swagger docs.</p>
  <div class="links">
    <a class="chip" href="/docs">Swagger /docs</a>
    <a class="chip" href="/readyz">Ready /readyz</a>
    <a class="chip" href="/healthz">Health /healthz</a>
  </div>
  <div class="panel">
    <label for="apiKey">X-API-Key (optional in ENV=dev)</label>
    <input id="apiKey" type="password" autocomplete="off" placeholder="leave blank in local dev"/>
    <label for="sessionId">session_id</label>
    <input id="sessionId" value="browser-demo"/>
    <label for="message">message</label>
    <textarea id="message">I slept well after a walk.</textarea>
    <button id="send" type="button">POST /v1/agent/invoke</button>
    <pre id="out">Response will appear here.</pre>
  </div>
  <p class="hint">Full journaling UI: <code>./start.sh</code> (Flask on :5000). This page is for API smoke tests.</p>
</main>
<script>
const out = document.getElementById("out");
document.getElementById("send").addEventListener("click", async () => {
  const btn = document.getElementById("send");
  btn.disabled = true;
  out.textContent = "Calling…";
  const headers = {"Content-Type": "application/json"};
  const key = document.getElementById("apiKey").value.trim();
  if (key) headers["X-API-Key"] = key;
  try {
    const res = await fetch("/v1/agent/invoke", {
      method: "POST",
      headers,
      body: JSON.stringify({
        session_id: document.getElementById("sessionId").value.trim() || "browser-demo",
        message: document.getElementById("message").value,
      }),
    });
    const text = await res.text();
    let pretty = text;
    try { pretty = JSON.stringify(JSON.parse(text), null, 2); } catch (_) {}
    out.textContent = res.status + " " + res.statusText + "\\n\\n" + pretty;
  } catch (err) {
    out.textContent = String(err);
  } finally {
    btn.disabled = false;
  }
});
</script>
</body>
</html>
"""

# Tiny 16x16 SVG favicon (avoids empty-tab icon + repeated client confusion).
_FAVICON_SVG = (
    b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16">'
    b'<rect width="16" height="16" rx="3" fill="#0b5fff"/>'
    b'<path d="M4 4h8v2H4zm0 3h8v2H4zm0 3h5v2H4z" fill="#fff"/>'
    b"</svg>"
)


@app.get("/", response_class=HTMLResponse)
def root():
    """Browser landing with a smoke-test form (API still lives under /v1)."""
    return HTMLResponse(_ROOT_HTML)


@app.get("/favicon.ico")
def favicon():
    return Response(content=_FAVICON_SVG, media_type="image/svg+xml")


@app.get("/healthz")
def healthz():
    """Liveness only — no DB/LLM round-trips (Fly should probe /readyz)."""
    cfg = load_config()
    store = _store
    return {
        "status": "ok",
        "version": app.version,
        "retrieval": {
            "enabled": bool(store and getattr(store, "enabled", False)),
            "backend": getattr(store, "backend_name", "uninitialized") if store else "uninitialized",
            "healthy": None,
        },
        "llm_backend": cfg.llm_backend,
        "chunk_size": cfg.chunk_size,
        "chunk_overlap": cfg.chunk_overlap,
    }


@app.get("/readyz")
def readyz():
    store = _get_store()
    if store.enabled and not store.healthcheck():
        detail = getattr(store, "dimension_error", None) or "vector store unhealthy"
        raise HTTPException(status_code=503, detail=detail)
    cfg = load_config()
    provider = _get_provider()
    llm_ok = True
    try:
        llm_ok = bool(provider.healthcheck())
    except Exception:
        llm_ok = False
    env = _runtime_env()
    if not llm_ok and env not in ("dev", "test"):
        raise HTTPException(status_code=503, detail="llm backend unhealthy")
    return {
        "status": "ready",
        "llm": {
            "backend": cfg.llm_backend,
            "healthy": llm_ok,
        },
        "embedding_dimension_ok": getattr(store, "dimension_ok", True),
    }


@app.get("/ping")
def ping():
    """Compat alias for the Flask /ping consumers."""
    return healthz()


@app.post("/v1/ingest")
def ingest(req: IngestRequest, _: None = Depends(_require_api_key)):
    store = _get_store()
    if not store.enabled:
        raise HTTPException(status_code=400, detail="retrieval disabled")
    namespace = _fixed_namespace(req.namespace)
    # Apply redaction when PRIVACY_MODE=strict
    cfg = load_config()
    text = req.text
    if cfg.privacy_mode == 'strict':
        text = redact(text)
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail='text empty after redaction')
    t0 = time.perf_counter()
    ok = store.add_entry(
        req.entry_id,
        text,
        req.metadata,
        namespace=namespace,
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    if not ok:
        raise HTTPException(status_code=500, detail="ingest failed")
    return {"ok": True, "entry_id": req.entry_id, "latency_ms": round(latency_ms, 2)}


@app.post("/v1/agent/invoke", response_model=AgentInvokeResponse)
def agent_invoke(req: AgentInvokeRequest, _: None = Depends(_require_api_key)):
    cfg = load_config()
    from ..agent.graph import run_agent_turn

    prior = _load_session(req.session_id)
    namespace = _fixed_namespace(req.namespace)
    with trace_agent_run(
        "journal_agent.invoke",
        metadata={"session_id": req.session_id, "llm_backend": cfg.llm_backend},
    ) as trace:
        out = run_agent_turn(
            req.message,
            store=_get_store(),
            namespace=namespace,
            prior_state=prior,
        )
        for event in out.get("tool_trace") or []:
            trace.add(event.get("tool", "tool"), **{
                k: v for k, v in event.items() if k != "tool"
            })

    _save_session(req.session_id, {
        "messages": out.get("messages") or [],
        "pending_write": out.get("pending_write"),
        "awaiting_confirmation": bool(out.get("awaiting_confirmation")),
        "tool_trace": out.get("tool_trace") or [],
        "namespace": namespace,
    })
    return AgentInvokeResponse(
        response=out.get("final_response") or "",
        session_id=req.session_id,
        last_tool=out.get("last_tool"),
        awaiting_confirmation=bool(out.get("awaiting_confirmation")),
        tool_trace=out.get("tool_trace") or [],
        provider=cfg.llm_backend,
    )


@app.post("/v1/agent/stream")
async def agent_stream(req: AgentInvokeRequest, _: None = Depends(_require_api_key)):
    """SSE stream: tool events first, then a final response event."""
    cfg = load_config()
    from ..agent.graph import run_agent_turn

    prior = _load_session(req.session_id)
    namespace = _fixed_namespace(req.namespace)

    def event_gen():
        try:
            with trace_agent_run(
                "journal_agent.stream",
                metadata={"session_id": req.session_id},
            ) as trace:
                yield _sse("status", {"phase": "running", "provider": cfg.llm_backend})
                out = run_agent_turn(
                    req.message,
                    store=_get_store(),
                    namespace=namespace,
                    prior_state=prior,
                )
                for event in out.get("tool_trace") or []:
                    trace.add(event.get("tool", "tool"), **{
                        k: v for k, v in event.items() if k != "tool"
                    })
                    yield _sse("tool", event)
                _save_session(req.session_id, {
                    "messages": out.get("messages") or [],
                    "pending_write": out.get("pending_write"),
                    "awaiting_confirmation": bool(out.get("awaiting_confirmation")),
                    "tool_trace": out.get("tool_trace") or [],
                    "namespace": namespace,
                })
                yield _sse(
                    "final",
                    {
                        "response": out.get("final_response") or "",
                        "awaiting_confirmation": bool(out.get("awaiting_confirmation")),
                        "last_tool": out.get("last_tool"),
                        "provider": cfg.llm_backend,
                    },
                )
        except Exception as e:
            logger.exception("agent stream failed")
            detail = str(e) if _runtime_env() in ("dev", "test") else "agent stream failed"
            yield _sse("error", {"detail": detail or "agent stream failed"})

    return StreamingResponse(event_gen(), media_type="text/event-stream")


def _sse(event: str, data: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"



@app.delete("/v1/entries/{entry_id}")
def delete_entry(entry_id: str, namespace: Optional[str] = None, _: None = Depends(_require_api_key)):
    """Delete a single entry by ID."""
    store = _get_store()
    if not store.enabled:
        raise HTTPException(status_code=400, detail="retrieval disabled")
    ns = _fixed_namespace(namespace)
    store.delete_entry(entry_id, namespace=ns)
    return {"ok": True, "entry_id": entry_id}


@app.delete("/v1/namespace")
def delete_namespace(confirm: str = "", namespace: Optional[str] = None, _: None = Depends(_require_api_key)):
    """Clear all entries in a namespace. Requires confirm=true query param."""
    if confirm != "true":
        raise HTTPException(status_code=400, detail="confirm=true required")
    store = _get_store()
    if not store.enabled:
        raise HTTPException(status_code=400, detail="retrieval disabled")
    ns = _fixed_namespace(namespace)
    store.clear_namespace(ns)
    return {"ok": True, "namespace": ns}


def create_app() -> FastAPI:
    return app
