#!/usr/bin/env python3
# GateGuard: callers: operator shell / manual smoke.
# No existing scripts/smoke_service.py (only scripts/live_agent_eval.py for deployed RAG).
# No data files. JSON: {session_id, message} -> AgentInvokeResponse.
# User: "use the suitable to test this project and make sure everything is connected correctly"
"""Smoke-test a running Journal Agent service.

  JOURNAL_AGENT_URL=http://127.0.0.1:8080 python scripts/smoke_service.py
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

BASE = os.getenv("JOURNAL_AGENT_URL", "http://127.0.0.1:8080").rstrip("/")
API_KEY = os.getenv("JOURNAL_AGENT_API_KEY", "").strip()


def _req(method: str, path: str, body: dict | None = None) -> tuple[int, object]:
    data = None
    headers = {"Accept": "application/json, text/html"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
        headers["Accept"] = "application/json"
    if API_KEY:
        headers["X-API-Key"] = API_KEY
    request = urllib.request.Request(
        f"{BASE}{path}", data=data, headers=headers, method=method
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as resp:
            raw = resp.read().decode("utf-8")
            try:
                return resp.status, json.loads(raw)
            except json.JSONDecodeError:
                return resp.status, raw
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = raw
        return e.code, payload


def main() -> int:
    failures = 0
    print(f"== smoke against {BASE} ==")

    code, root = _req("GET", "/")
    ok = code == 200 and (
        (isinstance(root, dict) and root.get("service") == "Journal Agent")
        or (isinstance(root, str) and "Journal Agent" in root)
    )
    print(f"GET /          -> {code} {'OK' if ok else 'FAIL'}")
    failures += 0 if ok else 1

    code, ready = _req("GET", "/readyz")
    llm = (ready or {}).get("llm", {}) if isinstance(ready, dict) else {}
    ok = code == 200 and isinstance(ready, dict) and ready.get("status") == "ready"
    print(
        f"GET /readyz    -> {code} {'OK' if ok else 'FAIL'} "
        f"(llm={llm.get('backend')} healthy={llm.get('healthy')})"
    )
    failures += 0 if ok else 1

    code, health = _req("GET", "/healthz")
    ok = code == 200 and isinstance(health, dict)
    backend = health.get("llm_backend") if isinstance(health, dict) else "?"
    print(f"GET /healthz   -> {code} {'OK' if ok else 'FAIL'} (llm_backend={backend})")
    failures += 0 if ok else 1

    code, body = _req(
        "POST",
        "/v1/agent/invoke",
        {
            "session_id": "smoke-test",
            "message": "I slept well after a walk today.",
        },
    )
    ok = code == 200 and isinstance(body, dict) and bool(body.get("response"))
    preview = ""
    if isinstance(body, dict):
        preview = str(body.get("response", ""))[:160].replace("\n", " ")
        provider = body.get("provider")
    else:
        preview = str(body)[:160]
        provider = "?"
    print(f"POST /v1/agent/invoke -> {code} {'OK' if ok else 'FAIL'}")
    print(f"  provider={provider}")
    print(f"  response_preview={preview!r}")
    failures += 0 if ok else 1

    if failures:
        print(f"\nFAIL: {failures} check(s) failed")
        return 1
    print("\nPASS: service connected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
