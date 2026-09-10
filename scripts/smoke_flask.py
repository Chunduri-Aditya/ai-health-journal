#!/usr/bin/env python3
# GateGuard: callers start.sh, scripts/run_bugbot_verify.sh.
# No prior smoke_flask.py. Affected API: GET /ping, POST /analyze.
# Data schemas: {entry, quality_mode} -> {insight|error}.
# User: i want to always test for each time i start the project
"""Smoke-test a running Flask Journal UI (/analyze).

  JOURNAL_FLASK_URL=http://127.0.0.1:5050 python scripts/smoke_flask.py
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

BASE = os.getenv("JOURNAL_FLASK_URL", "http://127.0.0.1:5050").rstrip("/")
TIMEOUT = int(os.getenv("SMOKE_TIMEOUT_SEC", "180"))


def _req(method: str, path: str, body: dict | None = None) -> tuple[int, object]:
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        f"{BASE}{path}", data=data, headers=headers, method=method
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as resp:
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
    print(f"== flask smoke against {BASE} ==")

    code, ping = _req("GET", "/ping")
    ok = code == 200 and isinstance(ping, dict) and ping.get("status") == "ok"
    print(f"GET /ping     -> {code} {'OK' if ok else 'FAIL'}")
    failures += 0 if ok else 1

    code, body = _req(
        "POST",
        "/analyze",
        {
            "entry": "I slept well after a short walk today.",
            "quality_mode": True,
        },
    )
    insight = ""
    err = ""
    if isinstance(body, dict):
        insight = str(body.get("insight") or "")
        err = str(body.get("error") or "")
    else:
        err = str(body)[:200]
    ok = code == 200 and bool(insight.strip())
    print(f"POST /analyze -> {code} {'OK' if ok else 'FAIL'}")
    if insight:
        print(f"  insight_preview={insight[:160].replace(chr(10), ' ')!r}")
    if err:
        print(f"  error={err[:200]!r}")
        low = err.lower()
        if "api key" in low or "401" in low or "unauthorized" in low:
            print(
                "  hint: OPENAI_COMPATIBLE_API_KEY looks invalid/expired. "
                "Refresh the Groq key in .env or set LLM_BACKEND=ollama."
            )
    failures += 0 if ok else 1

    if failures:
        print(f"\nFAIL: {failures} flask smoke check(s) failed")
        return 1
    print("\nPASS: flask analyze connected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
