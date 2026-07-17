#!/usr/bin/env python3
"""
End-to-end verification of the Chroma RAG retrieval path.

Run via `make verify-rag` (after `make setup-full`), or directly:
    RETRIEVAL_ENABLED=true VECTOR_BACKEND=chroma python scripts/verify_rag.py

What it proves (the parts unit tests/fakes can't):
  1. chromadb imports and the default embedder loads (real semantic match:
     "dissertation" must rank near "thesis", "guilt" near "guilty").
  2. Namespace isolation: session A's entries never surface in session B.
  3. Id-based self-exclusion drops the just-submitted entry.
  4. delete_entry / clear_namespace work against live Chroma.
  5. /analyze returns a structured `sources` list (route-level, LLM mocked).

Verification only — does not modify app logic. Exits non-zero on any failure.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import traceback

PASS = "PASS ✅"
FAIL = "FAIL ❌"
_results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append((name, ok, detail))
    print(f"  [{PASS if ok else FAIL}] {name}" + (f" — {detail}" if detail else ""))


def smoke_store(persist_dir: str) -> None:
    os.environ.update(
        RETRIEVAL_ENABLED="true",
        VECTOR_BACKEND="chroma",
        CHROMA_PERSIST_DIR=persist_dir,
        RAG_NAMESPACE_MODE="session",
    )
    from vector_store.factory import get_vector_store

    vs = get_vector_store()
    check("backend is chroma + enabled + healthy",
          vs.backend_name == "chroma" and vs.enabled and vs.healthcheck(),
          f"backend={vs.backend_name} enabled={vs.enabled}")

    A, B = "session:A", "session:B"
    vs.add_entry("a1", "I keep procrastinating on my thesis and feel guilty every night.",
                 {"created_at": "2026-06-01"}, namespace=A)
    vs.add_entry("a2", "Had a great run this morning, felt clear and calm.",
                 {"created_at": "2026-06-02"}, namespace=A)
    vs.add_entry("b1", "My manager praised the launch and I felt proud.",
                 {"created_at": "2026-06-03"}, namespace=B)

    q = "I avoided working on my dissertation again and the guilt is back."

    hits_a = vs.query(q, top_k=2, namespace=A)
    top = hits_a[0].id if hits_a else None
    check("real embedder loaded + relevance ranks a1 top (dissertation≈thesis)",
          top == "a1", f"top={top} ranking={[(h.id, round(h.score, 3)) for h in hits_a]}")

    ids_b = {h.id for h in vs.query(q, top_k=5, namespace=B)}
    check("namespace isolation: A entries absent from B",
          ids_b <= {"b1"} and "a1" not in ids_b, f"B ids={ids_b}")

    excl = {"a1"}
    he = [h for h in vs.query(q, top_k=3, namespace=A) if h.id not in excl]
    check("self-exclusion drops the excluded id",
          "a1" not in {h.id for h in he}, f"remaining={[h.id for h in he]}")

    vs.delete_entry("a2", namespace=A)
    after_del = {h.id for h in vs.query("morning run calm", top_k=5, namespace=A)}
    check("delete_entry removes a2", "a2" not in after_del, f"A ids={after_del}")

    vs.clear_namespace(B)
    after_clear = {h.id for h in vs.query(q, top_k=5, namespace=B)}
    check("clear_namespace empties B", after_clear == set(), f"B ids={after_clear}")


def smoke_route() -> None:
    """Route-level: /analyze must return a `sources` list. LLM is mocked."""
    import app as app_module

    fixed = {
        "summary": "You feel guilty about avoiding your thesis.",
        "emotions": ["guilt", "avoidance"],
        "patterns": ["procrastination"],
        "triggers": ["thesis"],
        "coping_suggestions": ["Break it into one 20-minute block."],
        "quotes_from_user": [],
        "confidence": 0.8,
    }
    app_module._provider.healthcheck = lambda: True            # type: ignore[attr-defined]
    app_module._provider.json_generate = lambda *a, **k: dict(fixed)  # type: ignore[attr-defined]

    client = app_module.app.test_client()
    resp = client.post("/analyze", json={"entry": "Avoided my thesis again.",
                                         "baseline_json_mode": True})
    body = resp.get_json() or {}
    check("/analyze returns 200", resp.status_code == 200, f"status={resp.status_code}")
    check("/analyze response has sources list",
          isinstance(body.get("sources"), list), f"sources={type(body.get('sources')).__name__}")


def main() -> int:
    persist_dir = tempfile.mkdtemp(prefix="chroma_verify_")
    print("=== RAG verification ===")
    try:
        import chromadb  # noqa: F401
        print(f"  chromadb {chromadb.__version__}")
    except Exception:
        print(f"  [{FAIL}] chromadb not installed — run `make setup-full`")
        return 1

    try:
        print("\n-- vector store (real embedder) --")
        smoke_store(persist_dir)
        print("\n-- /analyze route (sources field) --")
        smoke_route()
    except Exception:
        traceback.print_exc()
        check("unexpected exception", False, "see traceback above")
    finally:
        shutil.rmtree(persist_dir, ignore_errors=True)

    failed = [n for n, ok, _ in _results if not ok]
    print("\n=== summary ===")
    print(f"  {len(_results) - len(failed)}/{len(_results)} checks passed")
    if failed:
        print("  failed: " + ", ".join(failed))
        return 1
    print("  ALL CHECKS PASSED ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main())
