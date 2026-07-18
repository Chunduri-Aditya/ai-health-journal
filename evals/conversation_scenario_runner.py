#!/usr/bin/env python3
"""Drive multi-turn journaling scenarios through the real /analyze route and
report how well the conversation remembers itself.

Unlike evals/rag_retrieval_eval.py (which queries an isolated store once), this
runs the *full loop*: each turn is POSTed to /analyze, which stores the entry in
a real ephemeral Chroma store and then — on later turns — retrieves earlier ones
as grounding `sources`. The LLM is the deterministic ThemeAwareFakeProvider, so
the only thing under test is memory: does turn N surface the prior turns a scenario
declares in `expect_retrieves_prior_ids`, and does a crisis turn route to support?

Run:
    PYTHONPATH=. python evals/conversation_scenario_runner.py
or: make scenario-run

Exits non-zero if aggregate memory recall falls below SCENARIO_RECALL_FLOOR
(default 0.80) or any crisis expectation fails — a regression gate.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

# Repo root on path so `app`, `tests.support`, and `vector_store` import cleanly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RECALL_FLOOR = float(os.getenv("SCENARIO_RECALL_FLOOR", "0.80"))

PASS = "PASS ✅"
FAIL = "FAIL ❌"


def _run_scenario(scenario: Dict[str, Any], persist_dir: str) -> Dict[str, Any]:
    """Drive one scenario end to end; return per-turn + aggregate results."""
    os.environ["CHROMA_PERSIST_DIR"] = persist_dir
    os.environ["RETRIEVAL_ENABLED"] = "true"
    os.environ["VECTOR_BACKEND"] = "chroma"

    import app as app_module
    from tests.support.fake_provider import ThemeAwareFakeProvider
    from tests.support.scenarios import resolve_source_ids
    from vector_store.chroma_store import ChromaStore

    store = ChromaStore(default_namespace=f"scenario_{scenario['suite_id']}")
    app_module.vector_store = store
    app_module._provider = ThemeAwareFakeProvider()

    recall_scores: List[float] = []
    top1_hits: List[bool] = []
    crisis_ok: List[bool] = []
    turn_rows: List[str] = []

    with app_module.app.test_client() as client:
        for turn in scenario["turns"]:
            resp = client.post("/analyze", json={"entry": turn["entry"], "quality_mode": True})
            body = resp.get_json() or {}
            analysis = body.get("analysis", {})
            source_ids = [s["id"] for s in body.get("sources", [])]

            # Map the just-returned opaque source ids back to scenario turn ids
            # via the entry text we stored (the runner owns the id->text map).
            note = f"  turn {turn['turn']} [{turn['id']}]"

            expected = turn.get("expect_retrieves_prior_ids")
            if expected:
                # Sources are opaque entry_ ids; match them to prior turn ids by
                # comparing the stored snippet against earlier entries.
                matched = resolve_source_ids(body.get("sources", []), scenario, turn["turn"])
                hit_set = set(expected) & set(matched)
                recall = len(hit_set) / len(expected)
                top1 = bool(matched) and matched[0] in expected
                recall_scores.append(recall)
                top1_hits.append(top1)
                note += (
                    f"  recall={recall:.2f} top1={'✓' if top1 else '✗'} "
                    f"expected={expected} got={matched}"
                )
            else:
                note += f"  (no memory expectation)  sources={len(source_ids)}"

            if "expect_crisis_support" in turn:
                got = bool(analysis.get("crisis_support"))
                ok = got is turn["expect_crisis_support"]
                crisis_ok.append(ok)
                note += f"  crisis_support={got} {'✓' if ok else '✗'}"

            turn_rows.append(note)

    store.clear_namespace(f"scenario_{scenario['suite_id']}")

    mean_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 1.0
    return {
        "suite_id": scenario["suite_id"],
        "rows": turn_rows,
        "mean_recall": mean_recall,
        "top1_rate": (sum(top1_hits) / len(top1_hits)) if top1_hits else 1.0,
        "crisis_ok": all(crisis_ok),
        "crisis_checked": len(crisis_ok),
    }


def main() -> int:
    from tests.support.scenarios import all_scenario_paths, load_scenario, validate_scenario

    paths = all_scenario_paths()
    print(f"=== Conversation scenario runner ({len(paths)} scenarios) ===")
    print(f"  recall floor: {RECALL_FLOOR}\n")

    results = []
    for path in paths:
        scenario = load_scenario(path)
        validate_scenario(scenario)
        persist_dir = tempfile.mkdtemp(prefix="scenario_")
        try:
            result = _run_scenario(scenario, persist_dir)
        finally:
            shutil.rmtree(persist_dir, ignore_errors=True)
        results.append(result)

        print(f"-- {result['suite_id']} --")
        for row in result["rows"]:
            print(row)
        print(
            f"  → mean_recall={result['mean_recall']:.3f} "
            f"top1_rate={result['top1_rate']:.3f} "
            f"crisis={'ok' if result['crisis_ok'] else 'FAILED'} "
            f"({result['crisis_checked']} checked)\n"
        )

    overall_recall = sum(r["mean_recall"] for r in results) / len(results) if results else 0.0
    crisis_all_ok = all(r["crisis_ok"] for r in results)
    passed = overall_recall >= RECALL_FLOOR and crisis_all_ok

    print("=== summary ===")
    print(f"  overall mean recall: {overall_recall:.3f}  (floor {RECALL_FLOOR})")
    print(f"  crisis expectations: {'all met' if crisis_all_ok else 'FAILURES present'}")
    print(PASS if passed else FAIL)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
