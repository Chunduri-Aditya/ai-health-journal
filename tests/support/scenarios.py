"""Loader and validator for multi-turn journaling scenarios.

A scenario is an ordered conversation ("journey") that drives the /analyze route
turn by turn. The validator is deliberately strict so scenario JSON cannot rot
silently: turns must be numbered from 1 without gaps, ids unique, and every
`expect_retrieves_prior_ids` must reference an *earlier* turn's id (memory can
only recall the past).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

SCENARIO_DIR = Path(__file__).resolve().parents[2] / "evals" / "scenarios"


def load_scenario(path: str | Path) -> Dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def all_scenario_paths() -> List[Path]:
    return sorted(SCENARIO_DIR.glob("*.json"))


def validate_scenario(scenario: Dict[str, Any]) -> None:
    """Raise AssertionError with a precise message on any structural problem."""
    assert isinstance(scenario, dict), "scenario must be an object"
    for key in ("suite_id", "description", "turns"):
        assert key in scenario, f"missing top-level key: {key!r}"

    turns = scenario["turns"]
    assert isinstance(turns, list) and turns, "turns must be a non-empty list"

    seen_ids: set[str] = set()
    for index, turn in enumerate(turns, start=1):
        assert isinstance(turn, dict), f"turn {index} must be an object"
        assert turn.get("turn") == index, f"turn {index} has out-of-order 'turn'={turn.get('turn')!r}"
        tid = turn.get("id")
        assert isinstance(tid, str) and tid, f"turn {index} needs a non-empty string id"
        assert tid not in seen_ids, f"duplicate turn id: {tid!r}"
        entry = turn.get("entry")
        assert isinstance(entry, str) and entry.strip(), f"turn {index} needs a non-empty entry"

        for prior in turn.get("expect_retrieves_prior_ids", []):
            assert prior in seen_ids, (
                f"turn {index} ({tid!r}) expects to retrieve {prior!r}, "
                f"which is not an earlier turn"
            )
        if "expect_crisis_support" in turn:
            assert isinstance(turn["expect_crisis_support"], bool), (
                f"turn {index} expect_crisis_support must be a bool"
            )
        seen_ids.add(tid)


def resolve_source_ids(
    sources: Sequence[Dict[str, Any]],
    scenario: Dict[str, Any],
    current_turn: int,
) -> List[str]:
    """Map opaque /analyze `sources` back to scenario turn ids, in rank order.

    The route returns entry_ ids the caller never chose, so retrieval is scored
    by matching each source's snippet against the text of an *earlier* turn. Used
    by both the slow integration test and the scenario runner.
    """
    prior = {t["id"]: t["entry"] for t in scenario["turns"] if t["turn"] < current_turn}
    resolved: List[str] = []
    for src in sources:
        snippet = (src.get("snippet") or "").rstrip("…")
        for tid, entry in prior.items():
            if snippet and (snippet in entry or entry.startswith(snippet)):
                resolved.append(tid)
                break
    return resolved
