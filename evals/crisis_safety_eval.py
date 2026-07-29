#!/usr/bin/env python3
"""Sensitivity and specificity for the deterministic crisis/distress floor.

Why this exists
---------------
docs/CLINICAL_DESIGN.md section 8 recorded the gap plainly: crisis detection had
adversarial tests but no reported sensitivity or specificity. A green pytest run
says "the cases I thought of still pass". It does not say what fraction of real
crisis phrasing the floor catches, and it does not put a number on the failure
mode that actually matters, which is a missed crisis.

What is measured
----------------
The DETERMINISTIC FLOOR ONLY: `_is_crisis` is called with an empty verifier
verdict, so nothing here depends on an LLM. That is deliberate. The floor is the
guarantee that survives a failed verifier call, a wrong model judgment, or an
offline machine, so it is the thing worth having a number for. The verifier adds
coverage on top of this and is measured separately by the scenario runner.

Asymmetry
---------
Reported separately rather than folded into one score, because the two errors
are not equally bad (app.py:80):

  false negative - positivity or a reframe reaches someone in real danger.
                   Not recoverable. This is THE number to watch.
  false positive - one unnecessary supportive message. Recoverable, and an
                   accepted cost of the design.

Run:
    PYTHONPATH=. python evals/crisis_safety_eval.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Keep the app import side-effect free: no retrieval, no vector store.
os.environ.setdefault("RETRIEVAL_ENABLED", "false")
os.environ.setdefault("VECTOR_BACKEND", "none")

DEFAULT_CASES = Path(__file__).resolve().parent / "crisis_cases.json"

# Regression floor for crisis sensitivity, EXCLUDING documented known gaps.
# Set to 1.0 because the floor's whole purpose is to fail closed: a regression
# that starts missing covered phrasing must break the build, not degrade quietly.
SENSITIVITY_FLOOR = float(os.getenv("CRISIS_SENSITIVITY_FLOOR", "1.0"))
# Specificity is allowed to sit lower: false positives are an accepted cost.
SPECIFICITY_FLOOR = float(os.getenv("CRISIS_SPECIFICITY_FLOOR", "0.90"))


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    from app import _is_crisis, _is_distressed

    with args.cases.open(encoding="utf-8") as handle:
        cases = json.load(handle)["cases"]

    empty_verdict: Dict[str, object] = {}
    rows: List[dict] = []
    for case in cases:
        text = case["text"]
        rows.append(
            {
                **case,
                "crisis_fired": _is_crisis(text, empty_verdict),
                "distress_fired": _is_distressed(text),
            }
        )

    covered = [r for r in rows if not r.get("known_gap")]
    gaps = [r for r in rows if r.get("known_gap")]

    # ── Crisis tier: binary crisis vs not-crisis ──────────────────────────────
    tp = [r for r in covered if r["label"] == "crisis" and r["crisis_fired"]]
    fn = [r for r in covered if r["label"] == "crisis" and not r["crisis_fired"]]
    fp = [r for r in covered if r["label"] != "crisis" and r["crisis_fired"]]
    tn = [r for r in covered if r["label"] != "crisis" and not r["crisis_fired"]]

    sensitivity = _rate(len(tp), len(tp) + len(fn))
    specificity = _rate(len(tn), len(tn) + len(fp))
    ppv = _rate(len(tp), len(tp) + len(fp))
    npv = _rate(len(tn), len(tn) + len(fn))

    print("=== Crisis floor: deterministic only (empty verifier verdict) ===")
    print(f"  cases: {len(covered)} scored, {len(gaps)} documented gaps excluded\n")
    print(f"  Sensitivity (recall on crisis) : {sensitivity:.3f}   <- the number that matters")
    print(f"  Specificity (recall on neutral): {specificity:.3f}")
    print(f"  PPV (precision on crisis)      : {ppv:.3f}")
    print(f"  NPV                            : {npv:.3f}")
    print(f"  Confusion: TP={len(tp)} FN={len(fn)} FP={len(fp)} TN={len(tn)}")

    print("\n--- FALSE NEGATIVES (missed crisis: the unrecoverable error) ---")
    if not fn:
        print("  none")
    for r in fn:
        print(f"  [{r['category']}] {r['text']!r}")

    print("\n--- FALSE POSITIVES (unnecessary support message: accepted cost) ---")
    if not fp:
        print("  none")
    for r in fp:
        tag = " (documented, accepted)" if r.get("accepted_fp") else ""
        print(f"  [{r['category']}] {r['text']!r}{tag}")

    # ── Distress tier ─────────────────────────────────────────────────────────
    d_tp = [r for r in covered if r["label"] == "distress" and r["distress_fired"]]
    d_fn = [r for r in covered if r["label"] == "distress" and not r["distress_fired"]]
    d_fp = [r for r in covered if r["label"] == "neutral" and r["distress_fired"]]
    d_tn = [r for r in covered if r["label"] == "neutral" and not r["distress_fired"]]

    print("\n=== Distress tier (neutral vs distress; crisis rows excluded) ===")
    print(f"  Sensitivity: {_rate(len(d_tp), len(d_tp) + len(d_fn)):.3f}")
    print(f"  Specificity: {_rate(len(d_tn), len(d_tn) + len(d_fp)):.3f}")
    print(f"  Confusion: TP={len(d_tp)} FN={len(d_fn)} FP={len(d_fp)} TN={len(d_tn)}")
    for r in d_fn:
        print(f"    missed distress: {r['text']!r}")
    for r in d_fp:
        print(f"    neutral flagged distress: {r['text']!r}")

    # ── Tier ordering invariant ───────────────────────────────────────────────
    # Crisis must win over distress in _apply_reframe_gate. Any row where both
    # fire is fine (crisis takes precedence); a crisis row where ONLY distress
    # fires would mean a crisis entry gets the softer response.
    misrouted = [
        r for r in covered
        if r["label"] == "crisis" and r["distress_fired"] and not r["crisis_fired"]
    ]
    print("\n=== Tier ordering: crisis must never be handled as distress ===")
    print(f"  misrouted: {len(misrouted)}")
    for r in misrouted:
        print(f"    {r['text']!r}")

    # ── Documented gaps, reported not hidden ──────────────────────────────────
    print("\n=== Documented gaps (excluded from the scores above) ===")
    for r in gaps:
        state = "caught anyway" if r["crisis_fired"] else "missed, as documented"
        print(f"  [{state}] {r['text'][:58]!r}")

    passed = (
        sensitivity >= SENSITIVITY_FLOOR
        and specificity >= SPECIFICITY_FLOOR
        and not misrouted
    )
    print(
        f"\nfloors: sensitivity>={SENSITIVITY_FLOOR} specificity>={SPECIFICITY_FLOOR} misrouted==0"
    )
    print("PASS" if passed else "FAIL")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "sensitivity": sensitivity,
                    "specificity": specificity,
                    "ppv": ppv,
                    "npv": npv,
                    "confusion": {"tp": len(tp), "fn": len(fn), "fp": len(fp), "tn": len(tn)},
                    "false_negatives": [r["text"] for r in fn],
                    "false_positives": [r["text"] for r in fp],
                    "known_gaps_caught": [r["text"] for r in gaps if r["crisis_fired"]],
                },
                handle,
                indent=2,
            )
        print(f"wrote {args.json_out}")

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
