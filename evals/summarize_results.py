#!/usr/bin/env python3
"""
Summarize evaluation result JSON files into a markdown report.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def load_result(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Result file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _pct_with_grounding(case_results: List[Dict[str, Any]]) -> float:
    if not case_results:
        return 0.0
    with_grounding = 0
    for case in case_results:
        analysis = case.get("analysis") or {}
        gs = analysis.get("grounding_sources") or []
        if isinstance(gs, list) and len(gs) > 0:
            with_grounding += 1
    return 100.0 * with_grounding / len(case_results)


def summarize(paths: List[Path]) -> str:
    rows = []
    by_mode: Dict[str, Dict[str, Any]] = {}

    for p in paths:
        if not p or not p.exists():
            print(f"Warning: skipping missing file: {p}", file=sys.stderr)
            continue
        try:
            data = load_result(p)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: failed to load {p}: {e}", file=sys.stderr)
            continue
        if not data:
            print(f"Warning: empty data from {p}", file=sys.stderr)
            continue
        mode = data.get("mode", p.stem)
        agg = data.get("aggregate_metrics", {})
        faith = agg.get("faithfulness", {}).get("mean", 0.0)
        no_inv = agg.get("no_invention", {}).get("mean", 0.0)
        relev = agg.get("answer_relevancy", {}).get("mean", 0.0)
        parse_fail = data.get("parse_failures", 0)
        num_cases = data.get("num_cases", 0)
        pct_ground = _pct_with_grounding(data.get("case_results", []))
        rows.append(
            {
                "mode": mode,
                "faithfulness": faith,
                "no_invention": no_inv,
                "answer_relevancy": relev,
                "pct_grounding": pct_ground,
                "parse_failures": parse_fail,
                "num_cases": num_cases,
            }
        )
        by_mode[mode] = rows[-1]

    lines = []
    lines.append("# Evaluation Summary\n")
    lines.append("| Mode | Faithfulness | No Invention | Answer Relevancy | % with Grounding | Parse Failures |")
    lines.append("|------|-------------:|-------------:|-----------------:|-----------------:|---------------:|")
    for r in rows:
        lines.append(
            f"| {r['mode']} | {r['faithfulness']:.3f} | {r['no_invention']:.3f} | "
            f"{r['answer_relevancy']:.3f} | {r['pct_grounding']:.1f}% | "
            f"{r['parse_failures']}/{r['num_cases']} |"
        )

    # Optional baseline vs quality deltas
    if "baseline_json" in by_mode and "quality" in by_mode:
        b = by_mode["baseline_json"]
        q = by_mode["quality"]
        lines.append("\n## Deltas (Quality - Baseline JSON)\n")
        for metric in ["faithfulness", "no_invention", "answer_relevancy"]:
            dv = q[metric] - b[metric]
            lines.append(f"- **{metric}**: {dv:+.3f}")

    return "\n".join(lines) + "\n"


def main(argv: List[str]) -> None:
    if len(argv) < 2:
        print("Usage: summarize_results.py results1.json [results2.json ...]", file=sys.stderr)
        sys.exit(1)

    paths = [Path(p) for p in argv[1:] if p and p.strip()]
    if not paths:
        print("Error: No valid file paths provided", file=sys.stderr)
        sys.exit(1)
    md = summarize(paths)
    print(md)


if __name__ == "__main__":
    main(sys.argv)

