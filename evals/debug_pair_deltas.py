#!/usr/bin/env python3
"""
Debug script to analyze why DPO pairs are being filtered.

Shows per-case deltas between baseline_json and quality modes.
"""

import json
import sys
from typing import Dict, Any, List, Tuple

def get_cases(obj: Dict[str, Any]) -> List[Dict]:
    """Extract case results from evaluation JSON."""
    # Try common keys
    for key in ["case_results", "cases", "results", "items"]:
        if key in obj and isinstance(obj[key], list):
            return obj[key]
    
    # If no list found, check top-level structure
    if isinstance(obj, list):
        return obj
    
    raise KeyError(f"Can't find cases list. Keys: {list(obj.keys())}")

def extract_metrics(case: Dict[str, Any]) -> Tuple[float, float, float]:
    """Extract faithfulness, relevancy, no_invention from case."""
    metrics = case.get('metrics', {})
    
    faith = metrics.get('faithfulness', case.get('faithfulness', 0.0))
    rel = metrics.get('answer_relevancy', metrics.get('relevancy', case.get('relevancy', 0.0)))
    no_invent = metrics.get('no_invention', case.get('no_invention', 0.0))
    
    return float(faith), float(rel), float(no_invent)

def get_entry_text(case: Dict[str, Any]) -> str:
    """Extract entry text from case."""
    return case.get('entry', case.get('prompt', case.get('input', '')))[:80]

def main():
    if len(sys.argv) < 3:
        print("Usage: python evals/debug_pair_deltas.py <baseline_json> <quality_json>")
        sys.exit(1)
    
    baseline_path = sys.argv[1]
    quality_path = sys.argv[2]
    
    print(f"Loading baseline: {baseline_path}")
    print(f"Loading quality: {quality_path}\n")
    
    baseline_data = json.load(open(baseline_path))
    quality_data = json.load(open(quality_path))
    
    baseline_cases = get_cases(baseline_data)
    quality_cases = get_cases(quality_data)
    
    print(f"Baseline cases: {len(baseline_cases)}")
    print(f"Quality cases: {len(quality_cases)}\n")
    
    if len(baseline_cases) != len(quality_cases):
        print(f"⚠️  Warning: Case count mismatch ({len(baseline_cases)} vs {len(quality_cases)})")
        print("Matching by entry_id...\n")
        
        # Create entry_id -> case mapping
        baseline_map = {}
        for case in baseline_cases:
            entry_id = case.get('entry_id', case.get('entry', ''))
            baseline_map[entry_id] = case
        
        quality_map = {}
        for case in quality_cases:
            entry_id = case.get('entry_id', case.get('entry', ''))
            quality_map[entry_id] = case
        
        common_ids = set(baseline_map.keys()) & set(quality_map.keys())
        print(f"Common entry_ids: {len(common_ids)}")
        
        baseline_cases = [baseline_map[eid] for eid in sorted(common_ids)]
        quality_cases = [quality_map[eid] for eid in sorted(common_ids)]
    
    rows = []
    for i, (cb, cq) in enumerate(zip(baseline_cases, quality_cases), start=1):
        fb, rb, nb = extract_metrics(cb)
        fq, rq, nq = extract_metrics(cq)
        
        # Composite scores
        sb = (fb + rb + nb) / 3
        sq = (fq + rq + nq) / 3
        delta = sq - sb
        
        entry_text = get_entry_text(cb)
        entry_id = cb.get('entry_id', f'case_{i}')
        
        rows.append((
            delta, i, sb, sq,
            (fb, rb, nb), (fq, rq, nq),
            entry_text, entry_id
        ))
    
    # Sort by delta (largest improvement first)
    rows.sort(reverse=True, key=lambda x: x[0])
    
    print("=" * 80)
    print("TOP 20 IMPROVEMENTS (Quality - Baseline)")
    print("=" * 80)
    print(f"{'#':<4} {'Δ':<8} {'Base':<8} {'Qual':<8} {'Base (F/R/N)':<20} {'Qual (F/R/N)':<20} {'Entry':<30}")
    print("-" * 80)
    
    for d, i, sb, sq, sbt, sqt, p, eid in rows[:20]:
        base_str = f"{sbt[0]:.2f}/{sbt[1]:.2f}/{sbt[2]:.2f}"
        qual_str = f"{sqt[0]:.2f}/{sqt[1]:.2f}/{sqt[2]:.2f}"
        print(f"{i:<4} {d:+.3f}   {sb:.3f}    {sq:.3f}    {base_str:<20} {qual_str:<20} {p[:30]}")
    
    print("\n" + "=" * 80)
    print("BOTTOM 20 (Quality Worse Than Baseline)")
    print("=" * 80)
    print(f"{'#':<4} {'Δ':<8} {'Base':<8} {'Qual':<8} {'Base (F/R/N)':<20} {'Qual (F/R/N)':<20} {'Entry':<30}")
    print("-" * 80)
    
    for d, i, sb, sq, sbt, sqt, p, eid in rows[-20:]:
        base_str = f"{sbt[0]:.2f}/{sbt[1]:.2f}/{sbt[2]:.2f}"
        qual_str = f"{sqt[0]:.2f}/{sqt[1]:.2f}/{sqt[2]:.2f}"
        print(f"{i:<4} {d:+.3f}   {sb:.3f}    {sq:.3f}    {base_str:<20} {qual_str:<20} {p[:30]}")
    
    # Statistics
    print("\n" + "=" * 80)
    print("STATISTICS")
    print("=" * 80)
    
    deltas = [r[0] for r in rows]
    positive_deltas = [d for d in deltas if d > 0.01]
    negative_deltas = [d for d in deltas if d < -0.01]
    near_zero = [d for d in deltas if -0.01 <= d <= 0.01]
    
    print(f"Total cases: {len(rows)}")
    print(f"Quality better (Δ > 0.01): {len(positive_deltas)} ({len(positive_deltas)/len(rows)*100:.1f}%)")
    print(f"Quality worse (Δ < -0.01): {len(negative_deltas)} ({len(negative_deltas)/len(rows)*100:.1f}%)")
    print(f"Near tie (|Δ| <= 0.01): {len(near_zero)} ({len(near_zero)/len(rows)*100:.1f}%)")
    print(f"\nMean delta: {sum(deltas)/len(deltas):.4f}")
    print(f"Median delta: {sorted(deltas)[len(deltas)//2]:.4f}")
    print(f"Max improvement: {max(deltas):.4f}")
    print(f"Max regression: {min(deltas):.4f}")
    
    # Check for cases where quality should win but might be filtered
    print("\n" + "=" * 80)
    print("CASES WHERE QUALITY SHOULD WIN (but might be filtered)")
    print("=" * 80)
    
    quality_wins = []
    for d, i, sb, sq, sbt, sqt, p, eid in rows:
        fb, rb, nb = sbt
        fq, rq, nq = sqt
        
        # Quality wins if:
        # - Higher faithfulness AND no_invention == 1.00
        # - OR no_invention higher
        # - OR baseline has issues but quality doesn't
        should_win = (
            (fq >= 0.95 and nq == 1.00 and (fq > fb or nq > nb)) or
            (fq >= 0.95 and nq == 1.00 and fb < 0.95) or
            (fq >= 0.95 and nq == 1.00 and nb < 1.00)
        )
        
        if should_win and d <= 0.01:  # Should win but delta is small
            quality_wins.append((i, eid, d, fb, fq, nb, nq, p))
    
    if quality_wins:
        print(f"Found {len(quality_wins)} cases where quality should win but delta is small:")
        for i, eid, d, fb, fq, nb, nq, p in quality_wins[:10]:
            print(f"  #{i} ({eid}): Δ={d:+.3f}, base_f={fb:.2f} qual_f={fq:.2f}, base_ni={nb:.2f} qual_ni={nq:.2f} | {p[:50]}")
    else:
        print("No obvious cases where quality should win but doesn't.")

if __name__ == "__main__":
    main()
