#!/usr/bin/env python3
"""
Compare evaluation results before and after DPO fine-tuning.

Generates a markdown summary showing improvements/regressions.
"""

import json
import argparse
import os
from typing import Dict, Any
from datetime import datetime

def load_eval_results(path: str) -> Dict[str, Any]:
    """Load evaluation results from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)

def extract_metrics(results: Dict[str, Any]) -> Dict[str, float]:
    """Extract aggregate metrics from results."""
    agg = results.get('aggregate_metrics', {})
    return {
        'faithfulness': agg.get('faithfulness', {}).get('mean', 0.0),
        'answer_relevancy': agg.get('answer_relevancy', {}).get('mean', 0.0),
        'no_invention': agg.get('no_invention', {}).get('mean', 0.0),
        'instruction_following': agg.get('instruction_following', {}).get('mean', 0.0),
        'latency': agg.get('latency', {}).get('mean', 0.0),
        'parse_failures': results.get('parse_failures', 0),
        'num_cases': results.get('num_cases', 0),
    }

def compare_results(before: Dict[str, Any], after: Dict[str, Any], dataset_name: str) -> str:
    """Generate comparison markdown."""
    before_metrics = extract_metrics(before)
    after_metrics = extract_metrics(after)
    
    md = f"# Before/After Comparison: {dataset_name}\n\n"
    md += f"**Generated:** {datetime.now().isoformat()}\n\n"
    
    md += "## Summary\n\n"
    
    # Parse failures
    before_pf = before_metrics['parse_failures']
    after_pf = after_metrics['parse_failures']
    if after_pf > before_pf:
        md += f"⚠️ **Parse Failures:** {before_pf} → {after_pf} (REGRESSION)\n\n"
    elif after_pf < before_pf:
        md += f"✅ **Parse Failures:** {before_pf} → {after_pf} (IMPROVEMENT)\n\n"
    else:
        md += f"✅ **Parse Failures:** {before_pf} → {after_pf} (NO CHANGE)\n\n"
    
    # Metrics table
    md += "## Metrics Comparison\n\n"
    md += "| Metric | Before | After | Change | Status |\n"
    md += "|--------|--------|-------|--------|--------|\n"
    
    metrics_to_compare = ['faithfulness', 'answer_relevancy', 'no_invention', 'instruction_following']
    
    for metric in metrics_to_compare:
        before_val = before_metrics[metric]
        after_val = after_metrics[metric]
        change = after_val - before_val
        change_pct = (change / before_val * 100) if before_val > 0 else 0
        
        # Determine status
        if metric == 'no_invention':
            # Must be 1.00
            if after_val < 1.00:
                status = "⚠️ REGRESSION"
            elif after_val == 1.00 and before_val < 1.00:
                status = "✅ IMPROVEMENT"
            else:
                status = "✅ NO CHANGE"
        elif metric == 'faithfulness':
            # Must be >= 0.95
            if after_val < 0.95:
                status = "⚠️ REGRESSION (below threshold)"
            elif after_val > before_val:
                status = "✅ IMPROVEMENT"
            elif after_val == before_val:
                status = "✅ NO CHANGE"
            else:
                status = "⚠️ REGRESSION"
        else:
            # Higher is better
            if after_val > before_val:
                status = "✅ IMPROVEMENT"
            elif after_val == before_val:
                status = "✅ NO CHANGE"
            else:
                status = "⚠️ REGRESSION"
        
        md += f"| {metric.replace('_', ' ').title()} | {before_val:.3f} | {after_val:.3f} | {change:+.3f} ({change_pct:+.1f}%) | {status} |\n"
    
    # Latency
    before_lat = before_metrics['latency']
    after_lat = after_metrics['latency']
    lat_change = after_lat - before_lat
    lat_change_pct = (lat_change / before_lat * 100) if before_lat > 0 else 0
    md += f"| Latency (s) | {before_lat:.2f} | {after_lat:.2f} | {lat_change:+.2f} ({lat_change_pct:+.1f}%) | ℹ️ INFO |\n"
    
    md += "\n## Acceptance Criteria\n\n"
    
    # Check gates
    gates_passed = []
    gates_failed = []
    
    if after_metrics['parse_failures'] == 0:
        gates_passed.append("Parse failures == 0")
    else:
        gates_failed.append(f"Parse failures == 0 (actual: {after_metrics['parse_failures']})")
    
    if after_metrics['no_invention'] == 1.00:
        gates_passed.append("No invention == 1.00")
    else:
        gates_failed.append(f"No invention == 1.00 (actual: {after_metrics['no_invention']:.3f})")
    
    if after_metrics['faithfulness'] >= 0.95:
        gates_passed.append("Faithfulness >= 0.95")
    else:
        gates_failed.append(f"Faithfulness >= 0.95 (actual: {after_metrics['faithfulness']:.3f})")
    
    if gates_passed:
        md += "✅ **Passed:**\n"
        for gate in gates_passed:
            md += f"- {gate}\n"
        md += "\n"
    
    if gates_failed:
        md += "⚠️ **Failed:**\n"
        for gate in gates_failed:
            md += f"- {gate}\n"
        md += "\n"
    
    md += "## Verdict\n\n"
    
    if len(gates_failed) == 0:
        md += "✅ **NO REGRESSIONS** - Tuning is safe to use.\n"
    else:
        md += "⚠️ **REGRESSIONS DETECTED** - Review before deploying.\n"
    
    return md

def main():
    parser = argparse.ArgumentParser(description='Compare before/after evaluation results')
    parser.add_argument('--before-baseline', type=str, help='Path to before baseline results JSON')
    parser.add_argument('--after-baseline', type=str, help='Path to after baseline results JSON')
    parser.add_argument('--before-quality', type=str, help='Path to before quality results JSON')
    parser.add_argument('--after-quality', type=str, required=True, help='Path to after quality results JSON')
    parser.add_argument('--dataset', type=str, default='quick_tests', help='Dataset name for report')
    parser.add_argument('--output', type=str, default='evals/summary_before_after_tuning.md', help='Output markdown file')
    args = parser.parse_args()
    
    # Load results
    after_quality = load_eval_results(args.after_quality)
    
    if args.before_quality:
        before_quality = load_eval_results(args.before_quality)
        md = compare_results(before_quality, after_quality, f"{args.dataset} (Quality Mode)")
    else:
        # Just report after results
        after_metrics = extract_metrics(after_quality)
        md = f"# After Tuning Results: {args.dataset}\n\n"
        md += f"**Generated:** {datetime.now().isoformat()}\n\n"
        md += "## Metrics\n\n"
        md += f"- Parse Failures: {after_metrics['parse_failures']}\n"
        md += f"- Mean Faithfulness: {after_metrics['faithfulness']:.3f}\n"
        md += f"- Mean No-Invention: {after_metrics['no_invention']:.3f}\n"
        md += f"- Mean Answer Relevancy: {after_metrics['answer_relevancy']:.3f}\n"
        md += f"- Mean Latency: {after_metrics['latency']:.2f}s\n"
    
    # Write output
    with open(args.output, 'w') as f:
        f.write(md)
    
    print(f"✅ Comparison report saved to: {args.output}")
    print("\n" + md)

if __name__ == "__main__":
    main()
