#!/usr/bin/env python3
"""
Convert evaluation results into DPO preference pairs.

Reads baseline and quality mode results for the same entries,
creates preference pairs where quality mode is "chosen" and
baseline is "rejected" (when quality is better).

STRICT FILTERING:
- Quality parse_failures == 0
- Quality faithfulness >= 0.95
- Quality no_invention == 1.00
- Quality is better than baseline OR baseline is legacy format
"""

import json
import os
import sys
import argparse
from typing import Dict, List, Any
from datetime import datetime

def load_eval_results(results_path: str) -> Dict[str, Any]:
    """Load evaluation results from JSON file."""
    with open(results_path, 'r') as f:
        return json.load(f)

def build_prompt(entry: str, retrieved_context: str = "") -> str:
    """
    Build the full prompt for DPO training.
    Includes system constraints, entry, and retrieved context if available.
    """
    system_constraints = """CRITICAL RULES:
1. NEVER invent or assume facts about the user's past, history, or experiences not mentioned
2. ONLY use information from the current entry and any RETRIEVED_CONTEXT provided
3. Do not make assumptions about relationships, events, or patterns not explicitly stated
4. Do not provide medical diagnosis - suggest professional evaluation if concerns arise
5. Use hedged language ("may", "might", "could") when uncertain
6. Return ONLY valid JSON that matches the schema. No markdown, no commentary, no code fences."""
    
    prompt = f"{system_constraints}\n\n"
    prompt += f"Analyze this journal entry and return the JSON analysis:\n\n"
    prompt += f"JOURNAL ENTRY:\n{entry}\n\n"
    
    if retrieved_context:
        prompt += f"RETRIEVED_CONTEXT (from past entries):\n{retrieved_context}\n\n"
        prompt += "You may reference the retrieved context, but do not invent connections. "
        prompt += "Only use information explicitly stated in the entry or context.\n\n"
    
    prompt += "Return ONLY valid JSON, no additional text."
    return prompt

def contains_forbidden_content(text: str) -> bool:
    """
    Check if text contains forbidden content (diagnosis, medication directives, etc.).
    """
    forbidden_patterns = [
        r'\b(diagnos[ie]s?|diagnosed|diagnosing)\b',
        r'\b(take|prescribe|dosage|dose|mg|milligram)\s+\d+',
        r'\b(SSRI|antidepressant|medication|prescription)\s+(should|must|need)',
        r'\b100%\s+(certain|certainty|sure|guaranteed)',
        r'\b(send|upload|share|store)\s+(to|with|on)\s+(server|cloud|database)',
    ]
    import re
    text_lower = text.lower()
    for pattern in forbidden_patterns:
        if re.search(pattern, text_lower):
            return True
    return False


def should_keep_pair(baseline_result: Dict, quality_result: Dict, use_baseline_json: bool = False) -> tuple[bool, str]:
    """
    Determine if a pair should be kept based on STRICT quality criteria.
    
    Returns:
        (should_keep: bool, reason: str)
    """
    # Check quality parse failures
    quality_parse_failures = quality_result.get('parse_failures', 0)
    if quality_parse_failures > 0:
        return False, "quality_parse_failure"
    
    quality_metrics = quality_result.get('metrics', {})
    baseline_metrics = baseline_result.get('metrics', {})
    
    # STRICT: Quality faithfulness must be >= 0.95
    quality_faithfulness = quality_metrics.get('faithfulness', 0.0)
    if quality_faithfulness < 0.95:
        return False, "quality_faithfulness_too_low"
    
    # STRICT: Quality no_invention must be 1.00
    quality_no_invention = quality_metrics.get('no_invention', 0.0)
    if quality_no_invention < 1.00:
        return False, "quality_no_invention_not_perfect"
    
    # Get baseline output (prefer baseline_json if available)
    if use_baseline_json:
        baseline_output = baseline_result.get('baseline_json_output') or baseline_result.get('baseline_output')
    else:
        baseline_output = baseline_result.get('baseline_output')
    
    quality_output = quality_result.get('quality_output')
    
    # If baseline is legacy format (not JSON), quality is always better
    if baseline_output is None or baseline_output == '':
        return True, "baseline_legacy_format"
    
    # Check for forbidden content in baseline
    if contains_forbidden_content(baseline_output):
        return True, "baseline_contains_forbidden_content"
    
    # Compare metrics
    baseline_faithfulness = baseline_metrics.get('faithfulness', 0.0)
    baseline_no_invention = baseline_metrics.get('no_invention', 0.0)
    baseline_unsupported = baseline_metrics.get('unsupported_claims', [])
    baseline_answer_relevancy = baseline_metrics.get('answer_relevancy', 0.0)
    
    quality_unsupported = quality_metrics.get('unsupported_claims', [])
    quality_answer_relevancy = quality_metrics.get('answer_relevancy', 0.0)
    
    # Quality is better if:
    # - Higher faithfulness, OR
    # - Higher no_invention, OR
    # - Higher answer_relevancy, OR
    # - Baseline has unsupported claims but quality doesn't
    quality_better = (
        quality_faithfulness > baseline_faithfulness or
        quality_no_invention > baseline_no_invention or
        quality_answer_relevancy > baseline_answer_relevancy or
        (baseline_unsupported and len(baseline_unsupported) > 0 and 
         (not quality_unsupported or len(quality_unsupported) == 0))
    )
    
    # TIE-BREAKER: If metrics are equal (or very close), prefer quality if it has better structure/safety
    if not quality_better:
        # Check if it's a tie (within 0.01)
        is_tie = (
            abs(quality_faithfulness - baseline_faithfulness) < 0.01 and
            abs(quality_no_invention - baseline_no_invention) < 0.01 and
            abs(quality_answer_relevancy - baseline_answer_relevancy) < 0.01
        )
        
        if is_tie:
            # Tie-breaker: prefer quality if it has fewer unsupported claims or better refusal language
            baseline_has_claims = bool(baseline_unsupported and len(baseline_unsupported) > 0)
            quality_has_claims = bool(quality_unsupported and len(quality_unsupported) > 0)
            
            if not quality_has_claims and baseline_has_claims:
                return True, "quality_tie_breaker_fewer_claims"
            
            # Check for better refusal/safety language in quality (heuristic)
            quality_text = quality_output.lower() if quality_output else ""
            baseline_text = baseline_output.lower() if baseline_output else ""
            
            safety_phrases = ["cannot", "unable to", "don't have", "should consult", "professional", "cannot diagnose", "cannot prescribe"]
            quality_has_safety = any(phrase in quality_text for phrase in safety_phrases)
            baseline_has_safety = any(phrase in baseline_text for phrase in safety_phrases)
            
            if quality_has_safety and not baseline_has_safety:
                return True, "quality_tie_breaker_better_safety"
        
        return False, "quality_not_better_than_baseline"
    
    return True, "quality_better"

def build_dpo_pairs(baseline_results_path: str, quality_results_path: str, output_path: str, use_baseline_json: bool = False) -> Dict[str, Any]:
    """
    Build DPO preference pairs from evaluation results.
    Matches entries by entry_id and dataset_version.
    
    Returns:
        Dictionary with stats about pairs created
    """
    baseline_results = load_eval_results(baseline_results_path)
    quality_results = load_eval_results(quality_results_path)
    
    # Create entry_id -> result mapping
    baseline_map = {}
    quality_map = {}
    
    for r in baseline_results.get('case_results', []):
        entry_id = r.get('entry_id', r.get('entry', ''))  # Fallback to entry if no entry_id
        baseline_map[entry_id] = r
    
    for r in quality_results.get('case_results', []):
        entry_id = r.get('entry_id', r.get('entry', ''))
        quality_map[entry_id] = r
    
    pairs = []
    stats = {
        'total_entries': 0,
        'pairs_created': 0,
        'pairs_filtered_out': 0,
        'reasons': {}
    }
    
    # Process entries that exist in both results
    common_entry_ids = set(baseline_map.keys()) & set(quality_map.keys())
    stats['total_entries'] = len(common_entry_ids)
    
    for entry_id in common_entry_ids:
        baseline_result = baseline_map[entry_id]
        quality_result = quality_map[entry_id]
        
        # Verify dataset_version matches (if present)
        baseline_version = baseline_result.get('dataset_version', '1.0')
        quality_version = quality_result.get('dataset_version', '1.0')
        if baseline_version != quality_version:
            stats['pairs_filtered_out'] += 1
            reason = 'dataset_version_mismatch'
            stats['reasons'][reason] = stats['reasons'].get(reason, 0) + 1
            continue
        
        # Check if we should keep this pair
        should_keep, reason = should_keep_pair(baseline_result, quality_result, use_baseline_json=use_baseline_json)
        
        if not should_keep:
            stats['pairs_filtered_out'] += 1
            stats['reasons'][reason] = stats['reasons'].get(reason, 0) + 1
            continue
        
        # Build the pair
        entry = quality_result.get('entry', baseline_result.get('entry', ''))
        retrieved_context = quality_result.get('retrieved_context', '')
        prompt = build_prompt(entry, retrieved_context)
        
        # Use quality_output and baseline output (prefer baseline_json if available)
        chosen = quality_result.get('quality_output') or json.dumps(quality_result.get('final_json', quality_result.get('analysis', {})))
        if use_baseline_json:
            rejected = baseline_result.get('baseline_json_output') or baseline_result.get('baseline_output') or json.dumps(baseline_result.get('analysis', {}))
        else:
            rejected = baseline_result.get('baseline_output') or json.dumps(baseline_result.get('analysis', {}))
        
        pair = {
            'prompt': prompt,
            'chosen': chosen,
            'rejected': rejected,
            'metadata': {
                'entry_id': entry_id,
                'entry': entry,
                'category': quality_result.get('category', 'unknown'),
                'quality_faithfulness': quality_result.get('metrics', {}).get('faithfulness', 0.0),
                'quality_no_invention': quality_result.get('metrics', {}).get('no_invention', 0.0),
                'baseline_faithfulness': baseline_result.get('metrics', {}).get('faithfulness', 0.0),
                'baseline_no_invention': baseline_result.get('metrics', {}).get('no_invention', 0.0),
                'quality_unsupported_claims': quality_result.get('metrics', {}).get('unsupported_claims', []),
                'baseline_unsupported_claims': baseline_result.get('metrics', {}).get('unsupported_claims', []),
                'rag_enabled': quality_result.get('rag_enabled', False)
            }
        }
        
        pairs.append(pair)
        stats['pairs_created'] += 1
    
    # Write pairs to JSONL
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for pair in pairs:
            f.write(json.dumps(pair) + '\n')
    
    # Write sample file (first 3 pairs)
    sample_path = output_path.replace('.jsonl', '.sample.jsonl')
    with open(sample_path, 'w') as f:
        for pair in pairs[:3]:
            # Truncate outputs for readability
            sample_pair = {
                'prompt': pair['prompt'][:200] + '...' if len(pair['prompt']) > 200 else pair['prompt'],
                'chosen': pair['chosen'][:300] + '...' if len(pair['chosen']) > 300 else pair['chosen'],
                'rejected': pair['rejected'][:300] + '...' if len(pair['rejected']) > 300 else pair['rejected'],
                'metadata': pair['metadata']
            }
            f.write(json.dumps(sample_pair) + '\n')
    
    return {
        'stats': stats,
        'output_path': output_path,
        'sample_path': sample_path,
        'num_pairs': len(pairs)
    }

def main():
    parser = argparse.ArgumentParser(description='Build DPO dataset from evaluation results')
    parser.add_argument('--baseline', type=str, default=None, help='Path to baseline (legacy) evaluation results JSON')
    parser.add_argument('--baseline_json', type=str, default=None, help='Path to baseline_json evaluation results JSON (preferred)')
    parser.add_argument('--quality', type=str, required=True, help='Path to quality evaluation results JSON')
    parser.add_argument('--output', type=str, default='train/dpo_pairs.jsonl', help='Output path for DPO pairs JSONL')
    args = parser.parse_args()
    
    if not args.baseline_json and not args.baseline:
        parser.error("Either --baseline_json or --baseline must be provided")
    
    use_baseline_json = args.baseline_json is not None
    baseline_path = args.baseline_json or args.baseline
    
    print("Building DPO preference pairs...")
    print(f"Baseline results: {baseline_path} ({'baseline_json' if use_baseline_json else 'baseline legacy'})")
    print(f"Quality results: {args.quality}")
    print(f"Output: {args.output}\n")
    
    result = build_dpo_pairs(baseline_path, args.quality, args.output, use_baseline_json=use_baseline_json)
    
    print("=" * 60)
    print("DPO Dataset Build Complete")
    print("=" * 60)
    print(f"Total entries processed: {result['stats']['total_entries']}")
    print(f"Pairs created: {result['stats']['pairs_created']}")
    print(f"Pairs filtered out: {result['stats']['pairs_filtered_out']}")
    print(f"\nFilter reasons:")
    for reason, count in result['stats']['reasons'].items():
        print(f"  - {reason}: {count}")
    print(f"\nOutput saved to: {result['output_path']}")
    print(f"Sample saved to: {result['sample_path']}")
    print(f"Total pairs: {result['num_pairs']}")

if __name__ == "__main__":
    main()
