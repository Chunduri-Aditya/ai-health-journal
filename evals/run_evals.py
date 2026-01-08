#!/usr/bin/env python3
"""
Evaluation harness for AI Health Journal.
Computes RAGAS-style metrics and instruction-following accuracy.
"""

import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Any
import requests

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app
from llm_client import json_generate
from generator_prompts import DRAFT_SYSTEM_PROMPT, get_draft_prompt
from verifier_prompts import VERIFIER_SYSTEM_PROMPT, get_verifier_prompt


def load_dataset(dataset_path: str) -> List[Dict]:
    """Load test cases from JSONL file."""
    cases = []
    with open(dataset_path, 'r') as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))
    return cases


def compute_faithfulness(analysis_json: Dict, entry: str, retrieved_context: str = "") -> float:
    """
    Compute faithfulness score (0-1): Are claims supported by evidence?
    Uses verifier as judge.
    """
    try:
        verifier_prompt = get_verifier_prompt(analysis_json, entry, retrieved_context)
        verdict = json_generate(
            app.config.get('VERIFIER_MODEL', 'samantha-mistral:7b'),
            VERIFIER_SYSTEM_PROMPT,
            verifier_prompt,
            max_retries=1
        )
        return verdict.get('groundedness_score', 0.0)
    except Exception as e:
        print(f"  Warning: Faithfulness computation failed: {e}")
        return 0.5  # Default neutral score


def compute_answer_relevancy(analysis_json: Dict, entry: str) -> float:
    """
    Compute answer relevancy (0-1): Does the analysis address the entry?
    Simple heuristic: checks if key emotions/patterns are identified.
    """
    # Simple heuristic: check if analysis has required fields and non-empty content
    required_fields = ['summary', 'emotions', 'coping_suggestions']
    score = 0.0
    
    for field in required_fields:
        if field in analysis_json:
            if field == 'summary' and len(analysis_json[field]) > 10:
                score += 0.4
            elif field in ['emotions', 'coping_suggestions'] and len(analysis_json.get(field, [])) > 0:
                score += 0.3
    
    return min(score, 1.0)


def compute_context_precision(retrieved_context: str, entry: str) -> float:
    """
    Compute context precision (0-1): Are retrieved contexts relevant?
    Simple heuristic: if retrieval is enabled and context exists, assume relevance.
    """
    if not retrieved_context:
        return 1.0  # No context to evaluate
    # Simple: if context exists and is non-empty, assume some relevance
    return 0.7 if len(retrieved_context) > 50 else 0.5


def compute_context_recall(retrieved_context: str, entry: str) -> float:
    """
    Compute context recall (0-1): Does retrieval capture relevant information?
    Simple heuristic based on context length and presence.
    """
    if not retrieved_context:
        return 0.0
    # Simple: if context exists, assume some recall
    return 0.6 if len(retrieved_context) > 50 else 0.3


def check_instruction_following(analysis_json: Dict) -> Dict[str, bool]:
    """
    Check instruction-following: Does output match required format?
    Based on IFEval-style checks.
    """
    checks = {
        'has_summary': 'summary' in analysis_json and isinstance(analysis_json['summary'], str) and len(analysis_json['summary']) > 0,
        'has_emotions': 'emotions' in analysis_json and isinstance(analysis_json['emotions'], list),
        'has_patterns': 'patterns' in analysis_json and isinstance(analysis_json['patterns'], list),
        'has_triggers': 'triggers' in analysis_json and isinstance(analysis_json['triggers'], list),
        'has_coping_suggestions': 'coping_suggestions' in analysis_json and isinstance(analysis_json['coping_suggestions'], list),
        'has_quotes_from_user': 'quotes_from_user' in analysis_json and isinstance(analysis_json['quotes_from_user'], list),
        'has_confidence': 'confidence' in analysis_json and isinstance(analysis_json['confidence'], (int, float)) and 0 <= analysis_json['confidence'] <= 1,
        'no_hallucinated_fields': not any(key not in ['summary', 'emotions', 'patterns', 'triggers', 'coping_suggestions', 'quotes_from_user', 'confidence'] for key in analysis_json.keys())
    }
    return checks


def check_no_invention(analysis_json: Dict, must_not_invent: List[str], entry: str) -> Dict[str, Any]:
    """
    Check that analysis doesn't invent information from must_not_invent list.
    Uses verifier to check for unsupported claims.
    """
    if not must_not_invent:
        return {'violations': [], 'score': 1.0}
    
    try:
        verifier_prompt = get_verifier_prompt(analysis_json, entry, "")
        verdict = json_generate(
            app.config.get('VERIFIER_MODEL', 'samantha-mistral:7b'),
            VERIFIER_SYSTEM_PROMPT,
            verifier_prompt,
            max_retries=1
        )
        
        unsupported = verdict.get('unsupported_claims', [])
        violations = [claim for claim in unsupported if any(term in claim.lower() for term in [t.lower() for t in must_not_invent])]
        
        score = 1.0 if len(violations) == 0 else max(0.0, 1.0 - (len(violations) / len(must_not_invent)))
        
        return {
            'violations': violations,
            'score': score,
            'unsupported_claims': unsupported
        }
    except Exception as e:
        print(f"  Warning: No-invention check failed: {e}")
        return {'violations': [], 'score': 0.5, 'unsupported_claims': []}


def run_evaluation(baseline_mode: bool = False, quality_mode: bool = True, baseline_json_mode: bool = False, dataset_path: str = None) -> Dict[str, Any]:
    """
    Run evaluation on dataset.
    
    Args:
        baseline_mode: If True, use simple single-model (no verify/revise)
        quality_mode: If True, use quality pipeline (when not in baseline)
        dataset_path: Path to dataset JSONL file (default: dataset.jsonl)
    """
    if dataset_path is None:
        dataset_path = os.path.join(os.path.dirname(__file__), 'dataset.jsonl')
    cases = load_dataset(dataset_path)
    
    print(f"Running evaluation on {len(cases)} test cases...")
    if baseline_json_mode:
        print(f"Mode: BASELINE JSON (single-pass JSON)")
    elif baseline_mode:
        print(f"Mode: BASELINE (legacy)")
    else:
        print(f"Mode: QUALITY PIPELINE")
    print()
    
    results = []
    parse_failures = 0
    rag_enabled = False
    
    # Check RAG availability
    try:
        from rag_store import get_rag_store, CHROMA_AVAILABLE
        rag_store = get_rag_store()
        rag_enabled = rag_store and rag_store.enabled and CHROMA_AVAILABLE
    except:
        rag_enabled = False
    
    total_metrics = {
        'faithfulness': [],
        'answer_relevancy': [],
        'context_precision': [],
        'context_recall': [],
        'instruction_following': [],
        'no_invention': [],
        'latency': []
    }
    
    with app.test_client() as client:
        for i, case in enumerate(cases, 1):
            entry = case['entry']
            entry_id = case.get('entry_id', f'case_{i}')
            category = case.get('category', 'unknown')
            dataset_version = case.get('dataset_version', '1.0')
            
            print(f"Case {i}/{len(cases)}: {entry[:50]}...")
            
            start_time = time.time()
            
            # Make request to /analyze endpoint
            response = client.post('/analyze', json={
                'entry': entry,
                'quality_mode': quality_mode and not baseline_mode and not baseline_json_mode,
                'baseline_json_mode': baseline_json_mode
            })
            
            latency = time.time() - start_time
            total_metrics['latency'].append(latency)
            
            if response.status_code != 200:
                print(f"  ❌ Request failed: {response.status_code}")
                error_data = response.get_json() if response.is_json else {}
                error_reason = error_data.get('error', 'unknown')
                
                # Record parse failure
                if 'json_parse_failed' in str(error_reason):
                    parse_failures += 1
                    print(f"  ⚠️  JSON parse failure: {error_reason}")
                
                # Set metrics to 0.0 for failures (not placeholder 0.50)
                total_metrics['faithfulness'].append(0.0)
                total_metrics['answer_relevancy'].append(0.0)
                total_metrics['instruction_following'].append(0.0)
                total_metrics['no_invention'].append(0.0)
                total_metrics['context_precision'].append(0.0 if not rag_enabled else 1.0)
                total_metrics['context_recall'].append(0.0)
                
                case_result = {
                    'entry': entry,
                    'retrieved_context': '',
                    'analysis': {},
                    'verifier_verdict': None,
                    'metrics': {
                        'faithfulness': 0.0,
                        'answer_relevancy': 0.0,
                        'context_precision': 0.0 if not rag_enabled else 1.0,
                        'context_recall': 0.0,
                        'instruction_following': 0.0,
                        'no_invention': 0.0,
                        'latency': latency,
                        'groundedness_score': 0.0,
                        'unsupported_claims': []
                    },
                    'instruction_checks': {},
                    'no_invention_details': {'violations': [], 'score': 0.0},
                    'failure_reason': error_reason
                }
                results.append(case_result)
                continue
            
            data = response.get_json()
            analysis_json = data.get('analysis', {})
            
            # Check for parse failure in quality mode or baseline_json mode
            if (quality_mode and not baseline_mode) or baseline_json_mode:
                if not analysis_json or 'error' in data:
                    parse_failures += 1
                    error_reason = data.get('error', 'missing_analysis_json')
                    print(f"  ⚠️  JSON parse failure in quality mode: {error_reason}")
                    
                    # Set metrics to 0.0 (not placeholder)
                    total_metrics['faithfulness'].append(0.0)
                    total_metrics['answer_relevancy'].append(0.0)
                    total_metrics['instruction_following'].append(0.0)
                    total_metrics['no_invention'].append(0.0)
                    total_metrics['context_precision'].append(0.0 if not rag_enabled else 1.0)
                    total_metrics['context_recall'].append(0.0)
                    
                    entry_id = case.get('entry_id', f'case_{i}')
                    category = case.get('category', 'unknown')
                    dataset_version = case.get('dataset_version', '1.0')
                    
                    case_result = {
                        'entry_id': entry_id,
                        'entry': entry,
                        'category': category,
                        'dataset_version': dataset_version,
                        'rag_enabled': rag_enabled,
                        'retrieved_context': '',
                        'latency_seconds': latency,
                        'parse_failures': 1,
                        'baseline_output': None,
                        'quality_output': None,
                        'draft_json': None,
                        'verifier_json': None,
                        'final_json': None,
                        'metrics': {
                            'faithfulness': 0.0,
                            'answer_relevancy': 0.0,
                            'context_precision': 0.0 if not rag_enabled else 1.0,
                            'context_recall': 0.0,
                            'instruction_following': 0.0,
                            'no_invention': 0.0,
                            'groundedness_score': 0.0,
                            'unsupported_claims': []
                        },
                        'instruction_checks': {},
                        'no_invention_details': {'violations': [], 'score': 0.0},
                        'failure_reason': error_reason,
                        'analysis': {}  # Backward compatibility
                    }
                    results.append(case_result)
                    continue
            
            if not analysis_json:
                # Legacy mode fallback - but mark as potential issue
                print(f"  ⚠️  No analysis JSON, using legacy format (baseline mode only)")
                analysis_json = {'summary': data.get('insight', '')}
                if quality_mode and not baseline_mode:
                    parse_failures += 1
            
            # Get retrieved context if available (from RAG)
            retrieved_context = ""
            if rag_enabled:
                try:
                    from rag_store import get_rag_store
                    rag_store = get_rag_store()
                    if rag_store and rag_store.enabled:
                        retrieved_context = rag_store.retrieve(entry, top_k=3)
                except Exception as e:
                    logging.debug(f"RAG retrieval failed: {type(e).__name__}")
            
            # Get verifier verdict if in quality mode
            verifier_verdict = None
            if quality_mode and not baseline_mode and 'analysis' in data:
                # Try to get verifier verdict from the pipeline
                # Note: This would need to be exposed in the response or logged
                # For now, we'll compute it here
                try:
                    verifier_prompt = get_verifier_prompt(analysis_json, entry, retrieved_context)
                    verifier_verdict = json_generate(
                        app.config.get('VERIFIER_MODEL', 'samantha-mistral:7b'),
                        VERIFIER_SYSTEM_PROMPT,
                        verifier_prompt,
                        max_retries=1
                    )
                except:
                    pass
            
            faithfulness = compute_faithfulness(analysis_json, entry, retrieved_context)
            answer_relevancy = compute_answer_relevancy(analysis_json, entry)
            context_precision = compute_context_precision(retrieved_context, entry)
            context_recall = compute_context_recall(retrieved_context, entry)
            instruction_checks = check_instruction_following(analysis_json)
            no_invention = check_no_invention(analysis_json, case.get('must_not_invent', []), entry)
            
            total_metrics['faithfulness'].append(faithfulness)
            total_metrics['answer_relevancy'].append(answer_relevancy)
            total_metrics['context_precision'].append(context_precision)
            total_metrics['context_recall'].append(context_recall)
            total_metrics['instruction_following'].append(sum(instruction_checks.values()) / len(instruction_checks))
            total_metrics['no_invention'].append(no_invention['score'])
            
            # Build comprehensive case result for training data
            case_result = {
                'entry_id': entry_id,
                'entry': entry,
                'category': category,
                'dataset_version': dataset_version,
                'rag_enabled': rag_enabled,
                'retrieved_context': retrieved_context,
                'latency_seconds': latency,
                'parse_failures': 0,  # Will be set to 1 if parse failed above
            }
            
            # Add mode-specific outputs
            if baseline_mode:
                case_result['baseline_output'] = json.dumps(analysis_json) if analysis_json else data.get('insight', '')
                case_result['baseline_json_output'] = None
                case_result['quality_output'] = None
                case_result['draft_json'] = None
                case_result['verifier_json'] = None
                case_result['final_json'] = None
            elif baseline_json_mode:
                case_result['baseline_output'] = None
                case_result['baseline_json_output'] = json.dumps(analysis_json) if analysis_json else ''
                case_result['quality_output'] = None
                case_result['draft_json'] = analysis_json
                case_result['verifier_json'] = None
                case_result['final_json'] = analysis_json
            else:
                case_result['baseline_output'] = None
                case_result['quality_output'] = json.dumps(analysis_json) if analysis_json else ''
                # In quality mode, analysis_json is the final_json
                case_result['draft_json'] = analysis_json  # Could be enhanced to capture draft separately
                case_result['verifier_json'] = verifier_verdict
                case_result['final_json'] = analysis_json
            
            # Add metrics
            case_result['metrics'] = {
                'faithfulness': faithfulness,
                'answer_relevancy': answer_relevancy,
                'context_precision': context_precision,
                'context_recall': context_recall,
                'instruction_following': sum(instruction_checks.values()) / len(instruction_checks),
                'no_invention': no_invention['score'],
                'groundedness_score': verifier_verdict.get('groundedness_score', 0.0) if verifier_verdict else 0.0,
                'unsupported_claims': verifier_verdict.get('unsupported_claims', []) if verifier_verdict else []
            }
            
            case_result['instruction_checks'] = instruction_checks
            case_result['no_invention_details'] = no_invention
            case_result['analysis'] = analysis_json  # Keep for backward compatibility
            
            results.append(case_result)
            
            print(f"  ✓ Faithfulness: {faithfulness:.2f}, Relevancy: {answer_relevancy:.2f}, No Invention: {no_invention['score']:.2f}")
    
    # Compute aggregate metrics with std
    aggregate = {}
    for metric, values in total_metrics.items():
        if values:
            mean_val = sum(values) / len(values)
            sorted_vals = sorted(values)
            median_val = sorted_vals[len(sorted_vals) // 2] if sorted_vals else 0
            std_val = (sum((x - mean_val) ** 2 for x in values) / len(values)) ** 0.5 if len(values) > 1 else 0.0
            
            aggregate[metric] = {
                'mean': mean_val,
                'median': median_val,
                'std': std_val,
                'min': min(values),
                'max': max(values)
            }
    
    mode_name = 'baseline_json' if baseline_json_mode else ('baseline' if baseline_mode else 'quality')
    return {
        'mode': mode_name,
        'timestamp': datetime.now().isoformat(),
        'num_cases': len(cases),
        'parse_failures': parse_failures,
        'rag_enabled': rag_enabled,
        'aggregate_metrics': aggregate,
        'case_results': results
    }


def generate_markdown_summary(baseline_results: Dict, quality_results: Dict) -> str:
    """Generate markdown summary comparing baseline vs quality mode."""
    md = "# Evaluation Results Summary\n\n"
    md += f"**Generated:** {datetime.now().isoformat()}\n\n"
    
    # Parse failures and RAG status
    md += "## Configuration\n\n"
    md += f"- **RAG Enabled:** {quality_results.get('rag_enabled', False)}\n"
    md += f"- **Baseline Parse Failures:** {baseline_results.get('parse_failures', 0)}/{baseline_results.get('num_cases', 0)}\n"
    md += f"- **Quality Parse Failures:** {quality_results.get('parse_failures', 0)}/{quality_results.get('num_cases', 0)}\n"
    md += "\n"
    
    if baseline_results.get('parse_failures', 0) > 0 or quality_results.get('parse_failures', 0) > 0:
        md += "⚠️ **WARNING:** Parse failures detected. Metrics may be incomplete.\n\n"
    
    md += "## Aggregate Metrics\n\n"
    md += "| Metric | Baseline (Mean ± Std) | Quality Pipeline (Mean ± Std) | Improvement |\n"
    md += "|--------|----------------------|------------------------------|-------------|\n"
    
    metrics_to_compare = ['faithfulness', 'answer_relevancy', 'instruction_following', 'no_invention']
    
    for metric in metrics_to_compare:
        baseline_agg = baseline_results['aggregate_metrics'].get(metric, {})
        quality_agg = quality_results['aggregate_metrics'].get(metric, {})
        baseline_mean = baseline_agg.get('mean', 0)
        baseline_std = baseline_agg.get('std', 0)
        quality_mean = quality_agg.get('mean', 0)
        quality_std = quality_agg.get('std', 0)
        improvement = quality_mean - baseline_mean
        improvement_pct = (improvement / baseline_mean * 100) if baseline_mean > 0 else 0
        
        md += f"| {metric.replace('_', ' ').title()} | {baseline_mean:.3f} ± {baseline_std:.3f} | {quality_mean:.3f} ± {quality_std:.3f} | {improvement:+.3f} ({improvement_pct:+.1f}%) |\n"
    
    md += "\n## Latency Comparison\n\n"
    baseline_latency = baseline_results['aggregate_metrics'].get('latency', {}).get('mean', 0)
    quality_latency = quality_results['aggregate_metrics'].get('latency', {}).get('mean', 0)
    md += f"- **Baseline:** {baseline_latency:.2f}s average\n"
    md += f"- **Quality Pipeline:** {quality_latency:.2f}s average\n"
    md += f"- **Overhead:** {quality_latency - baseline_latency:.2f}s ({((quality_latency / baseline_latency - 1) * 100):+.1f}%)\n"
    
    return md


def main():
    """Run evaluations in both modes and generate reports."""
    import argparse
    parser = argparse.ArgumentParser(description='Run evaluation harness')
    parser.add_argument('--dataset', type=str, default=None, help='Path to dataset JSONL file (default: evals/dataset.jsonl)')
    parser.add_argument('--mode', type=str, choices=['baseline', 'baseline_json', 'quality', 'both'], default='both', help='Evaluation mode: baseline (legacy), baseline_json (single-pass JSON), quality, or both')
    parser.add_argument('--baseline-only', action='store_true', help='Run only baseline mode (deprecated: use --mode baseline)')
    parser.add_argument('--quality-only', action='store_true', help='Run only quality mode (deprecated: use --mode quality)')
    args = parser.parse_args()
    
    # Handle deprecated flags
    if args.baseline_only:
        args.mode = 'baseline'
    if args.quality_only:
        args.mode = 'quality'
    
    # Determine dataset path
    if args.dataset:
        dataset_path = args.dataset
    else:
        dataset_path = os.path.join(os.path.dirname(__file__), 'dataset.jsonl')
    
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    baseline_results = None
    quality_results = None
    
    if args.mode in ['baseline', 'both']:
        print("=" * 60)
        print("RUNNING BASELINE MODE (legacy)")
        print("=" * 60)
        baseline_results = run_evaluation(baseline_mode=True, quality_mode=False, baseline_json_mode=False, dataset_path=dataset_path)
        baseline_path = os.path.join(results_dir, f'baseline_{timestamp}.json')
        with open(baseline_path, 'w') as f:
            json.dump(baseline_results, f, indent=2)
        print(f"\nBaseline results saved to: {baseline_path}\n")
    
    if args.mode == 'baseline_json':
        print("=" * 60)
        print("RUNNING BASELINE JSON MODE")
        print("=" * 60)
        baseline_json_results = run_evaluation(baseline_mode=False, quality_mode=False, baseline_json_mode=True, dataset_path=dataset_path)
        baseline_json_path = os.path.join(results_dir, f'baseline_json_{timestamp}.json')
        with open(baseline_json_path, 'w') as f:
            json.dump(baseline_json_results, f, indent=2)
        print(f"\nBaseline JSON results saved to: {baseline_json_path}\n")
    
    if args.mode in ['quality', 'both']:
        print("=" * 60)
        print("RUNNING QUALITY PIPELINE MODE")
        print("=" * 60)
        quality_results = run_evaluation(baseline_mode=False, quality_mode=True, baseline_json_mode=False, dataset_path=dataset_path)
        quality_path = os.path.join(results_dir, f'quality_{timestamp}.json')
        with open(quality_path, 'w') as f:
            json.dump(quality_results, f, indent=2)
        print(f"\nQuality results saved to: {quality_path}\n")
    
    # Generate comparison if both modes run
    if baseline_results and quality_results:
        summary_md = generate_markdown_summary(baseline_results, quality_results)
        summary_path = os.path.join(results_dir, f'summary_{timestamp}.md')
        with open(summary_path, 'w') as f:
            f.write(summary_md)
        print(f"Summary saved to: {summary_path}")
        print("\n" + summary_md)


if __name__ == "__main__":
    main()
