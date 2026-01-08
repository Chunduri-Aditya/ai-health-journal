#!/usr/bin/env python3
"""
Distill eval history into a behavior contract (rules, few-shot examples, failure patterns).

Reads all eval JSONs from artifacts/test_runs/evals/ and produces:
- behavior/rules.json (high-level constraints)
- behavior/few_shot.jsonl (compact examples)
- behavior/failure_patterns.json (known failure modes)
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

ARTIFACTS_DIR = project_root / "artifacts" / "test_runs" / "evals"
BEHAVIOR_DIR = project_root / "behavior"


def load_eval_results() -> List[Dict[str, Any]]:
    """Load all eval JSON files from artifacts directory."""
    results = []
    if not ARTIFACTS_DIR.exists():
        print(f"Warning: {ARTIFACTS_DIR} does not exist. No eval results to distill.")
        return results
    
    for json_file in ARTIFACTS_DIR.glob("*.json"):
        if json_file.stat().st_size < 10:  # Skip 4-byte null files
            continue
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if data and isinstance(data, dict):
                    results.append(data)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Failed to load {json_file}: {e}")
    
    return results


def extract_rules(eval_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract high-level behavioral rules from eval results."""
    rules = {
        "no_invention": True,  # Never make up facts
        "no_diagnosis": True,  # Never diagnose medical/mental health conditions
        "self_harm_protocol": True,  # Handle self-harm mentions with care
        "ask_clarifying_when_missing_context": True,  # Ask when context is unclear
        "consent_gate_for_retrieval": True,  # Require explicit consent for RAG
        "grounding_required": True,  # All claims must be grounded in entry or retrieved context
        "uncertainty_acknowledgment": True,  # Acknowledge uncertainty when appropriate
    }
    
    # Infer from metrics
    for result in eval_results:
        metrics = result.get("aggregate_metrics", {})
        no_invention_mean = metrics.get("no_invention", {}).get("mean", 1.0)
        faithfulness_mean = metrics.get("faithfulness", {}).get("mean", 1.0)
        
        if no_invention_mean < 0.95:
            rules["no_invention"] = True  # Reinforce if we see failures
        if faithfulness_mean < 0.90:
            rules["grounding_required"] = True
    
    return rules


def extract_few_shot_examples(eval_results: List[Dict[str, Any]], max_examples: int = 15) -> List[Dict[str, Any]]:
    """Extract compact few-shot examples from high-scoring cases."""
    examples = []
    
    for result in eval_results:
        case_results = result.get("case_results", [])
        mode = result.get("mode", "unknown")
        
        for case in case_results:
            metrics = case.get("metrics", {})
            faithfulness = metrics.get("faithfulness", 1.0)
            no_invention = metrics.get("no_invention", 1.0)
            entry = case.get("entry", "")
            
            # Only include high-quality examples
            if faithfulness >= 0.9 and no_invention >= 0.95 and entry:
                analysis = case.get("final_json") or case.get("quality_output")
                if isinstance(analysis, str):
                    try:
                        analysis = json.loads(analysis)
                    except json.JSONDecodeError:
                        continue
                
                if isinstance(analysis, dict):
                    # Extract key fields
                    summary = analysis.get("summary", "")[:150]  # Truncate
                    emotions = analysis.get("emotions", [])[:3]
                    coping = analysis.get("coping_suggestions", [])[:2]
                    
                    examples.append({
                        "input": entry[:200],  # Keep short
                        "output": {
                            "summary": summary,
                            "emotions": emotions,
                            "coping_suggestions": coping,
                        },
                        "mode": mode,
                        "scores": {
                            "faithfulness": faithfulness,
                            "no_invention": no_invention,
                        }
                    })
    
    # Deduplicate by input text (simple)
    seen = set()
    unique_examples = []
    for ex in examples:
        input_key = ex["input"][:50]  # First 50 chars as key
        if input_key not in seen:
            seen.add(input_key)
            unique_examples.append(ex)
    
    # Sort by scores and take top N
    unique_examples.sort(key=lambda x: x["scores"]["faithfulness"] + x["scores"]["no_invention"], reverse=True)
    return unique_examples[:max_examples]


def extract_failure_patterns(eval_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract known failure patterns from low-scoring cases."""
    failures = []
    
    for result in eval_results:
        case_results = result.get("case_results", [])
        
        for case in case_results:
            metrics = case.get("metrics", {})
            faithfulness = metrics.get("faithfulness", 1.0)
            no_invention = metrics.get("no_invention", 1.0)
            entry = case.get("entry", "")
            category = case.get("category", "")
            
            # Identify failures
            if faithfulness < 0.8 or no_invention < 0.9:
                failure_type = "low_faithfulness" if faithfulness < 0.8 else "invention_detected"
                
                failures.append({
                    "type": failure_type,
                    "category": category,
                    "entry_preview": entry[:100],
                    "scores": {
                        "faithfulness": faithfulness,
                        "no_invention": no_invention,
                    },
                    "lesson": "Require stronger grounding" if faithfulness < 0.8 else "Avoid making up details",
                })
    
    # Deduplicate by type + category
    seen = set()
    unique_failures = []
    for f in failures:
        key = (f["type"], f["category"])
        if key not in seen:
            seen.add(key)
            unique_failures.append(f)
    
    return unique_failures[:10]  # Keep top 10


def main():
    """Main distillation process."""
    print("Distilling eval results into behavior contract...")
    
    eval_results = load_eval_results()
    if not eval_results:
        print("No eval results found. Run evals first or ensure artifacts/test_runs/evals/ contains JSON files.")
        return 1
    
    print(f"Loaded {len(eval_results)} eval result files.")
    
    # Create behavior directory
    BEHAVIOR_DIR.mkdir(exist_ok=True)
    
    # Extract components
    rules = extract_rules(eval_results)
    few_shot = extract_few_shot_examples(eval_results)
    failures = extract_failure_patterns(eval_results)
    
    # Write rules.json
    rules_path = BEHAVIOR_DIR / "rules.json"
    with open(rules_path, 'w', encoding='utf-8') as f:
        json.dump(rules, f, indent=2)
    print(f"✓ Wrote {len(rules)} rules to {rules_path}")
    
    # Write few_shot.jsonl
    few_shot_path = BEHAVIOR_DIR / "few_shot.jsonl"
    with open(few_shot_path, 'w', encoding='utf-8') as f:
        for ex in few_shot:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')
    print(f"✓ Wrote {len(few_shot)} few-shot examples to {few_shot_path}")
    
    # Write failure_patterns.json
    failures_path = BEHAVIOR_DIR / "failure_patterns.json"
    with open(failures_path, 'w', encoding='utf-8') as f:
        json.dump({"patterns": failures}, f, indent=2)
    print(f"✓ Wrote {len(failures)} failure patterns to {failures_path}")
    
    print("\nBehavior contract distilled successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
