"""
Load behavior contract and inject into prompts at runtime.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

_BEHAVIOR_DIR = Path(__file__).parent


def load_behavior_contract() -> Dict[str, Any]:
    """Load the behavior contract (rules, few-shot examples, failure patterns)."""
    contract = {
        "rules": {},
        "few_shot": [],
        "failure_patterns": [],
    }
    
    # Load rules
    rules_path = _BEHAVIOR_DIR / "rules.json"
    if rules_path.exists():
        try:
            with open(rules_path, 'r', encoding='utf-8') as f:
                contract["rules"] = json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    
    # Load few-shot examples
    few_shot_path = _BEHAVIOR_DIR / "few_shot.jsonl"
    if few_shot_path.exists():
        try:
            with open(few_shot_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        contract["few_shot"].append(json.loads(line))
        except (json.JSONDecodeError, IOError):
            pass
    
    # Load failure patterns
    failures_path = _BEHAVIOR_DIR / "failure_patterns.json"
    if failures_path.exists():
        try:
            with open(failures_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                contract["failure_patterns"] = data.get("patterns", [])
        except (json.JSONDecodeError, IOError):
            pass
    
    return contract


def get_behavior_system_prompt_snippet() -> str:
    """Generate a system prompt snippet from behavior rules."""
    contract = load_behavior_contract()
    rules = contract.get("rules", {})
    
    snippets = []
    
    if rules.get("no_invention"):
        snippets.append("Never invent facts, details, or events not present in the user's entry.")
    
    if rules.get("no_diagnosis"):
        snippets.append("Never diagnose medical or mental health conditions. Suggest consulting a professional when appropriate.")
    
    if rules.get("grounding_required"):
        snippets.append("All claims must be grounded in the user's entry or retrieved context. Acknowledge uncertainty when context is insufficient.")
    
    if rules.get("uncertainty_acknowledgment"):
        snippets.append("When you cannot ground a claim, label it as a hypothesis or uncertainty.")
    
    if rules.get("self_harm_protocol"):
        snippets.append("If the user mentions self-harm, provide supportive language and suggest professional help.")
    
    if snippets:
        return "\n".join(f"- {s}" for s in snippets)
    return ""


def get_relevant_few_shot(user_input: str, max_examples: int = 2) -> List[Dict[str, Any]]:
    """
    Return few-shot examples relevant to user input (simple keyword matching).
    
    This is a lightweight approach - no vector DB needed.
    """
    contract = load_behavior_contract()
    few_shot = contract.get("few_shot", [])
    
    if not few_shot:
        return []
    
    # Simple keyword matching (case-insensitive)
    user_lower = user_input.lower()
    user_words = set(user_lower.split())
    
    scored = []
    for ex in few_shot:
        input_text = ex.get("input", "").lower()
        input_words = set(input_text.split())
        
        # Simple overlap score
        overlap = len(user_words & input_words)
        if overlap > 0:
            scored.append((overlap, ex))
    
    # Sort by overlap and return top N
    scored.sort(key=lambda x: x[0], reverse=True)
    return [ex for _, ex in scored[:max_examples]]


def get_failure_lessons() -> List[str]:
    """Return lessons learned from failure patterns."""
    contract = load_behavior_contract()
    patterns = contract.get("failure_patterns", [])
    
    lessons = []
    seen = set()
    for pattern in patterns:
        lesson = pattern.get("lesson", "")
        if lesson and lesson not in seen:
            seen.add(lesson)
            lessons.append(lesson)
    
    return lessons
