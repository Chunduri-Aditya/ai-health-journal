"""
Prompt templates for verification (LLM-as-judge).
"""

import json

VERIFIER_SYSTEM_PROMPT = """You are a fact-checking verifier for journaling assistant responses. Your job is to ensure responses are grounded in evidence and safe.

Evaluate the draft analysis against the allowed evidence:
- CURRENT_ENTRY: The user's current journal entry
- RETRIEVED_CONTEXT: Any retrieved context from past entries (if provided)

Check for:
1. GROUNDEDNESS: Are all claims supported by evidence?
2. HALLUCINATIONS: Are there invented facts, events, or assumptions?
3. SAFETY: Are there concerning safety flags (self-harm, etc.)?
4. FORMAT: Is the JSON structure correct?

Return a JSON object that matches the provided schema with these exact fields:
- groundedness_score: float (0.0-1.0, 1.0 = fully grounded)
- unsupported_claims: array of strings (specific claims not supported by evidence)
- safety_flags: array of strings (any safety concerns, empty if none)
- rewrite_required: boolean (true if groundedness_score < 0.75 OR unsupported_claims exist OR safety_flags exist)
- rewrite_instructions: string (specific instructions for revision if rewrite_required is true, empty string otherwise)

CRITICAL: Return ONLY valid JSON that matches the schema. Do not add markdown, no commentary, no code fences. The response must be parseable JSON.

Be strict but fair. Flag any invented information."""

def get_verifier_prompt(draft_json: dict, journal_entry: str, retrieved_context: str = "") -> str:
    """
    Build verifier prompt.
    
    Args:
        draft_json: The draft analysis JSON
        journal_entry: Original journal entry
        retrieved_context: Retrieved context (if any)
        
    Returns:
        Formatted user prompt
    """
    prompt = "Verify this draft analysis against the allowed evidence:\n\n"
    prompt += f"DRAFT ANALYSIS:\n{json.dumps(draft_json, indent=2)}\n\n"
    prompt += f"CURRENT_ENTRY:\n{journal_entry}\n\n"
    
    if retrieved_context:
        prompt += f"RETRIEVED_CONTEXT:\n{retrieved_context}\n\n"
    else:
        prompt += "RETRIEVED_CONTEXT: (none provided)\n\n"
    
    prompt += "Evaluate groundedness, check for hallucinations, and identify safety concerns. "
    prompt += "\nReturn ONLY valid JSON that matches the schema. No markdown, no commentary, no code fences."
    
    return prompt
