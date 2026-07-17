"""
Prompt templates for verification (LLM-as-judge) and revision.
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


def get_revision_prompt(
    draft_json: dict,
    verdict: dict,
    journal_entry: str,
    retrieved_context: str = "",
) -> str:
    """
    Build a revision prompt from the verifier's verdict.

    Args:
        draft_json: The original draft analysis
        verdict: The verifier's output JSON
        journal_entry: Original journal entry
        retrieved_context: Retrieved context (if any)

    Returns:
        Formatted revision prompt
    """
    groundedness_score = verdict.get("groundedness_score", 0.0)
    unsupported = verdict.get("unsupported_claims", [])
    safety_flags = verdict.get("safety_flags", [])
    instructions = verdict.get(
        "rewrite_instructions",
        "Fix unsupported claims and ensure all information is grounded in the entry.",
    )

    prompt = "Revise this analysis based on the verification feedback below.\n\n"
    prompt += f"ORIGINAL DRAFT:\n{json.dumps(draft_json, indent=2)}\n\n"
    prompt += "VERIFICATION FEEDBACK:\n"
    prompt += f"- Groundedness Score: {groundedness_score}\n"
    if unsupported:
        prompt += f"- Unsupported Claims: {unsupported}\n"
    if safety_flags:
        prompt += f"- Safety Flags: {safety_flags}\n"
    prompt += f"- Instructions: {instructions}\n\n"
    prompt += f"CURRENT ENTRY:\n{journal_entry}\n\n"
    if retrieved_context:
        prompt += f"RETRIEVED CONTEXT:\n{retrieved_context}\n\n"
    prompt += (
        "Return the revised JSON with the same structure as the original draft. "
        "Ensure all claims are grounded in the entry. "
        "Return ONLY valid JSON. No markdown, no commentary, no code fences."
    )
    return prompt
