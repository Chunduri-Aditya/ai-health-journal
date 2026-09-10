# GateGuard fact: callers agent/respond.py, agent/graph.py, app.py. User: Implement the plan as specified. Do NOT edit the plan file.
"""Crisis detection and support messaging.

Deterministic floor beneath the verifier's judgment. Reflexive, first-person
self-harm phrasing forces the support path even if the verifier misses it or
its call failed, so the reframe/positivity path fails closed.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional

# Two tiers: crisis and distress. Crisis is self-harm, distress is hopelessness/
# worthlessness. Kept tight (first person, reflexive) so idioms like "this job
# is killing me" do not trigger it.
_CRISIS_PATTERNS = re.compile(
    r"\b("
    r"kill(?:ing)?\s+myself|"
    r"end(?:ing)?\s+my\s+life|"
    r"end(?:ing)?\s+it\s+all|"
    r"take\s+my\s+own\s+life|"
    r"want(?:s|ed)?\s+to\s+die|"
    r"wish\s+(?:i\s+was|i\s+were)\s+dead|"
    r"better\s+off\s+(?:if\s+i\s+(?:was|were)\s+)?dead|"
    r"give\s+up\s+on\s+life|"
    r"(?:no\s+longer|don'?t)\s+want(?:s)?\s+to\s+(?:live|be\s+here|be\s+alive|wake\s+up)|"
    r"hurt(?:ing)?\s+myself|harm(?:ing)?\s+myself|"
    r"unalive(?:\s+myself)?|"
    r"self[-\s]?harm|"
    r"suicid(?:e|al)|"
    r"overdos(?:e|ed|ing)\s+on\s+(?:my|the)\s+(?:pills|meds|medication)|"
    r"tak(?:e|ing)\s+all\s+(?:my|the)\s+pills|"
    r"jump(?:ing)?\s+off\s+(?:a|the)\s+bridge"
    r")\b",
    re.IGNORECASE,
)

_HARM_TO_OTHERS_PATTERNS = re.compile(
    r"\b("
    r"i(?:'m| am)\s+(?:going\s+to|planning\s+to)\s+(?:hurt|attack|harm)\s+(?:him|her|them)\b|"
    r"i(?:'ll| will)\s+hurt\s+(?:him|her|them)\s+(?:badly|seriously|for\s+real)"
    r")\b",
    re.IGNORECASE,
)

_ADVERB_GAP = r"(?:just|really|honestly|kind\s+of|sort\s+of)\s+"
_DISTRESS_PATTERNS = re.compile(
    r"\b("
    rf"i(?:'m| am)\s+(?:{_ADVERB_GAP})?(?:such\s+)?(?:a\s+)?(?:failure|worthless|useless|broken|pathetic|a\s+burden)|"
    rf"i\s+(?:{_ADVERB_GAP})?feel\s+like\s+(?:such\s+)?(?:a\s+)?(?:failure|burden|nothing)|"
    rf"i\s+(?:{_ADVERB_GAP})?feel\s+(?:worthless|useless|broken|pathetic)|"
    r"i\s+hate\s+myself|"
    r"i(?:'m| am)\s+not\s+good\s+enough|"
    r"i\s+can'?t\s+do\s+anything\s+right|"
    r"i(?:'ll| will)\s+never\s+get\s+(?:better|past\s+this)|"
    r"nothing\s+(?:i\s+do\s+)?matters|"
    r"what'?s\s+the\s+point|"
    r"no\s+one\s+(?:cares|would\s+notice)|"
    r"i\s+(?:always|constantly)\s+(?:ruin|screw\s+up|mess\s+up)"
    r")\b",
    re.IGNORECASE,
)

_REPORTED_SPEECH = re.compile(
    r"\b(?:he|she|they|someone|everyone|nobody|people|my\s+\w+|his|her|their)\s+"
    r"(?:said|says|told\s+me|calls?\s+me|called\s+me|thinks?|thought)\b[^.!?]{0,25}$",
    re.IGNORECASE,
)

CRISIS_SUPPORT_MESSAGE = os.getenv("AIHJ_CRISIS_MESSAGE") or (
    "It sounds like you're carrying something really heavy right now, and you "
    "don't have to carry it alone. Please consider reaching out to a crisis line "
    "in your area or someone you trust. If you're in immediate danger, contact "
    "local emergency services."
)

DISTRESS_STEADYING_MESSAGE = os.getenv("AIHJ_STEADYING_MESSAGE") or (
    "That sounds genuinely hard, and it makes sense that it's sitting heavily "
    "with you. Nothing below is a verdict on you. It's a reflection of what you "
    "wrote, so take it at whatever pace feels right."
)


def is_crisis(text: str, verdict: Optional[Dict[str, Any]] = None) -> bool:
    """Crisis decision for the reframe gate.

    Fails closed: fires if the verifier judged a crisis, if any safety flag
    names self-harm or harm to others, OR if the raw entry matches the
    reflexive self-harm floor or the (narrower) harm-to-others floor. The
    floors cover the case where the verifier call failed or missed it.
    """
    if verdict and verdict.get("crisis_detected"):
        return True
    if verdict:
        flags = " ".join(verdict.get("safety_flags", [])).lower().replace("_", " ").replace("-", " ")
        crisis_flag_terms = (
            "self harm", "suicid", "harm to self", "danger to self", "self injury",
            "harm ideation", "harm to others", "danger to others", "threat to others",
        )
        if any(term in flags for term in crisis_flag_terms):
            return True
    if _CRISIS_PATTERNS.search(text or ""):
        return True
    return bool(_HARM_TO_OTHERS_PATTERNS.search(text or ""))


def is_distress(text: str) -> bool:
    """Elevated distress that is not crisis: hopelessness, worthlessness, self-blame.

    Entry-text only, unlike is_crisis: this tier exists to acknowledge how the
    user is speaking about themselves, which is visible in the raw text and does
    not need the verifier's judgment. Matches attributed to someone else are
    skipped (see _REPORTED_SPEECH).
    """
    for match in _DISTRESS_PATTERNS.finditer(text or ""):
        if _REPORTED_SPEECH.search((text or "")[: match.start()]):
            continue
        return True
    return False
