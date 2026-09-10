"""Deterministic emotional valence scoring for journal text.

Why this exists
---------------
Measured failure (see docs/IMPROVEMENTS.md section 7): the dense embedder encodes
topic but not emotional valence. "a good day at work where I felt appreciated"
retrieves the user's *worst* work entries, because those entries are about work.
In a journaling assistant the retrieved entries become the grounding context for
what the user reads back, so on a good day the system grounds its reflection in
the person's hardest writing.

Why a lexicon rather than the model
-----------------------------------
Retrieval happens BEFORE any LLM call in the /analyze flow, so query-side valence
must be computable without one. Scoring the stored entry with a model and the
query with a lexicon would make the two incomparable, so both sides use this
single deterministic function. This also follows the repo's standing rule that a
mechanically checkable property belongs in code, not in model judgment.

Known limitations, stated rather than implied
---------------------------------------------
- A lexicon cannot read sarcasm, irony, or mixed valence. "Excited about the new
  city but grieving this one" is genuinely both, and this returns near zero.
- Intensity is not modelled. "mildly annoyed" and "utterly devastated" both score
  as negative.
- Negation handling is a fixed backward window, so it catches "not good" and
  misses "I would not say that today was good".
- CALIBRATION CAVEAT: this lexicon was authored by someone who had seen
  evals/rag_retrieval_cases_v2.json. General affect vocabulary was used rather
  than terms lifted from those documents, but the overlap is not zero, so
  measured gains on that corpus are optimistic. A held-out set is required
  before treating these numbers as an unbiased estimate.
"""

from __future__ import annotations

import re
from typing import List, Literal

Polarity = Literal["positive", "negative", "neutral"]

# Neutral band. |score| below this is treated as "no clear valence" and callers
# should fall through to valence-agnostic behaviour rather than guess.
NEUTRAL_BAND = 0.15

# Words that flip the polarity of a term appearing shortly after them.
_NEGATORS = frozenset(
    """
    not no never none cannot cant dont didnt doesnt wasnt werent isnt arent
    wont wouldnt couldnt shouldnt hardly barely without nothing nobody
    """.split()
)

# How many tokens back to look for a negator.
_NEGATION_WINDOW = 3

_POSITIVE = frozenset(
    """
    good great better best wonderful lovely nice pleasant happy happier joy
    joyful glad delighted grateful thankful thanks thanked thank appreciate
    appreciated proud hope hopeful optimistic relief relieved calm calmer
    peaceful settled steady close connected supported seen heard understood
    loved love warm warmth kind gentle safe comfort comforted content
    energy energized refreshed rested restored clear focused capable strong
    accomplished achieved progress win won success successful celebrate
    excited enjoy enjoyed laughed smiled smile lighter easier smooth
    healing healed forward breakthrough gratitude blessing cherish
    congrats congratulations
    """.split()
)

_NEGATIVE = frozenset(
    """
    bad worse worst awful terrible horrible miserable unhappy sad sadness
    depressed down low empty numb hopeless helpless worthless useless
    failure failed failing inadequate pathetic broken damaged
    exhausted exhausting drained draining tired weary burnout sluggish
    tense tension stress stressed stressful pressure overwhelmed overwhelming
    anxious anxiety dread dreading panic panicked afraid scared fear fearful
    worried worry worrying nervous restless insomnia foggy racing
    guilt guilty shame ashamed regret regretted embarrassed humiliated
    angry anger furious frustrated irritated resentful bitter
    argued argue argument fight fought fighting conflict clash
    rejected rejection dismissed ignored invisible disposable abandoned
    lonely alone isolated disconnected unheard unseen unappreciated
    hurt hurting pain painful ache aching suffering crying cried tears
    struggling struggle difficult hard grief grieving loss lost
    mistake mistakes catastrophizing spiralling spiraling ruminating
    hate hates hated hating loathe despise disgust
    """.split()
)

# Self-directed distress phrasing. Sourced from _DISTRESS_PATTERNS in app.py so
# the two classifiers agree: an entry the app routes to its distress tier must
# not read as valence-neutral here, or retrieval could ground a distressed entry
# in the user's cheerful ones. Added after tests/test_valence.py caught exactly
# that disagreement.
_DISTRESS_PHRASES = (
    "nothing matters",
    "what's the point",
    "whats the point",
    "no point",
    "hate myself",
    "not good enough",
    "anything right",
    "give up",
    "a burden",
    "no one cares",
)

# Multiword expressions carry valence that unigrams miss or invert. Checked
# against the raw lowercased text before tokenisation.
_POSITIVE_PHRASES = (
    "paid off",
    "went well",
    "turned out fine",
    "turned out okay",
    "for nothing",  # as in "all that dread for nothing" -- resolution, not loss
    "felt close",
    "felt seen",
    "felt lighter",
    "meant a lot",
    "on top of",
    "talked things through",
    "clear headed",
    "coming back",
    "first time in",
    "slept well",
    "good tired",
)

_NEGATIVE_PHRASES = (
    "burned out",
    "burnt out",
    "keep up",
    "cannot keep",
    "do not work",
    "does not work",
    "not speaking",
    "took credit",
    "talking over",
    "cancelled on me",
    "let down",
    "hung up",
    "stayed late",
    "barely slept",
    "mind racing",
    "waking up exhausted",
    "freeze up",
    "said nothing",
    "not feel heard",
    "same thing again",
)

# Emoticons are invisible to _TOKEN (it only keeps letters and apostrophes), so
# a visible, common sentiment marker was silently discarded before scoring ever
# ran. Found via an independent validation against GoEmotions (a third-party,
# human-annotated Reddit dataset, see evals/valence_external_validation.py):
# "I'm really sorry about your situation :(" scored positive because "sorry"
# is genuinely ambiguous outside a first-person frame and ":(" -- the clearest
# signal in the sentence -- was never looked at. Matched on the raw text before
# tokenisation, same treatment as the multiword phrase lists above. Not
# exhaustive: covers the common ASCII forms, not the full space of emoji or
# regional variants.
_EMOTICON_BOUNDARY = r"(?=[\s.,!?]|$)"
_POSITIVE_EMOTICONS = re.compile(
    r"(?:^|\s)(?::-?\)+|:-?d\b|=\)+|\(+:|\^_?\^)" + _EMOTICON_BOUNDARY, re.IGNORECASE
)
_NEGATIVE_EMOTICONS = re.compile(
    r"(?:^|\s)(?::-?\(+|:'\(+|:-?/|:-?\\|=\(+|\)+-?:)" + _EMOTICON_BOUNDARY, re.IGNORECASE
)

_TOKEN = re.compile(r"[a-z']+")


def _normalise(token: str) -> str:
    """Strip apostrophes so "don't" matches the negator list entry "dont"."""
    return token.replace("'", "")


def score(text: str) -> float:
    """Return a valence score in roughly [-1.0, 1.0].

    Positive means the text reads as a good experience, negative as a hard one,
    near zero means neutral, mixed, or unrecognised. The magnitude is a
    confidence-ish ratio, not a calibrated probability.
    """
    if not text:
        return 0.0

    lowered = text.lower()
    hits = 0.0
    total = 0.0

    for phrase in _POSITIVE_PHRASES:
        if phrase in lowered:
            hits += 1.0
            total += 1.0
    for phrase in _NEGATIVE_PHRASES + _DISTRESS_PHRASES:
        if phrase in lowered:
            hits -= 1.0
            total += 1.0

    if _POSITIVE_EMOTICONS.search(text):
        hits += 1.0
        total += 1.0
    if _NEGATIVE_EMOTICONS.search(text):
        hits -= 1.0
        total += 1.0

    tokens = [_normalise(t) for t in _TOKEN.findall(lowered)]
    for idx, token in enumerate(tokens):
        if token in _POSITIVE:
            polarity = 1.0
        elif token in _NEGATIVE:
            polarity = -1.0
        else:
            continue
        window = tokens[max(0, idx - _NEGATION_WINDOW) : idx]
        if any(w in _NEGATORS for w in window):
            polarity = -polarity
        hits += polarity
        total += 1.0

    if total == 0.0:
        return 0.0
    return max(-1.0, min(1.0, hits / total))


def classify(text: str) -> Polarity:
    """Bucket `score` into positive / negative / neutral."""
    value = score(text)
    if value > NEUTRAL_BAND:
        return "positive"
    if value < -NEUTRAL_BAND:
        return "negative"
    return "neutral"


def agrees(a: str, b: str) -> bool:
    """True when two texts share a non-neutral polarity.

    Neutral on either side returns False, meaning "no evidence of agreement"
    rather than "disagreement". Callers must treat that as a reason to fall
    through to valence-agnostic behaviour, never as a reason to exclude.
    """
    pa, pb = classify(a), classify(b)
    return pa == pb and pa != "neutral"


def opposes(a: str, b: str) -> bool:
    """True only when two texts carry clearly OPPOSITE non-neutral polarity.

    Distinct from `not agrees(...)`: a neutral or mixed text neither agrees nor
    opposes. That distinction is the whole point, see partition_by_agreement.
    """
    pa, pb = classify(a), classify(b)
    return "neutral" not in (pa, pb) and pa != pb


def partition_by_agreement(query: str, texts: List[str]) -> List[int]:
    """Reorder `texts` by demoting clear valence contradictions to the back.

    DEMOTION ONLY. Agreement never promotes; only clear opposition demotes.
    Relevance is the primary signal and valence is a weak corrective, so a
    promoting variant lets valence override relevance outright: a document the
    dense ranker put first can be pushed below one it put ninth purely on
    sentiment. Two promoting variants were built and measured before this,
    and both are recorded in docs/IMPROVEMENTS.md section 9 rather than deleted:

      - agreeing / not-agreeing (two-way): regressed temporal 1.000 -> 0.500,
        because it demoted neutral text as though neutrality were contradiction.
      - agreeing / unknown / opposing (three-way): fixed that contract violation
        and changed nothing measurable, because promotion still let weakly
        relevant same-valence entries outrank a highly relevant neutral one.

    Mixed valence is not an edge case in journaling. It is what the most
    emotionally significant entries look like ("excited about the new city but
    grieving this one"), and a scalar score collapses them to neutral. Treating
    that collapse as contradiction is precisely backwards, which is why neutral
    and mixed text is left exactly where the ranker put it.

    Parameter free by design: no weight to tune, so this cannot be quietly
    fitted to whatever corpus it was last measured on.
    """
    if classify(query) == "neutral":
        return list(range(len(texts)))
    kept: List[int] = []
    opposing: List[int] = []
    for i, text in enumerate(texts):
        (opposing if opposes(query, text) else kept).append(i)
    return kept + opposing
