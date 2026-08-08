#!/usr/bin/env python3
"""Validate valence.py against GoEmotions, a third-party human-annotated dataset.

Why this exists
----------------
docs/IMPROVEMENTS.md section 9 flagged the load-bearing weakness in valence.py's
own validation: every exemplar it was checked against was authored by the same
person who wrote the detector. That is calibration bias, not evidence. This
script closes the gap with an INDEPENDENT gold standard: GoEmotions, ~58k Reddit
comments hand-labeled for 27 emotions + neutral by Google Research, released
under Apache 2.0 (https://github.com/google-research/google-research/tree/master/goemotions,
paper: Demszky et al. 2020, "GoEmotions: A Dataset of Fine-Grained Emotions").

Neither valence.py nor its exemplar file were touched to produce this number.
The mapping from GoEmotions' 27 emotions to positive/negative/neutral (below)
was fixed BEFORE running the script once, and is not iterated against the score.

Why GoEmotions and not a mental-health-specific corpus
--------------------------------------------------------
A user explicitly asked for real Reddit stories to be scraped in as training
and test data. That request was declined: personal mental-health disclosures
becoming permanent, potentially identifiable fixtures in a public eval suite is
a real privacy and dignity problem regardless of the source post being public,
and it would also violate Reddit's terms on bulk reuse for training. Dreaddit
(a Reddit stress dataset) was considered and rejected too -- its access terms
could not be confirmed to carry any real license, and its content is longer,
more personal stress narratives, a meaningfully different risk profile.
GoEmotions is short, mostly mundane Reddit COMMENTS (not personal essays),
officially released by a corporate research lab specifically for reuse, under
a real permissive license, and the simplified split used here strips author
and subreddit identity entirely -- text, emotion label, and an opaque comment
id only. It validates a general-purpose sentiment lexicon, which is what
valence.py actually is.

Data handling
-------------
The dataset itself is NOT committed to this repo (kept in .runtime/, which is
gitignored, same treatment as every other third-party or generated artifact
here). This script downloads it on first run if missing.

Run:
    PYTHONPATH=. python evals/valence_external_validation.py
"""

from __future__ import annotations

import argparse
import csv
import sys
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import valence  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / ".runtime" / "third_party"
EMOTIONS_TXT = DATA_DIR / "emotions.txt"

_BASE_URL = (
    "https://raw.githubusercontent.com/google-research/google-research/"
    "master/goemotions/data"
)

# Fixed BEFORE running this script against valence.py, and not revisited based
# on the score it produces. Four emotions are deliberately excluded rather than
# forced into a bucket: confusion, curiosity, and realization are epistemic
# states without a consistent valence direction ("curiosity" about something
# dreadful is not positive), and surprise is valence-neutral by definition
# (pleasant and unpleasant surprise are both "surprise"). Forcing them into a
# bucket would inject label noise into the gold standard itself.
POSITIVE_EMOTIONS = {
    "admiration", "amusement", "approval", "caring", "desire", "excitement",
    "gratitude", "joy", "love", "optimism", "pride", "relief",
}
NEGATIVE_EMOTIONS = {
    "anger", "annoyance", "disappointment", "disapproval", "disgust",
    "embarrassment", "fear", "grief", "nervousness", "remorse", "sadness",
}
NEUTRAL_EMOTIONS = {"neutral"}
EXCLUDED_EMOTIONS = {"confusion", "curiosity", "realization", "surprise"}


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  fetching {url}")
    urllib.request.urlretrieve(url, dest)  # noqa: S310 - fixed, hardcoded HTTPS URL


def ensure_data(split_path: Path, split_name: str) -> None:
    if not EMOTIONS_TXT.exists():
        _download(f"{_BASE_URL}/emotions.txt", EMOTIONS_TXT)
    if not split_path.exists():
        _download(f"{_BASE_URL}/{split_name}.tsv", split_path)


def load_emotion_index() -> List[str]:
    return EMOTIONS_TXT.read_text(encoding="utf-8").splitlines()


def gold_bucket(emotion_names: List[str]) -> Optional[str]:
    """Map a comment's emotion label(s) to positive/negative/neutral, or None
    to exclude it from the gold standard (mixed-valence or epistemic-only)."""
    names = set(emotion_names)
    if names & EXCLUDED_EMOTIONS and not (names - EXCLUDED_EMOTIONS):
        return None  # only epistemic/ambiguous labels present
    names -= EXCLUDED_EMOTIONS
    if not names:
        return None

    has_pos = bool(names & POSITIVE_EMOTIONS)
    has_neg = bool(names & NEGATIVE_EMOTIONS)
    has_neu = bool(names & NEUTRAL_EMOTIONS)

    # Genuinely mixed-label comments (e.g. both "joy" and "sadness") are
    # excluded from the strict gold standard rather than forced to a side --
    # valence.py's own documented behaviour is to collapse mixed text to
    # neutral (see tests/test_valence.py), so scoring it against a forced
    # single label would penalise a documented, deliberate design choice.
    # These are reported separately below instead.
    if sum([has_pos, has_neg, has_neu]) > 1:
        return "mixed"
    if has_pos:
        return "positive"
    if has_neg:
        return "negative"
    if has_neu:
        return "neutral"
    return None


def load_rows(split_path: Path) -> List[Tuple[str, Optional[str]]]:
    """Return (comment_text, gold_bucket) pairs. gold_bucket may be 'mixed'."""
    emotion_names = load_emotion_index()
    rows: List[Tuple[str, Optional[str]]] = []
    with split_path.open(encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue
            text, label_ids = row[0], row[1]
            try:
                names = [emotion_names[int(i)] for i in label_ids.split(",")]
            except (ValueError, IndexError):
                continue
            rows.append((text, gold_bucket(names)))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        choices=["test", "dev", "train"],
        default="test",
        help="GoEmotions split to validate against (default: test)",
    )
    args = parser.parse_args()

    split_path = DATA_DIR / f"goemotions_{args.split}.tsv"
    print(f"=== valence.py vs GoEmotions {args.split}.tsv (independent, human-annotated) ===\n")
    ensure_data(split_path, args.split)
    rows = load_rows(split_path)

    scored = [(t, g) for t, g in rows if g in ("positive", "negative", "neutral")]
    mixed = [(t, g) for t, g in rows if g == "mixed"]
    excluded = len(rows) - len(scored) - len(mixed)

    print(f"  total comments: {len(rows)}")
    print(f"  scored (single-valence gold label): {len(scored)}")
    print(f"  excluded (epistemic-only label, e.g. curiosity/surprise): {excluded}")
    print(f"  mixed-valence (reported separately, not in headline score): {len(mixed)}\n")

    confusion: Counter = Counter()
    misses: Dict[str, List[str]] = {"positive": [], "negative": [], "neutral": []}

    for text, gold in scored:
        pred = valence.classify(text)
        confusion[(gold, pred)] += 1
        if pred != gold and len(misses[gold]) < 6:
            misses[gold].append(text)

    def rate(gold: str, pred: str) -> int:
        return confusion[(gold, pred)]

    total_by_gold = {
        g: sum(rate(g, p) for p in ("positive", "negative", "neutral"))
        for g in ("positive", "negative", "neutral")
    }

    print(f"{'gold':<10} {'n':>6} {'correct':>9} {'accuracy':>10}")
    print("-" * 38)
    overall_correct = 0
    overall_n = 0
    for gold in ("positive", "negative", "neutral"):
        n = total_by_gold[gold]
        correct = rate(gold, gold)
        overall_correct += correct
        overall_n += n
        acc = correct / n if n else 0.0
        print(f"{gold:<10} {n:>6} {correct:>9} {acc:>9.3f}")
    print("-" * 38)
    print(f"{'overall':<10} {overall_n:>6} {overall_correct:>9} {overall_correct/overall_n:>9.3f}")

    print("\n--- confusion matrix (rows=gold, cols=predicted) ---")
    header = f"{'':<10}" + "".join(f"{p:>10}" for p in ("positive", "negative", "neutral"))
    print(header)
    for gold in ("positive", "negative", "neutral"):
        row = f"{gold:<10}" + "".join(f"{rate(gold, p):>10}" for p in ("positive", "negative", "neutral"))
        print(row)

    print("\n--- sample misclassifications (up to 6 per gold class) ---")
    for gold, examples in misses.items():
        if not examples:
            continue
        print(f"\n  gold={gold}:")
        for text in examples:
            print(f"    valence.classify -> {valence.classify(text):<9} {text[:78]!r}")

    if mixed:
        collapsed_to_neutral = sum(1 for t, _ in mixed if valence.classify(t) == "neutral")
        print(
            f"\n--- mixed-valence comments (n={len(mixed)}): "
            f"{collapsed_to_neutral}/{len(mixed)} collapsed to neutral, as documented ---"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
