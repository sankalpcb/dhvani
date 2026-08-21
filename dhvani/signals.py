"""Pure, cheap text signals used by the risk scorer.

script_mix_entropy targets the Chirp 3 rendering-ambiguity defect: a hypothesis
that flips script repeatedly inside one utterance is the direct fingerprint.
"""

import math
import re
import unicodedata
from collections import Counter

_INDIC_SCRIPTS = (
    "DEVANAGARI", "MALAYALAM", "KANNADA",
    "TAMIL", "TELUGU", "BENGALI", "GUJARATI", "ORIYA", "GURMUKHI",
)

_VOWELS = set("aeiouy")


def script_of(ch: str) -> str | None:
    """Unicode script block of a letter, or None for non-letters."""
    if ch.isascii():
        return "LATIN" if ch.isalpha() else None
    name = unicodedata.name(ch, "")
    for script in _INDIC_SCRIPTS:
        if name.startswith(script):
            return script
    return None


def script_mix_entropy(text: str) -> float:
    """Normalized Shannon entropy over script blocks. 0.0 = single script.

    Normalized by log2(number of observed scripts), so any perfectly balanced
    mix of N scripts saturates at 1.0 regardless of N. This is an accepted
    limitation because the target domain is predominantly two-script mixing
    (Indic language + Latin/English).
    """
    counts = Counter(s for ch in text if (s := script_of(ch)) is not None)
    total = sum(counts.values())
    if total == 0 or len(counts) <= 1:
        return 0.0
    h = -sum((c / total) * math.log2(c / total) for c in counts.values())
    return h / math.log2(len(counts))


def romanization_smell(text: str) -> float:
    """Fraction of Latin tokens that look like neither English nor romanized Indic.

    Heuristic: a plausible word has at least one vowel and no run of four
    or more consonants. Note: the 4+ consonant check still false-positives on
    rare English words like "strengths" and "twelfths"; this is tolerated.
    """
    tokens = [t for t in re.findall(r"[A-Za-z]+", text)]
    if not tokens:
        return 0.0
    bad = 0
    for tok in tokens:
        low = tok.lower()
        if not (_VOWELS & set(low)):
            bad += 1
            continue
        if re.search(r"[bcdfghjklmnpqrstvwxyz]{4,}", low):
            bad += 1
    return bad / len(tokens)


def code_mixing_index(text: str) -> float:
    """Das & Gambaeck CMI: 100 * (1 - max_lang_tokens / non_neutral_tokens).

    Language is approximated by the dominant script of each token.
    """
    langs = []
    for tok in text.split():
        scripts = Counter(s for ch in tok if (s := script_of(ch)) is not None)
        if scripts:
            langs.append(scripts.most_common(1)[0][0])
    if not langs:
        return 0.0
    counts = Counter(langs)
    return 100.0 * (1.0 - counts.most_common(1)[0][1] / len(langs))
