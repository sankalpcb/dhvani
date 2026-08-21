"""Transliteration-optimized WER.

Plain WER treats a benign script flip and a semantic corruption as identical
single substitutions (spec 1.3). toWER maps every script to one writing system
before scoring, so benign variance stops inflating the error rate.

Prior art: Google Research, "Transliteration based approaches to improve
code-switched speech recognition performance". We adopt the metric; we do not
claim it.

Known limitation: this reduces but does not eliminate the penalty for
Latin-vs-Indic renderings of English loanwords, since romanized Malayalam of
"deployment" does not match English spelling. Removing that entirely requires a
phonetic distance or a loanword lexicon — out of scope for Phase 1.
"""

from collections import Counter

import jiwer
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

from dhvani.signals import script_of

_SCHEMES = {
    "DEVANAGARI": sanscript.DEVANAGARI,
    "MALAYALAM": sanscript.MALAYALAM,
    "KANNADA": sanscript.KANNADA,
    "TAMIL": sanscript.TAMIL,
    "TELUGU": sanscript.TELUGU,
    "BENGALI": sanscript.BENGALI,
    "GUJARATI": sanscript.GUJARATI,
    "ORIYA": sanscript.ORIYA,
    "GURMUKHI": sanscript.GURMUKHI,
}


def to_latin(text: str) -> str:
    """Map every Indic-script token to ITRANS, lowercase everything."""
    out = []
    for token in text.split():
        scripts = Counter(
            s for ch in token
            if (s := script_of(ch)) is not None and s != "LATIN"
        )
        if scripts:
            dominant = scripts.most_common(1)[0][0]
            scheme = _SCHEMES.get(dominant)
            if scheme is not None:
                token = transliterate(token, scheme, sanscript.ITRANS)
        out.append(token.lower())
    return " ".join(out)


def plain_wer(reference: str, hypothesis: str) -> float:
    """Standard WER. Empty reference and hypothesis scores 0."""
    if not reference.strip() and not hypothesis.strip():
        return 0.0
    return float(jiwer.wer(reference, hypothesis))


def to_wer(reference: str, hypothesis: str) -> float:
    """WER computed after transliterating both sides to a common script."""
    return plain_wer(to_latin(reference), to_latin(hypothesis))
