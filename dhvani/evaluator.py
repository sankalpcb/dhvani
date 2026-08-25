"""Transliteration-optimized WER.

Plain WER treats a benign script flip and a semantic corruption as identical
single substitutions (spec 1.3). toWER maps every script to one writing system
before scoring, so benign variance stops inflating the error rate.

Prior art: Google Research, "Transliteration based approaches to improve
code-switched speech recognition performance". We adopt the metric; we do not
claim it.

Known limitation: for an English loanword rendered in Indic script, toWER may
not reduce the penalty at all — romanized Malayalam of "deployment" does not
match the Latin spelling "deployment", so a benign script flip and a genuine
semantic corruption of that same loanword can score identically. Native words
that appear in different Indic scripts DO collapse correctly (transliteration
converges on a shared ITRANS spelling); this limitation is specific to
Latin-vs-Indic renderings of the same English loanword. Closing that gap
entirely requires a phonetic distance or a loanword lexicon — out of scope for
Phase 1.
"""

import unicodedata
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


# Candrabindu -> anusvara, per script. The two mark nasalisation and modern
# orthography uses them interchangeably in a great many words, but they
# transliterate differently (ITRANS ".n" against "m"), so toWER scored हाँ
# against हां as a total miss. That is not a corner case: it is how Chirp and
# the IndicVoices references routinely differ.
_NASAL_FOLD = str.maketrans({
    "\u0901": "\u0902",  # Devanagari
    "\u0981": "\u0982",  # Bengali
    "\u0a01": "\u0a02",  # Gurmukhi
    "\u0a81": "\u0a82",  # Gujarati
    "\u0b01": "\u0b02",  # Oriya
    "\u0c00": "\u0c02",  # Telugu (combining)
    "\u0c01": "\u0c02",  # Telugu
    "\u0c80": "\u0c82",  # Kannada
    "\u0d00": "\u0d02",  # Malayalam (combining)
    "\u0d01": "\u0d02",  # Malayalam
})

# Zero-width joiner and non-joiner change rendering, not words. Common in
# Malayalam chillu forms.
_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200c\u200d"))

def _is_punctuation(ch: str) -> bool:
    r"""True for Unicode punctuation only.

    NOT `[^\w\s]`, which was the first attempt and was badly wrong: Python's
    \w excludes category Mn, so that pattern stripped every Devanagari
    combining vowel sign. नौ and नो both collapsed to न -- two different
    words scored identical. Category-based filtering keeps marks (Mn, Mc)
    and removes only P*, which is what "punctuation" means here: the danda
    ।, its double ॥, and ordinary Latin punctuation.
    """
    return unicodedata.category(ch).startswith("P")


def normalize_orthography(text: str) -> str:
    """Fold spelling differences that are not transcription differences.

    Deliberately narrow. Every rule here changes how a word is WRITTEN and
    never which word it is:

      NFC          the same string in two Unicode forms is one string
      nasal fold   हाँ / हां -- candrabindu and anusvara, one word
      zero-width   ZWJ/ZWNJ affect rendering only
      punctuation  a sentence-final danda is not a substitution; it used to
                   transliterate to a literal "|" glued to the last token

    What it deliberately does NOT do is touch vowels or consonants. नौ (nine)
    and नो stay different words, and a test pins that -- normalization that
    collapsed them would be inflating a score rather than measuring one.

    It also does not reconcile numbers. Chirp emits "3456" where the
    references spell digits out, which costs Tier 1 real WER; that affects
    about 10% of the calibration corpus and needs a per-language number
    lexicon, not a character fold.

    Applied by to_wer() rather than by to_latin(), because to_latin's job is
    to map scripts and punctuation is not script.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_NASAL_FOLD).translate(_ZERO_WIDTH)
    return "".join(" " if _is_punctuation(ch) else ch for ch in text)


def plain_wer(reference: str, hypothesis: str) -> float:
    """Standard WER. Empty reference and hypothesis scores 0."""
    if not reference.strip() and not hypothesis.strip():
        return 0.0
    return float(jiwer.wer(reference, hypothesis))


def to_wer(reference: str, hypothesis: str) -> float:
    """WER computed after normalizing orthography and transliterating both
    sides to a common script.

    Normalization runs BEFORE transliteration on purpose: ITRANS output uses
    "." and "~" as meaningful characters, so stripping punctuation afterwards
    would corrupt the transliteration it is meant to clean up.
    """
    return plain_wer(to_latin(normalize_orthography(reference)),
                     to_latin(normalize_orthography(hypothesis)))
