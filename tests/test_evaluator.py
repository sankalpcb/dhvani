import pytest
from dhvani.evaluator import to_latin, to_wer, plain_wer


def test_identical_text_scores_zero():
    assert to_wer("hello world", "hello world") == 0.0


def test_plain_wer_counts_substitutions():
    assert plain_wer("a b c", "a b d") == pytest.approx(1 / 3)


def test_to_latin_is_idempotent_on_ascii():
    assert to_latin("hello world") == "hello world"


def test_same_word_in_two_indic_scripts_collapses():
    """Devanagari and Malayalam renderings of one word must converge."""
    deva, mala = "कम", "കമ"
    assert plain_wer(deva, mala) == 1.0
    assert to_wer(deva, mala) == 0.0


def test_loanword_script_variance_is_no_worse_than_plain_wer():
    """Non-regression only: toWER must never score *worse* than plain WER.

    This does NOT demonstrate toWER improving on the loanword case — for this
    fixture the two are equal (see
    test_english_loanword_in_indic_script_is_not_neutralized below, which
    pins that equality deliberately). This test just guards the floor: toWER
    must not make things worse than plain WER for a script-flipped loanword.
    """
    ref = "deployment अभी pending है"
    benign = "ഡിപ്ലോയ്‌മെന്റ് अभी pending है"
    assert to_wer(ref, benign) <= plain_wer(ref, benign)


def test_english_loanword_in_indic_script_is_not_neutralized():
    """KNOWN, ACCEPTED limitation — pinned deliberately, not a bug to fix.

    toWER transliterates to ITRANS and scores an exact string match; it has
    no phonetic distance or loanword lexicon (out of scope for Phase 1, see
    module docstring). An English loanword written in Indic script
    transliterates to a phonetic respelling (e.g. "deployment" rendered in
    Malayalam comes back as "diploy<ZWNJ>mènr", not "deployment"), so it does
    NOT match the reference's Latin spelling. toWER therefore gives ZERO
    improvement over plain WER for this case — both score the substitution
    identically.

    This is expected, not a regression. Native Indic words rendered in
    *different* Indic scripts DO collapse correctly (see
    test_same_word_in_two_indic_scripts_collapses) — the gap is specific to
    Latin-vs-Indic renderings of the same English loanword.

    If a future change (phonetic distance, loanword lexicon) makes toWER
    actually improve on this case, update this test deliberately rather than
    let it fail mysteriously. Same spirit as
    test_greedy_is_knowingly_suboptimal_with_heterogeneous_costs in
    tests/test_router.py.
    """
    ref = "deployment अभी pending है"
    benign = "ഡിപ്ലോയ്‌മെന്റ് अभी pending है"
    assert to_wer(ref, benign) == plain_wer(ref, benign)


def test_semantic_corruption_is_still_penalized():
    """The counterpart: destroying meaning must still cost.

    Also pins the *mechanism*, not just the outcome: the second assertion
    below only reaches 0.0 because a genuine cross-script transliteration
    equivalent (the same कम/കമ pair verified in
    test_same_word_in_two_indic_scripts_collapses) was recognized as a match.
    A no-op `to_latin` (e.g. plain `str.lower`) would leave "कम" and "കമ" as
    distinct strings and this second assertion would fail — so this test
    cannot be stub-passed by dropping real transliteration, unlike a bare
    `> 0.0` check on the corrupt case alone.
    """
    ref = "deployment अभी pending है"
    corrupt = "डिब्बा अभी pending है"
    assert to_wer(ref, corrupt) > 0.0

    equivalent_ref = "कम अभी pending है"
    equivalent_hyp = "കമ अभी pending है"
    assert to_wer(equivalent_ref, equivalent_hyp) == 0.0


def test_to_wer_is_bounded_below_by_zero():
    assert to_wer("", "") == 0.0


# --- orthographic normalization: same word, two spellings ---

def test_chandrabindu_and_anusvara_are_the_same_word():
    """हाँ and हां are one word written two ways -- chandrabindu against
    anusvara. toWER scored them as a total miss (ha.n vs ham), and this is
    not a corner case: it is how Chirp and the IndicVoices references
    routinely differ, and it dragged the measured Tier 1 delta down."""
    from dhvani.evaluator import to_wer
    assert to_wer("हाँ हाँ", "हां हां") == 0.0


def test_a_sentence_final_danda_is_not_an_error():
    """। transliterated to a literal '|' glued to the last token, so pure
    punctuation counted as a substitution."""
    from dhvani.evaluator import to_wer
    assert to_wer("योग दर्शन के जो हैं।", "योग दर्शन के जो हैं") == 0.0
    assert to_wer("नमस्ते, दुनिया.", "नमस्ते दुनिया") == 0.0


def test_different_vowels_are_still_different_words():
    """The guard on all of the above. नौ (nine) and नो are genuinely
    different, and normalization that collapsed them would be inflating
    Tier 1's score rather than measuring it."""
    from dhvani.evaluator import to_wer
    assert to_wer("नौ", "नो") > 0.0


def test_normalization_does_not_collapse_real_substitutions():
    from dhvani.evaluator import to_wer
    assert to_wer("योग दर्शन", "योग विज्ञान") > 0.0


def test_unicode_composition_is_not_an_error():
    """The same string in NFD and NFC must score identically; which form a
    provider emits is not a transcription difference."""
    import unicodedata
    from dhvani.evaluator import to_wer
    text = "कुतूहल गाँव में"
    assert to_wer(unicodedata.normalize("NFD", text),
                  unicodedata.normalize("NFC", text)) == 0.0


def test_zero_width_joiners_are_ignored():
    """ZWJ/ZWNJ change rendering, not words. Common in Malayalam chillu."""
    from dhvani.evaluator import to_wer
    assert to_wer("അവന്‍", "അവന്") == 0.0


def test_the_demo_clips_tier0_tier1_disagreement_is_orthographic_only():
    """The committed fixtures differ only in spelling, which is exactly the
    class of difference this normalization exists to stop counting."""
    from dhvani.evaluator import to_wer
    assert to_wer("कुतूहल गाँव में आधे घंटे टहलना समय की हानि नहीं है",
                  "कुतुहल गांव में आधे घंटे टहलना समय की हानि नहीं है") == 0.0
