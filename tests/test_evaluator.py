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


def test_benign_script_variance_is_penalized_less_than_plain_wer():
    """Spec 1.3: toWER must not treat a script flip like a semantic error."""
    ref = "deployment अभी pending है"
    benign = "ഡിപ്ലോയ്‌മെന്റ് अभी pending है"
    assert to_wer(ref, benign) <= plain_wer(ref, benign)


def test_semantic_corruption_is_still_penalized():
    """The counterpart: destroying meaning must still cost."""
    ref = "deployment अभी pending है"
    corrupt = "डिब्बा अभी pending है"
    assert to_wer(ref, corrupt) > 0.0


def test_to_wer_is_bounded_below_by_zero():
    assert to_wer("", "") == 0.0
