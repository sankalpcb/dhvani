import pytest
from dhvani.signals import (
    script_of, script_mix_entropy, romanization_smell, code_mixing_index,
)


def test_script_of_identifies_blocks():
    assert script_of("अ") == "DEVANAGARI"
    assert script_of("ം") == "MALAYALAM"
    assert script_of("ಅ") == "KANNADA"
    assert script_of("a") == "LATIN"
    assert script_of(" ") is None
    assert script_of("1") is None


def test_monoscript_text_has_zero_entropy():
    assert script_mix_entropy("मैंने उस को कर दिया") == 0.0
    assert script_mix_entropy("i fixed the bug") == 0.0


def test_empty_text_has_zero_entropy():
    assert script_mix_entropy("") == 0.0
    assert script_mix_entropy("123 !!!") == 0.0


def test_balanced_two_script_mix_has_max_entropy():
    # Four Devanagari letters, four Latin letters.
    assert script_mix_entropy("अआइई abcd") == pytest.approx(1.0, abs=0.01)


def test_skewed_mix_has_intermediate_entropy():
    h = script_mix_entropy("मैंने उस को कर दिया bug")
    assert 0.0 < h < 1.0


def test_romanization_smell_flags_non_words():
    assert romanization_smell("the deployment is pending") == 0.0
    assert romanization_smell("thh dplymnt zz xqk") > 0.5


def test_romanization_smell_accepts_y_vowel_words():
    # Words with 'y' as a vowel should not be flagged as suspicious.
    assert romanization_smell("why do you fly my sky") == 0.0
    # Verify the feature still works on genuinely suspicious text.
    assert romanization_smell("thh dplymnt zz xqk") > 0.5


# --- Fix round 2, minor 5: 'y' was both a vowel and a consonant ---

@pytest.mark.parametrize("word", ["rhythm", "myths", "lynch", "syzygy"])
def test_common_y_vowel_words_are_not_flagged(word):
    """_VOWELS counted 'y' as a vowel while the 4+ consonant run regex
    ALSO counted it as a consonant, so these ordinary English words all
    scored a full 1.0. romanization_smell carries 0.15 of the risk weight,
    so that pushed plain English text toward marked/review."""
    assert romanization_smell(word) == 0.0


def test_the_two_definitions_of_y_agree():
    """The consonant character class must be exactly the complement of
    _VOWELS. A letter in both sets is the bug this pins."""
    from dhvani.signals import _CONSONANTS, _VOWELS
    import string

    assert _VOWELS & _CONSONANTS == set()
    assert _VOWELS | _CONSONANTS == set(string.ascii_lowercase)


def test_genuinely_suspicious_text_is_still_flagged():
    """The fix must not blunt the signal it exists to provide."""
    assert romanization_smell("thh dplymnt zz xqk") > 0.5
    assert romanization_smell("ktrp shhh brnnt zzz") == 1.0


def test_remaining_false_positives_are_the_ones_documented():
    """The docstring names the false positives that actually survive:
    English words with four or more consecutive NON-y consonants. Pin them
    so the docstring stays honest."""
    assert romanization_smell("strengths") == 1.0
    assert romanization_smell("twelfths") == 1.0
    assert romanization_smell("angsts") == 1.0


def test_code_mixing_index_is_zero_for_monolingual():
    assert code_mixing_index("i fixed the bug today") == 0.0


def test_code_mixing_index_rises_with_mixing():
    low = code_mixing_index("मैंने उस को कर दिया था वहाँ bug")
    high = code_mixing_index("मैंने bug को fix किया")
    assert 0.0 < low < high <= 100.0
