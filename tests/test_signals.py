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


def test_code_mixing_index_is_zero_for_monolingual():
    assert code_mixing_index("i fixed the bug today") == 0.0


def test_code_mixing_index_rises_with_mixing():
    low = code_mixing_index("मैंने उस को कर दिया था वहाँ bug")
    high = code_mixing_index("मैंने bug को fix किया")
    assert 0.0 < low < high <= 100.0
