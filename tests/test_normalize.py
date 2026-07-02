"""Tests for arabic_rag_kit.normalize."""

from arabic_rag_kit import Normalizer, NormalizerConfig, normalize
from arabic_rag_kit.normalize import (
    convert_digits,
    normalize_alef,
    remove_diacritics,
    strip_control_chars,
)


def test_remove_diacritics():
    # "العَرَبِيَّة" with full tashkeel -> bare letters.
    text = "الْعَرَبِيَّةُ"
    assert remove_diacritics(text) == "العربية"


def test_remove_superscript_alef():
    # U+0670 superscript alef (dagger alef) in "هَٰذَا" style.
    text = "هذاٰ"
    assert "ٰ" not in remove_diacritics(text)


def test_remove_tatweel():
    assert normalize("كتـــــاب") == "كتاب"


def test_normalize_alef_variants():
    assert normalize_alef("أإآٱ") == "اااا"
    assert normalize("أحمد إلى آخر ٱسم") == "احمد الى اخر اسم"


def test_hamza_off_by_default_on_when_requested():
    text = "مؤمن شئ"
    # Off by default: carriers preserved.
    assert normalize(text) == "مؤمن شئ"
    # On: ؤ -> و, ئ -> ي.
    assert normalize(text, normalize_hamza=True) == "مومن شي"


def test_ta_marbuta_off_by_default_on_when_requested():
    text = "مدرسة"
    assert normalize(text) == "مدرسة"
    assert normalize(text, normalize_ta_marbuta=True) == "مدرسه"


def test_alef_maqsura_off_by_default_on_when_requested():
    text = "على"
    assert normalize(text) == "على"
    assert normalize(text, normalize_alef_maqsura=True) == "علي"


def test_convert_arabic_indic_digits():
    assert convert_digits("٠١٢٣٤٥٦٧٨٩") == "0123456789"


def test_convert_eastern_arabic_indic_digits():
    assert convert_digits("۰۱۲۳۴۵۶۷۸۹") == "0123456789"


def test_digits_in_context():
    assert normalize("الصفحة ١٢٣") == "الصفحة 123"


def test_strip_zero_width_and_bidi():
    text = "أ​ب‎﻿ج‮"
    cleaned = strip_control_chars(text)
    for ch in ("​", "‎", "﻿", "‮"):
        assert ch not in cleaned
    assert cleaned == "أبج"


def test_collapse_whitespace():
    assert normalize("  كلمة   \t\n  أخرى  ") == "كلمة اخرى"


def test_mixed_arabic_english_preserved():
    text = "مرحبا Hello ٤٢ World"
    assert normalize(text) == "مرحبا Hello 42 World"


def test_english_only_untouched_except_whitespace():
    assert normalize("The Quick  Brown Fox") == "The Quick Brown Fox"


def test_empty_and_whitespace_input():
    assert normalize("") == ""
    assert normalize("    ") == ""


def test_idempotency():
    text = "الْعَرَبِيَّةُ ﻿ ١٢٣ كتـــاب أحمد"
    once = normalize(text)
    twice = normalize(once)
    assert once == twice


def test_normalizer_class_reuse():
    norm = Normalizer(NormalizerConfig(normalize_hamza=True, normalize_ta_marbuta=True))
    assert norm("مؤسسة") == "موسسه"
    # Same instance is reusable and stateless.
    assert norm("شئ") == "شي"


def test_normalizer_callable_shorthand():
    norm = Normalizer()
    assert norm("كتـــاب") == norm.normalize("كتـــاب")


def test_config_toggles_everything_off():
    cfg = NormalizerConfig(
        remove_diacritics=False,
        remove_tatweel=False,
        normalize_alef=False,
        convert_digits=False,
        strip_control_chars=False,
        collapse_whitespace=False,
    )
    text = "أحمـد  ١٢٣"
    assert Normalizer(cfg).normalize(text) == text


def test_default_config_values():
    cfg = NormalizerConfig()
    assert cfg.remove_diacritics is True
    assert cfg.remove_tatweel is True
    assert cfg.normalize_alef is True
    assert cfg.convert_digits is True
    assert cfg.strip_control_chars is True
    assert cfg.collapse_whitespace is True
    # Meaning-changing folds default off.
    assert cfg.normalize_hamza is False
    assert cfg.normalize_ta_marbuta is False
    assert cfg.normalize_alef_maqsura is False
