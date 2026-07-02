"""Tests for arabic_rag_kit.chunk."""

import pytest

from arabic_rag_kit import Chunk, chunk_text, split_sentences

# --------------------------------------------------------------------------- #
# split_sentences
# --------------------------------------------------------------------------- #

def test_split_on_arabic_question_mark():
    sents = split_sentences("كيف حالك؟ أنا بخير.")
    assert sents == ["كيف حالك؟", "أنا بخير."]


def test_split_on_arabic_punctuation_variants():
    text = "أولا؛ ثانيا، ثالثا؟ انتهى۔"
    sents = split_sentences(text)
    assert sents == ["أولا؛", "ثانيا،", "ثالثا؟", "انتهى۔"]


def test_split_on_newlines():
    sents = split_sentences("سطر أول\nسطر ثان\n\nسطر ثالث")
    assert sents == ["سطر أول", "سطر ثان", "سطر ثالث"]


def test_split_latin_punctuation():
    assert split_sentences("Hello world! How are you?") == [
        "Hello world!",
        "How are you?",
    ]


def test_decimal_not_split():
    sents = split_sentences("القيمة هي 3.14 تقريبا.")
    assert sents == ["القيمة هي 3.14 تقريبا."]


def test_abbreviation_not_split():
    assert split_sentences("Dr. Ahmed arrived.") == ["Dr. Ahmed arrived."]
    assert split_sentences("Use e.g. this one.") == ["Use e.g. this one."]


def test_consecutive_terminators_absorbed():
    assert split_sentences("ماذا؟! لا أعرف...") == ["ماذا؟!", "لا أعرف..."]


def test_split_empty_input():
    assert split_sentences("") == []
    assert split_sentences("   \n  ") == []


# --------------------------------------------------------------------------- #
# chunk_text
# --------------------------------------------------------------------------- #

def _arabic_paragraph(n_sentences: int) -> str:
    base = "هذه جملة عربية رقم {} تحتوي على بعض الكلمات المفيدة."
    return " ".join(base.format(i) for i in range(n_sentences))


def test_returns_chunk_objects():
    chunks = chunk_text("جملة قصيرة.", chunk_size=100, chunk_overlap=10)
    assert len(chunks) == 1
    assert isinstance(chunks[0], Chunk)
    assert chunks[0].index == 0
    assert chunks[0].text == "جملة قصيرة."


def test_chunk_size_respected():
    text = _arabic_paragraph(40)
    chunks = chunk_text(text, chunk_size=120, chunk_overlap=30)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text) <= 120


def test_offsets_match_source_text():
    text = _arabic_paragraph(30)
    chunks = chunk_text(text, chunk_size=150, chunk_overlap=40)
    for c in chunks:
        assert text[c.start_char:c.end_char] == c.text


def test_indices_are_sequential():
    text = _arabic_paragraph(30)
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_overlap_produces_shared_content():
    text = _arabic_paragraph(30)
    chunks = chunk_text(text, chunk_size=150, chunk_overlap=50)
    assert len(chunks) > 1
    # Consecutive chunks overlap in the source: next starts before prev ends.
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        assert nxt.start_char < prev.end_char


def test_zero_overlap_no_shared_content():
    text = _arabic_paragraph(30)
    chunks = chunk_text(text, chunk_size=150, chunk_overlap=0)
    assert len(chunks) > 1
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        assert nxt.start_char >= prev.end_char


def test_prefers_sentence_boundaries():
    # Two sentences that together exceed chunk_size should split between them.
    a = "الجملة الأولى هنا."
    b = "الجملة الثانية هنا."
    text = a + " " + b
    chunks = chunk_text(text, chunk_size=len(a) + 2, chunk_overlap=0)
    assert chunks[0].text == a


def test_unbreakable_long_token_is_sliced():
    token = "x" * 250
    chunks = chunk_text(token, chunk_size=100, chunk_overlap=0)
    assert len(chunks) == 3
    for c in chunks:
        assert len(c.text) <= 100
    assert "".join(c.text for c in chunks) == token


def test_normalize_flag_applied():
    text = "الْعَرَبِيَّةُ ﻿ ١٢٣."
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=0, normalize=True)
    assert chunks[0].text == "العربية 123."


def test_empty_and_whitespace_input():
    assert chunk_text("") == []
    assert chunk_text("    \n  ") == []


def test_full_coverage_of_text():
    # Every character of the source (ignoring trimmed whitespace) is covered.
    text = _arabic_paragraph(20)
    chunks = chunk_text(text, chunk_size=140, chunk_overlap=0)
    reconstructed = ""
    cursor = 0
    for c in chunks:
        assert c.start_char >= cursor
        reconstructed += text[cursor:c.end_char] if c.start_char <= cursor else ""
        cursor = c.end_char
    # The last chunk reaches the end of the meaningful text.
    assert chunks[-1].end_char == len(text.rstrip())


@pytest.mark.parametrize(
    "size,overlap",
    [(0, 0), (-1, 0), (100, -5), (100, 100), (100, 150)],
)
def test_invalid_parameters_raise(size, overlap):
    with pytest.raises(ValueError):
        chunk_text("بعض النص", chunk_size=size, chunk_overlap=overlap)
