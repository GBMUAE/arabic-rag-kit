"""RAG-aware text chunking that respects Arabic sentence boundaries.

Pure standard library — the only import is this package's own
:mod:`arabic_rag_kit.normalize` (which is itself dependency-free).

Two public entry points:

* :func:`split_sentences` — split text into sentences on Arabic *and* Latin
  punctuation, without breaking on decimals or common abbreviations.
* :func:`chunk_text` — recursive character chunking that prefers to break on
  sentence boundaries, then on whitespace, and finally mid-token only when a
  single token is larger than ``chunk_size``. Returns :class:`Chunk` objects
  carrying exact character offsets into the chunked text.
"""

from __future__ import annotations

from dataclasses import dataclass

from .normalize import normalize as _normalize

__all__ = ["Chunk", "split_sentences", "chunk_text"]


# Sentence terminators: Arabic question mark (؟), Arabic semicolon (؛),
# Arabic comma (،), Arabic full stop (۔) and the Latin . ! ?
_TERMINATORS = frozenset(".!?؟؛،۔")

# Abbreviations whose trailing period should not end a sentence. Matched
# case-insensitively against the word immediately preceding the period.
_ABBREVIATIONS = frozenset({
    "dr", "mr", "mrs", "ms", "prof", "sr", "jr", "vs", "etc", "no", "st",
    "mt", "fig", "al", "ph", "inc", "ltd", "co", "eg", "ie", "e.g", "i.e",
})


@dataclass(frozen=True)
class Chunk:
    """A single chunk of text with its position in the source.

    Attributes:
        text: The chunk's text.
        index: Zero-based position of this chunk in the returned list.
        start_char: Inclusive start offset into the chunked text.
        end_char: Exclusive end offset into the chunked text.
    """

    text: str
    index: int
    start_char: int
    end_char: int


# --------------------------------------------------------------------------- #
# Sentence splitting.
# --------------------------------------------------------------------------- #

def _preceding_word(text: str, i: int) -> str:
    """Return the run of alphanumerics ending just before index ``i``."""
    j = i - 1
    chars: list[str] = []
    # Include internal dots so dotted abbreviations ("e.g", "i.e") are matched.
    while j >= 0 and (text[j].isalnum() or text[j] == "."):
        chars.append(text[j])
        j -= 1
    return "".join(reversed(chars)).strip(".")


def _is_period_boundary(text: str, i: int) -> bool:
    """Decide whether the ``.`` at ``text[i]`` really ends a sentence."""
    n = len(text)
    prev = text[i - 1] if i > 0 else ""
    nxt = text[i + 1] if i + 1 < n else ""

    # Decimal number: "3.14" — a period between two digits is not a boundary.
    if prev.isdigit() and nxt.isdigit():
        return False
    # Inline abbreviation / initialism: a period immediately followed by a
    # lowercase ASCII letter, e.g. "e.g." or "i.e." — not a boundary.
    if "a" <= nxt <= "z":
        return False
    # Known abbreviation ("Dr. Ahmed", "etc. ") followed by space + capital.
    word = _preceding_word(text, i)
    if word and word.lower() in _ABBREVIATIONS:
        return False
    return True


def _trim_span(text: str, start: int, end: int) -> tuple[int, int] | None:
    """Strip surrounding whitespace from ``[start, end)``; ``None`` if empty."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if end > start else None


def _iter_sentence_spans(text: str):
    """Yield ``(start, end)`` spans of trimmed sentences within ``text``."""
    n = len(text)
    seg_start = 0
    i = 0
    while i < n:
        ch = text[i]
        if ch == "\n":
            span = _trim_span(text, seg_start, i)
            if span:
                yield span
            i += 1
            seg_start = i
            continue
        if ch in _TERMINATORS:
            if ch == "." and not _is_period_boundary(text, i):
                i += 1
                continue
            # Absorb any run of trailing terminators, e.g. "؟!" or "...".
            j = i + 1
            while j < n and text[j] != "\n" and text[j] in _TERMINATORS:
                j += 1
            span = _trim_span(text, seg_start, j)
            if span:
                yield span
            i = j
            seg_start = j
            continue
        i += 1
    span = _trim_span(text, seg_start, n)
    if span:
        yield span


def split_sentences(text: str) -> list[str]:
    """Split ``text`` into sentences.

    Splits on Arabic punctuation (؟ ؛ ، ۔), Latin ``.`` ``!`` ``?`` and
    newlines. Periods inside decimal numbers (``3.14``) and common
    abbreviations (``e.g.``, ``Dr.``) do not create a break.

    Returns a list of trimmed sentence strings (empty list for empty input).
    """
    if not text:
        return []
    return [text[s:e] for (s, e) in _iter_sentence_spans(text)]


# --------------------------------------------------------------------------- #
# Chunking.
# --------------------------------------------------------------------------- #

def _iter_word_spans(text: str, start: int, end: int):
    """Yield spans of whitespace-delimited tokens within ``[start, end)``."""
    i = start
    while i < end:
        if text[i].isspace():
            i += 1
            continue
        j = i
        while j < end and not text[j].isspace():
            j += 1
        yield (i, j)
        i = j


def _unit_spans(text: str, chunk_size: int) -> list[tuple[int, int]]:
    """Break text into the finest units no larger than ``chunk_size``.

    Units are sentences; a sentence longer than ``chunk_size`` is broken into
    words; a word longer than ``chunk_size`` is broken into fixed-size slices.
    Every returned span therefore has ``end - start <= chunk_size``.
    """
    units: list[tuple[int, int]] = []
    for s, e in _iter_sentence_spans(text):
        if e - s <= chunk_size:
            units.append((s, e))
            continue
        for ws, we in _iter_word_spans(text, s, e):
            if we - ws <= chunk_size:
                units.append((ws, we))
            else:
                pos = ws
                while pos < we:
                    units.append((pos, min(pos + chunk_size, we)))
                    pos += chunk_size
    return units


def _merge_units(
    units: list[tuple[int, int]], chunk_size: int, chunk_overlap: int
) -> list[tuple[int, int]]:
    """Greedily pack units into chunk spans, honoring size and overlap."""
    m = len(units)
    spans: list[tuple[int, int]] = []
    a = 0
    while a < m:
        # Extend the window while the contiguous span stays within budget.
        b = a
        while b + 1 < m and (units[b + 1][1] - units[a][0]) <= chunk_size:
            b += 1
        start, end = units[a][0], units[b][1]
        spans.append((start, end))

        if b == m - 1:
            break
        if chunk_overlap == 0:
            a = b + 1
            continue

        # Start the next chunk so its overlap with this one is <= chunk_overlap,
        # snapping to a unit boundary. Always make forward progress.
        target = end - chunk_overlap
        next_a = None
        for k in range(a + 1, b + 1):
            if units[k][0] >= target:
                next_a = k
                break
        if next_a is None:
            next_a = b if b > a else a + 1
        if next_a <= a:
            next_a = a + 1
        a = next_a
    return spans


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    normalize: bool = False,
) -> list[Chunk]:
    """Split ``text`` into overlapping, sentence-aware chunks.

    Args:
        text: The text to chunk.
        chunk_size: Maximum characters per chunk. No chunk exceeds this bound.
        chunk_overlap: Approximate number of characters shared between
            consecutive chunks (for context continuity). Must be smaller than
            ``chunk_size``.
        normalize: If ``True``, run :func:`arabic_rag_kit.normalize` on the
            text first; the returned offsets then refer to the normalized text.

    Returns:
        A list of :class:`Chunk` objects (empty list for empty/whitespace
        input). Offsets index into the (possibly normalized) text.

    Raises:
        ValueError: If ``chunk_size <= 0``, ``chunk_overlap < 0``, or
            ``chunk_overlap >= chunk_size``.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    if normalize:
        text = _normalize(text)
    if not text or not text.strip():
        return []

    units = _unit_spans(text, chunk_size)
    if not units:
        return []

    spans = _merge_units(units, chunk_size, chunk_overlap)
    return [
        Chunk(text=text[s:e], index=idx, start_char=s, end_char=e)
        for idx, (s, e) in enumerate(spans)
    ]
