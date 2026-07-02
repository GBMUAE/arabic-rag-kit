"""Arabic text normalization for RAG and search pipelines.

This module is **pure standard library** — it has no third-party dependencies.

The public surface is:

* :func:`normalize` — a one-shot convenience function.
* :class:`NormalizerConfig` — a dataclass describing which operations to run.
* :class:`Normalizer` — a reusable, pre-configured normalizer.

Plus a set of small, composable helpers (``remove_diacritics``,
``normalize_alef``, ``convert_digits`` …) that each do exactly one thing so you
can build your own pipeline if the defaults do not fit.

All operations are Unicode-aware and safe to run on mixed Arabic/English text:
characters that are not targeted by a given operation are passed through
untouched.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = [
    "NormalizerConfig",
    "Normalizer",
    "normalize",
    "remove_diacritics",
    "remove_tatweel",
    "normalize_alef",
    "normalize_hamza",
    "normalize_ta_marbuta",
    "normalize_alef_maqsura",
    "convert_digits",
    "strip_control_chars",
    "collapse_whitespace",
]

# --------------------------------------------------------------------------- #
# Character sets (documented with their Unicode code points for reviewers).
# --------------------------------------------------------------------------- #

# Tashkeel / harakat: U+064B..U+0652 (fathatan..sukun) + U+0670 superscript alef.
_DIACRITICS = "".join(chr(c) for c in range(0x064B, 0x0653)) + "ٰ"
_DIACRITICS_RE = re.compile("[" + re.escape(_DIACRITICS) + "]")

# Tatweel / kashida.
_TATWEEL = "ـ"

# Alef variants -> plain alef (U+0627).
#   أ U+0623, إ U+0625, آ U+0622, ٱ U+0671  ->  ا U+0627
_ALEF_MAP = str.maketrans({
    "أ": "ا",
    "إ": "ا",
    "آ": "ا",
    "ٱ": "ا",
})

# Hamza carriers.
#   ؤ U+0624 -> و U+0648,  ئ U+0626 -> ي U+064A
_HAMZA_MAP = str.maketrans({
    "ؤ": "و",
    "ئ": "ي",
})

# Ta marbuta ة U+0629 -> ه U+0647
_TA_MARBUTA_MAP = str.maketrans({"ة": "ه"})

# Alef maqsura ى U+0649 -> ي U+064A
_ALEF_MAQSURA_MAP = str.maketrans({"ى": "ي"})

# Arabic-Indic (U+0660..U+0669) and Eastern Arabic-Indic (U+06F0..U+06F9)
# digits -> ASCII 0..9.
_DIGIT_MAP = str.maketrans(
    {chr(0x0660 + i): str(i) for i in range(10)}
    | {chr(0x06F0 + i): str(i) for i in range(10)}
)

# Zero-width and bidirectional control characters:
#   U+200B..U+200F (ZWSP, ZWNJ, ZWJ, LRM, RLM),
#   U+202A..U+202E (LRE, RLE, PDF, LRO, RLO),
#   U+FEFF (BOM / zero-width no-break space).
_CONTROL_CHARS = (
    "".join(chr(c) for c in range(0x200B, 0x2010))
    + "".join(chr(c) for c in range(0x202A, 0x202F))
    + "﻿"
)
_CONTROL_RE = re.compile("[" + re.escape(_CONTROL_CHARS) + "]")

_WHITESPACE_RE = re.compile(r"\s+")


# --------------------------------------------------------------------------- #
# Composable helpers — each does one thing and returns a new string.
# --------------------------------------------------------------------------- #

def remove_diacritics(text: str) -> str:
    """Remove Arabic tashkeel/harakat (U+064B–U+0652 and U+0670)."""
    return _DIACRITICS_RE.sub("", text)


def remove_tatweel(text: str) -> str:
    """Remove tatweel/kashida elongation characters (U+0640)."""
    return text.replace(_TATWEEL, "")


def normalize_alef(text: str) -> str:
    """Fold alef variants (أ إ آ ٱ) to plain alef (ا)."""
    return text.translate(_ALEF_MAP)


def normalize_hamza(text: str) -> str:
    """Fold hamza carriers (ؤ → و, ئ → ي)."""
    return text.translate(_HAMZA_MAP)


def normalize_ta_marbuta(text: str) -> str:
    """Fold ta marbuta (ة → ه)."""
    return text.translate(_TA_MARBUTA_MAP)


def normalize_alef_maqsura(text: str) -> str:
    """Fold alef maqsura (ى → ي)."""
    return text.translate(_ALEF_MAQSURA_MAP)


def convert_digits(text: str) -> str:
    """Convert Arabic-Indic and Eastern Arabic-Indic digits to ASCII 0–9."""
    return text.translate(_DIGIT_MAP)


def strip_control_chars(text: str) -> str:
    """Remove zero-width and bidi control characters."""
    return _CONTROL_RE.sub("", text)


def collapse_whitespace(text: str) -> str:
    """Collapse any run of whitespace to a single space and strip the ends."""
    return _WHITESPACE_RE.sub(" ", text).strip()


# --------------------------------------------------------------------------- #
# Config + reusable normalizer.
# --------------------------------------------------------------------------- #

@dataclass
class NormalizerConfig:
    """Toggles for every normalization step.

    Defaults are tuned for RAG/search recall on Modern Standard Arabic: the
    "aggressive" folds that change meaning (hamza, ta marbuta, alef maqsura)
    are **off** by default, while safe normalizations (diacritics, tatweel,
    alef, digits, control chars, whitespace) are **on**.
    """

    remove_diacritics: bool = True
    remove_tatweel: bool = True
    normalize_alef: bool = True
    normalize_hamza: bool = False
    normalize_ta_marbuta: bool = False
    normalize_alef_maqsura: bool = False
    convert_digits: bool = True
    strip_control_chars: bool = True
    collapse_whitespace: bool = True


class Normalizer:
    """A reusable normalizer built from a :class:`NormalizerConfig`.

    Create one instance and call it many times::

        norm = Normalizer(NormalizerConfig(normalize_hamza=True))
        norm("النَّصُّ العربي")  # -> "النص العربي"

    The instance is stateless with respect to input, so it is safe to share
    across threads.
    """

    def __init__(self, config: NormalizerConfig | None = None) -> None:
        self.config = config or NormalizerConfig()

    def normalize(self, text: str) -> str:
        """Apply the configured pipeline to ``text``."""
        if not text:
            return ""

        cfg = self.config
        # Order matters: strip invisibles first, then character folds, and
        # collapse whitespace last so earlier steps cannot leave stray runs.
        if cfg.strip_control_chars:
            text = strip_control_chars(text)
        if cfg.remove_diacritics:
            text = remove_diacritics(text)
        if cfg.remove_tatweel:
            text = remove_tatweel(text)
        if cfg.normalize_alef:
            text = normalize_alef(text)
        if cfg.normalize_hamza:
            text = normalize_hamza(text)
        if cfg.normalize_ta_marbuta:
            text = normalize_ta_marbuta(text)
        if cfg.normalize_alef_maqsura:
            text = normalize_alef_maqsura(text)
        if cfg.convert_digits:
            text = convert_digits(text)
        if cfg.collapse_whitespace:
            text = collapse_whitespace(text)
        return text

    # Allow ``normalizer(text)`` as a shorthand for ``normalizer.normalize``.
    __call__ = normalize


def normalize(
    text: str,
    *,
    remove_diacritics: bool = True,
    remove_tatweel: bool = True,
    normalize_alef: bool = True,
    normalize_hamza: bool = False,
    normalize_ta_marbuta: bool = False,
    normalize_alef_maqsura: bool = False,
    convert_digits: bool = True,
    strip_control_chars: bool = True,
    collapse_whitespace: bool = True,
) -> str:
    """Normalize Arabic (or mixed Arabic/English) text in one call.

    Every step is individually toggleable. See :class:`NormalizerConfig` for
    the defaults and what each flag does.

    Example::

        >>> normalize("الْأَرْقَام: ١٢٣ ﻿and English")
        'الارقام: 123 and English'
    """
    config = NormalizerConfig(
        remove_diacritics=remove_diacritics,
        remove_tatweel=remove_tatweel,
        normalize_alef=normalize_alef,
        normalize_hamza=normalize_hamza,
        normalize_ta_marbuta=normalize_ta_marbuta,
        normalize_alef_maqsura=normalize_alef_maqsura,
        convert_digits=convert_digits,
        strip_control_chars=strip_control_chars,
        collapse_whitespace=collapse_whitespace,
    )
    return Normalizer(config).normalize(text)
