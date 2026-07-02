"""arabic-rag-kit — prepare Arabic (and mixed Arabic/English) documents for RAG.

A small, dependency-light toolkit for the unglamorous-but-critical first mile
of an Arabic RAG or search pipeline: normalization, sentence-aware chunking,
and a provider-agnostic vector index.

Built by Hasan Odeh at Gulf Business Machines (GBM). MIT licensed.
"""

from __future__ import annotations

from .chunk import Chunk, chunk_text, split_sentences
from .normalize import Normalizer, NormalizerConfig, normalize

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # normalize
    "normalize",
    "Normalizer",
    "NormalizerConfig",
    # chunk
    "chunk_text",
    "split_sentences",
    "Chunk",
    # search (imported lazily; see __getattr__)
    "VectorIndex",
]


def __getattr__(name: str):
    """Lazily expose :class:`VectorIndex` without importing numpy at import time."""
    if name == "VectorIndex":
        from .search import VectorIndex

        return VectorIndex
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
