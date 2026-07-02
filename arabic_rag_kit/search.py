"""A tiny, provider-agnostic vector index for semantic search.

This module has an **optional** dependency on ``numpy``. The heavy lifting of
turning text into vectors is delegated to a caller-supplied ``embed_fn`` — the
index never hardcodes an embedding provider and never needs an API key.

Install the extra with::

    pip install "arabic-rag-kit[search]"

Example::

    from arabic_rag_kit import VectorIndex

    def embed(text):            # your embedding of choice
        ...

    index = VectorIndex(embed)
    index.add(["القاهرة عاصمة مصر", "باريس عاصمة فرنسا"])
    for hit in index.search("ما هي عاصمة مصر؟", k=1):
        print(hit.text, hit.score)
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["VectorIndex", "SearchResult", "sentence_transformers_embedder"]

EmbedFn = Callable[[str], Sequence[float]]

_NUMPY_HINT = (
    "VectorIndex requires numpy. Install it with:\n"
    '    pip install "arabic-rag-kit[search]"'
)


def _require_numpy():
    """Import numpy lazily, raising a helpful error if it is missing."""
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise ImportError(_NUMPY_HINT) from exc
    return np


@dataclass
class SearchResult:
    """A single search hit."""

    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    index: int = -1


class VectorIndex:
    """An in-memory cosine-similarity index over embedded texts.

    Args:
        embed_fn: Callable mapping a string to a vector (``list[float]`` or a
            numpy array). Called once per text on :meth:`add` and once per
            query on :meth:`search`.
        normalize: If ``True`` (default), stored vectors are L2-normalized so
            that cosine similarity reduces to a dot product. Set to ``False``
            only if your ``embed_fn`` already returns unit vectors.
    """

    def __init__(self, embed_fn: EmbedFn, *, normalize: bool = True) -> None:
        if not callable(embed_fn):
            raise TypeError("embed_fn must be callable")
        self._np = _require_numpy()
        self.embed_fn = embed_fn
        self.normalize = normalize
        self._matrix = None  # numpy array, shape (n, dim)
        self.texts: list[str] = []
        self.metadatas: list[dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self.texts)

    @property
    def dim(self) -> int | None:
        """Embedding dimension, or ``None`` if the index is empty."""
        if self._matrix is None:
            return None
        return int(self._matrix.shape[1])

    def _vectorize(self, text: str):
        np = self._np
        vec = self._np.asarray(self.embed_fn(text), dtype=np.float32).ravel()
        if vec.ndim != 1 or vec.size == 0:
            raise ValueError("embed_fn must return a non-empty 1-D vector")
        if self.normalize:
            norm = float(np.linalg.norm(vec))
            if norm > 0.0:
                vec = vec / norm
        return vec

    def add(
        self,
        texts: Iterable[str],
        metadatas: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        """Embed and add ``texts`` (with optional parallel ``metadatas``)."""
        texts = list(texts)
        if metadatas is not None and len(metadatas) != len(texts):
            raise ValueError("metadatas must be the same length as texts")
        if not texts:
            return

        np = self._np
        new_vecs = np.vstack([self._vectorize(t) for t in texts])
        if self._matrix is None:
            self._matrix = new_vecs
        else:
            if new_vecs.shape[1] != self._matrix.shape[1]:
                raise ValueError(
                    f"embedding dim {new_vecs.shape[1]} does not match "
                    f"existing dim {self._matrix.shape[1]}"
                )
            self._matrix = np.vstack([self._matrix, new_vecs])

        self.texts.extend(texts)
        if metadatas is None:
            self.metadatas.extend({} for _ in texts)
        else:
            self.metadatas.extend(dict(m) for m in metadatas)

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        """Return the top-``k`` matches for ``query`` by cosine similarity."""
        if k <= 0:
            raise ValueError("k must be a positive integer")
        if self._matrix is None or len(self.texts) == 0:
            return []

        np = self._np
        q = self._vectorize(query)
        # Cosine similarity. Stored rows are unit vectors when normalize=True;
        # otherwise divide by their norms here so the score stays in [-1, 1].
        scores = self._matrix @ q
        if not self.normalize:
            row_norms = np.linalg.norm(self._matrix, axis=1)
            qn = float(np.linalg.norm(q))
            denom = row_norms * (qn or 1.0)
            denom[denom == 0.0] = 1.0
            scores = scores / denom

        k = min(k, len(self.texts))
        # Partial top-k, then sort just those k descending.
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [
            SearchResult(
                text=self.texts[i],
                score=float(scores[i]),
                metadata=self.metadatas[i],
                index=int(i),
            )
            for i in top
        ]


def sentence_transformers_embedder(
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
) -> EmbedFn:
    """Return an ``embed_fn`` backed by ``sentence-transformers``.

    The model — and the ``sentence-transformers`` dependency — is loaded only
    when this function is called, not at import time. Install the extra with::

        pip install "arabic-rag-kit[embeddings]"

    Args:
        model_name: Any model from the sentence-transformers hub. The default
            is multilingual and handles Arabic well.

    Returns:
        A callable mapping ``str -> list[float]`` suitable for
        :class:`VectorIndex`.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence_transformers_embedder requires sentence-transformers. "
            'Install it with:\n    pip install "arabic-rag-kit[embeddings]"'
        ) from exc

    model = SentenceTransformer(model_name)

    def embed(text: str) -> list[float]:
        return model.encode(text, convert_to_numpy=True).tolist()

    return embed
