"""Tests for arabic_rag_kit.search.

Uses a tiny deterministic bag-of-words embedder so the suite runs fast and
without any real embedding model. Skips gracefully if numpy is not installed.
"""

import hashlib

import pytest

np = pytest.importorskip("numpy", reason="numpy is required for VectorIndex")

from arabic_rag_kit import VectorIndex  # noqa: E402
from arabic_rag_kit.search import SearchResult  # noqa: E402

_DIM = 64


def fake_embed(text: str):
    """Deterministic bag-of-words hashing embedder.

    Each whitespace token is hashed into one of ``_DIM`` buckets. Texts that
    share tokens get similar vectors, so cosine similarity is meaningful —
    but there is no randomness and no model download.
    """
    vec = [0.0] * _DIM
    for token in text.split():
        h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
        vec[h % _DIM] += 1.0
    if not any(vec):
        vec[0] = 1.0  # avoid the zero vector for empty strings
    return vec


def test_add_and_len():
    idx = VectorIndex(fake_embed)
    assert len(idx) == 0
    idx.add(["القاهرة عاصمة مصر", "باريس عاصمة فرنسا"])
    assert len(idx) == 2
    assert idx.dim == _DIM


def test_search_returns_most_similar_first():
    idx = VectorIndex(fake_embed)
    idx.add([
        "القاهرة عاصمة مصر",
        "باريس عاصمة فرنسا",
        "الطبخ المنزلي ممتع",
    ])
    results = idx.search("ما هي عاصمة مصر القاهرة", k=3)
    assert results[0].text == "القاهرة عاصمة مصر"
    # Scores are sorted in descending order.
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_search_result_type_and_fields():
    idx = VectorIndex(fake_embed)
    idx.add(["نص تجريبي"], metadatas=[{"id": 7}])
    (hit,) = idx.search("نص تجريبي", k=1)
    assert isinstance(hit, SearchResult)
    assert hit.text == "نص تجريبي"
    assert hit.metadata == {"id": 7}
    assert hit.index == 0
    assert hit.score == pytest.approx(1.0, abs=1e-6)


def test_metadata_is_returned():
    idx = VectorIndex(fake_embed)
    idx.add(
        ["وثيقة أولى", "وثيقة ثانية"],
        metadatas=[{"source": "a.txt"}, {"source": "b.txt"}],
    )
    results = idx.search("وثيقة ثانية", k=1)
    assert results[0].metadata["source"] == "b.txt"


def test_k_limits_results():
    idx = VectorIndex(fake_embed)
    idx.add([f"جملة رقم {i}" for i in range(10)])
    assert len(idx.search("جملة", k=3)) == 3
    # k larger than the corpus returns everything, not an error.
    assert len(idx.search("جملة", k=99)) == 10


def test_search_empty_index_returns_empty():
    idx = VectorIndex(fake_embed)
    assert idx.search("أي شيء", k=5) == []


def test_metadatas_length_mismatch_raises():
    idx = VectorIndex(fake_embed)
    with pytest.raises(ValueError):
        idx.add(["a", "b"], metadatas=[{"x": 1}])


def test_dimension_mismatch_raises():
    idx = VectorIndex(fake_embed)
    idx.add(["نص عادي"])
    with pytest.raises(ValueError):
        # A second embedder with a different dimension.
        idx.embed_fn = lambda t: [1.0, 2.0, 3.0]
        idx.add(["نص آخر"])


def test_invalid_k_raises():
    idx = VectorIndex(fake_embed)
    idx.add(["نص"])
    with pytest.raises(ValueError):
        idx.search("نص", k=0)


def test_non_callable_embed_fn_raises():
    with pytest.raises(TypeError):
        VectorIndex("not-callable")


def test_scores_are_cosine_bounded():
    idx = VectorIndex(fake_embed)
    idx.add(["ألف باء", "جيم دال", "ألف جيم"])
    for r in idx.search("ألف باء جيم", k=3):
        assert -1.0001 <= r.score <= 1.0001


def test_missing_numpy_raises_helpful_error(monkeypatch):
    """If numpy import fails, the error should point at the [search] extra."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "numpy":
            raise ImportError("No module named 'numpy'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"arabic-rag-kit\[search\]"):
        VectorIndex(fake_embed)
