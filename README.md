# arabic-rag-kit

**The missing first mile for Arabic RAG:** normalize, chunk, and index Arabic
(and mixed Arabic/English) documents — with a dependency-free core.

[![PyPI version](https://img.shields.io/pypi/v/arabic-rag-kit.svg)](https://pypi.org/project/arabic-rag-kit/)
[![Python versions](https://img.shields.io/pypi/pyversions/arabic-rag-kit.svg)](https://pypi.org/project/arabic-rag-kit/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/hasanodeh/arabic-rag-kit/actions/workflows/ci.yml/badge.svg)](https://github.com/hasanodeh/arabic-rag-kit/actions/workflows/ci.yml)

---

## Why this exists

Most RAG and search tooling is built and tested against English. Arabic brings
problems those tools quietly get wrong:

- **Diacritics (tashkeel), tatweel, and letter variants** (`أ`/`إ`/`آ` vs `ا`)
  fragment what should be the same token, tanking retrieval recall.
- **Invisible characters** — zero-width joiners and bidirectional control marks —
  sneak into copied text and corrupt indexes and embeddings.
- **Arabic-Indic digits** (`٠١٢٣`) and **Arabic punctuation** (`؟ ؛ ،`) are
  invisible to English-centric normalizers and sentence splitters, so chunks
  break in the wrong places.

`arabic-rag-kit` handles these correctly, with a **zero-dependency core** so you
can drop it into any pipeline. Embeddings and file loaders are opt-in extras —
the library never forces a vendor or an API key on you.

## Install

```bash
# Core: normalization + chunking. Zero third-party dependencies.
pip install arabic-rag-kit

# Add the numpy-backed vector index:
pip install "arabic-rag-kit[search]"

# Add the sentence-transformers embedder helper:
pip install "arabic-rag-kit[embeddings]"

# Add PDF/DOCX loaders:
pip install "arabic-rag-kit[docs]"

# Everything:
pip install "arabic-rag-kit[all]"
```

Requires Python **3.11+**.

## Quickstart

### 1. Normalize

```python
from arabic_rag_kit import normalize

raw = "الْعَرَبِيَّةُ لُغَةٌ جَمِيلَة… كتـــاب رقم ١٢٣"
print(normalize(raw))
# -> "العربية لغة جميلة… كتاب رقم 123"
```

Every step is toggleable. Meaning-changing folds (hamza, ta-marbuta, alef
maqsura) are **off by default** so you don't distort the text unless you ask:

```python
normalize("مؤسسة على مدرسة", normalize_hamza=True,
          normalize_ta_marbuta=True, normalize_alef_maqsura=True)
# -> "موسسه علي مدرسه"
```

Reuse a configured instance:

```python
from arabic_rag_kit import Normalizer, NormalizerConfig

norm = Normalizer(NormalizerConfig(normalize_hamza=True))
norm("شيء مؤكد")   # -> "شيء موكد"
```

### 2. Chunk (sentence-aware)

```python
from arabic_rag_kit import chunk_text

text = (
    "الذكاء الاصطناعي يغير طريقة عملنا. "
    "أنظمة استرجاع المعلومات تعتمد على تقطيع جيد للنص. "
    "كيف نضمن جودة التقطيع؟ عبر احترام حدود الجمل العربية."
)

chunks = chunk_text(text, chunk_size=80, chunk_overlap=20)
for c in chunks:
    print(f"[{c.index}] ({c.start_char}:{c.end_char}) {c.text}")
```

Chunks never exceed `chunk_size`, prefer to break on Arabic/Latin sentence
boundaries, and carry exact character offsets back into the source. `؟ ؛ ،` and
the Arabic full stop are all recognized; decimals (`3.14`) and abbreviations
(`Dr.`, `e.g.`) don't cause false breaks. Pass `normalize=True` to normalize
before chunking in one step.

### 3. Index & search (optional `[search]` extra)

`VectorIndex` never hardcodes an embedding provider — you hand it any
`embed_fn` (text → vector). Bring your own model, or use the built-in
sentence-transformers helper:

```python
from arabic_rag_kit import VectorIndex, chunk_text
from arabic_rag_kit.search import sentence_transformers_embedder

embed = sentence_transformers_embedder()   # multilingual, handles Arabic
index = VectorIndex(embed)

docs = [c.text for c in chunks]
index.add(docs, metadatas=[{"chunk": c.index} for c in chunks])

for hit in index.search("ما أهمية تقطيع النص؟", k=3):
    print(round(hit.score, 3), hit.metadata, hit.text)
```

Any callable works — no model download required for testing:

```python
def my_embed(text: str) -> list[float]:
    ...  # call OpenAI, Cohere, a local model, whatever
index = VectorIndex(my_embed)
```

### 4. Load documents (optional `[docs]` extra)

```python
from arabic_rag_kit.loaders import load_txt, load_pdf, load_docx

text = load_pdf("report_ar.pdf")     # needs [docs]
text = load_docx("memo_ar.docx")     # needs [docs]
text = load_txt("notes_ar.txt")      # stdlib, always available
```

## API overview

| Symbol | Import | Extra | What it does |
| --- | --- | --- | --- |
| `normalize(text, **opts)` | `arabic_rag_kit` | — | One-shot Arabic normalization |
| `Normalizer` / `NormalizerConfig` | `arabic_rag_kit` | — | Reusable, configured normalizer |
| `split_sentences(text)` | `arabic_rag_kit` | — | Arabic/Latin sentence splitting |
| `chunk_text(text, chunk_size, chunk_overlap, normalize)` | `arabic_rag_kit` | — | Sentence-aware chunking |
| `Chunk` | `arabic_rag_kit` | — | `text, index, start_char, end_char` |
| `VectorIndex` | `arabic_rag_kit` | `[search]` | Cosine-similarity vector index |
| `sentence_transformers_embedder(model_name)` | `arabic_rag_kit.search` | `[embeddings]` | Ready-made `embed_fn` |
| `load_txt` / `load_pdf` / `load_docx` | `arabic_rag_kit.loaders` | `[docs]`\* | File loaders (\*txt is stdlib) |

### Normalization options (defaults)

| Option | Default | Effect |
| --- | --- | --- |
| `remove_diacritics` | `True` | Strip tashkeel/harakat (U+064B–U+0652, U+0670) |
| `remove_tatweel` | `True` | Remove kashida elongation (U+0640) |
| `normalize_alef` | `True` | `أ إ آ ٱ` → `ا` |
| `normalize_hamza` | `False` | `ؤ` → `و`, `ئ` → `ي` |
| `normalize_ta_marbuta` | `False` | `ة` → `ه` |
| `normalize_alef_maqsura` | `False` | `ى` → `ي` |
| `convert_digits` | `True` | `٠–٩` and `۰–۹` → `0–9` |
| `strip_control_chars` | `True` | Remove zero-width & bidi controls |
| `collapse_whitespace` | `True` | Collapse runs of whitespace and trim |

## Development

```bash
pip install -e ".[dev]"
ruff check .
pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Built by GBM

Created and maintained by **Hasan Odeh** at **Gulf Business Machines (GBM)**.
Born out of real Arabic RAG work, and open-sourced because Arabic NLP deserves
better tooling. Contributions welcome.

## License

[MIT](LICENSE) © Gulf Business Machines (GBM)
