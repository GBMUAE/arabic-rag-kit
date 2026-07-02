# arabic-rag-kit

**The missing first mile for Arabic RAG:** normalize, chunk, and index Arabic
(and mixed Arabic/English) documents — with a dependency-free core.

[![PyPI version](https://img.shields.io/pypi/v/arabic-rag-kit.svg)](https://pypi.org/project/arabic-rag-kit/)
[![Python versions](https://img.shields.io/pypi/pyversions/arabic-rag-kit.svg)](https://pypi.org/project/arabic-rag-kit/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/GBMUAE/arabic-rag-kit/actions/workflows/ci.yml/badge.svg)](https://github.com/GBMUAE/arabic-rag-kit/actions/workflows/ci.yml)

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

## Use cases

Reach for `arabic-rag-kit` whenever Arabic text enters a search or LLM pipeline:

- **RAG over Arabic documents** — clean, split, and chunk PDFs/Word/text before
  embedding, so retrieval actually finds the right passage. This is the core
  use case the library is named for.
- **Better search recall** — normalize both the indexed text and the query so
  that `مُحَمَّد`, `محمّد`, and `محمـــد` all match `محمد`. Diacritics, tatweel,
  and alef variants stop fragmenting your index.
- **Deduplication / clustering** — use an aggressive normalization profile as a
  canonical "match key" to detect near-duplicate Arabic strings.
- **Cleaning scraped / copy-pasted text** — strip the invisible zero-width and
  bidirectional control characters that break tokenizers, embeddings, and diffs.
- **Data prep for fine-tuning or classification** — consistent normalization and
  digit handling (`٢٠٢٦` → `2026`) as a preprocessing step.
- **Sentence segmentation** — split Arabic text on `؟ ؛ ،` and the Arabic full
  stop (not just Latin punctuation) for summarization, translation, or display.
- **A lightweight in-memory vector search** — prototype semantic search with any
  embedding function, no database or vendor lock-in.

If your text is English-only, you don't need this. If it's Arabic or mixed
Arabic/English, these are exactly the sharp edges that quietly hurt quality.

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

### The 30-second RAG pipeline

The whole point of the library — turn a raw Arabic document into clean,
retrievable chunks and search them — end to end:

```python
from arabic_rag_kit import chunk_text, VectorIndex
from arabic_rag_kit.search import sentence_transformers_embedder

document = (
    "تأسست شركة جي بي إم في عام ١٩٩٠. "
    "تقدم الشركة حلولاً في مجال الذكاء الاصطناعي والحوسبة السحابية. "
    "كيف يمكن للعملاء البدء؟ عبر التواصل مع فريق المبيعات."
)

# 1) Normalize + split into overlapping, sentence-aware chunks in one step.
chunks = chunk_text(document, chunk_size=90, chunk_overlap=20, normalize=True)

# 2) Embed and index (bring any embedding function you like).
index = VectorIndex(sentence_transformers_embedder())
index.add(
    [c.text for c in chunks],
    metadatas=[{"chunk_id": c.index} for c in chunks],
)

# 3) Ask a question — retrieval finds the right passage.
for hit in index.search("كيف يبدأ العملاء؟", k=2):
    print(round(hit.score, 3), hit.metadata, hit.text)
```

The sections below break the same three steps down.

### 1. Normalize

```python
from arabic_rag_kit import normalize

raw = "الْعَرَبِيَّةُ لُغَةٌ جَمِيلَة… كتـــاب رقم ١٢٣"
normalize(raw)
# -> "العربية لغة جميلة… كتاب رقم 123"
```

**Why it matters — the same word, four ways, becomes one.** Diacritics, tatweel,
and alef variants all fold to a single canonical form, so search and dedup work:

```python
normalize("مُحَمَّد")   # diacritics  -> "محمد"
normalize("محمّد")      # shadda      -> "محمد"
normalize("محمـــد")    # tatweel     -> "محمد"

# Perfect for building a "match key" — normalize the index AND the query:
normalize("مُحَمَّد") == normalize("محمد")   # -> True
```

Every step is toggleable. Meaning-changing folds (hamza, ta-marbuta, alef
maqsura) are **off by default** so you don't distort text unless you ask:

```python
normalize("مسؤول", normalize_hamza=True)          # -> "مسوول"   (ؤ → و)
normalize("جامعة", normalize_ta_marbuta=True)      # -> "جامعه"   (ة → ه)
normalize("مصطفى", normalize_alef_maqsura=True)    # -> "مصطفي"   (ى → ي)
```

Digits from both Arabic-Indic sets are converted, and invisible control
characters are stripped:

```python
normalize("سنة ٢٠٢٦ و ۱۹۹۹")     # -> "سنة 2026 و 1999"
normalize("  الذكاء   الاصطناعي\n\tمفيد  ")   # -> "الذكاء الاصطناعي مفيد"
```

Reuse one configured instance across a whole corpus. For a search index you
often want the *aggressive* profile so more variants collapse together:

```python
from arabic_rag_kit import Normalizer, NormalizerConfig

search_key = Normalizer(NormalizerConfig(
    normalize_hamza=True,
    normalize_ta_marbuta=True,
    normalize_alef_maqsura=True,
))
search_key("المُؤسَّسة على الطُّلّاب")   # -> "الموسسه علي الطلاب"
```

### 2. Split & chunk (sentence-aware)

`split_sentences` understands Arabic punctuation and does not break on decimals
or abbreviations:

```python
from arabic_rag_kit import split_sentences

split_sentences("الإصدار 3.14 متاح الآن. راجع e.g. الوثائق! هل لديك سؤال؟")
# -> ['الإصدار 3.14 متاح الآن.', 'راجع e.g. الوثائق!', 'هل لديك سؤال؟']
```

`chunk_text` packs sentences into overlapping chunks for embedding:

```python
from arabic_rag_kit import chunk_text

text = (
    "الذكاء الاصطناعي يغير طريقة عملنا. "
    "أنظمة استرجاع المعلومات تعتمد على تقطيع جيد للنص. "
    "كيف نضمن جودة التقطيع؟ عبر احترام حدود الجمل العربية."
)

for c in chunk_text(text, chunk_size=80, chunk_overlap=20):
    print(f"[{c.index}] ({c.start_char}:{c.end_char}) {c.text}")
# [0] (0:34)   الذكاء الاصطناعي يغير طريقة عملنا.
# [1] (35:107) أنظمة استرجاع المعلومات تعتمد على تقطيع جيد للنص. كيف نضمن جودة التقطيع؟
# [2] (85:138) كيف نضمن جودة التقطيع؟ عبر احترام حدود الجمل العربية.
# note how chunk [2] starts at 85 — before [1] ends at 107 — that is the overlap.
```

Each `Chunk` has `text`, `index`, `start_char`, and `end_char`. Chunks never
exceed `chunk_size`, prefer to break on Arabic/Latin sentence boundaries, and
the offsets index straight back into the source (`text[c.start_char:c.end_char]
== c.text`) — handy for highlighting the retrieved passage. Pass
`normalize=True` to normalize and chunk in one call (offsets then refer to the
normalized text).

### 3. Index & search (optional `[search]` extra)

`VectorIndex` never hardcodes an embedding provider — you hand it any
`embed_fn` (text → vector). Option A: the built-in multilingual helper.

```python
from arabic_rag_kit import VectorIndex
from arabic_rag_kit.search import sentence_transformers_embedder

index = VectorIndex(sentence_transformers_embedder())
index.add(
    ["القاهرة عاصمة مصر", "باريس عاصمة فرنسا"],
    metadatas=[{"country": "مصر"}, {"country": "فرنسا"}],
)

for hit in index.search("ما هي عاصمة مصر؟", k=1):
    print(hit.text, round(hit.score, 3), hit.metadata)
    # -> القاهرة عاصمة مصر  <cosine score>  {'country': 'مصر'}
```

Option B: **bring your own model / API.** Any callable that returns a vector
works — OpenAI, Cohere, a local model, whatever. No vendor lock-in, no API key
baked into the library:

```python
from openai import OpenAI
client = OpenAI()

def openai_embed(text: str) -> list[float]:
    return client.embeddings.create(
        model="text-embedding-3-small", input=text
    ).data[0].embedding

index = VectorIndex(openai_embed)
```

Each result is a `SearchResult` with `.text`, `.score` (cosine similarity),
`.metadata`, and `.index`. Attach metadata (source file, page, chunk id, URL)
on `add` and read it back on every hit to cite or filter your answers.

### 4. Load documents (optional `[docs]` extra)

Loaders return plain text — feed it straight into the pipeline above:

```python
from arabic_rag_kit.loaders import load_txt, load_pdf, load_docx
from arabic_rag_kit import chunk_text, VectorIndex
from arabic_rag_kit.search import sentence_transformers_embedder

raw = load_pdf("tender_ar.pdf")        # needs [docs];  or load_docx / load_txt
chunks = chunk_text(raw, chunk_size=1000, chunk_overlap=200, normalize=True)

index = VectorIndex(sentence_transformers_embedder())
index.add(
    [c.text for c in chunks],
    metadatas=[{"source": "tender_ar.pdf", "chunk_id": c.index} for c in chunks],
)

answer_context = index.search("ما هي شروط التأهيل؟", k=4)
```

```python
load_txt("notes_ar.txt")     # stdlib, always available
load_pdf("report_ar.pdf")    # needs [docs]  (pypdf)
load_docx("memo_ar.docx")    # needs [docs]  (python-docx)
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
