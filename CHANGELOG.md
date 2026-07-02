# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-07-03

### Added
- Command-line interface `arabic-rag-kit` with three subcommands:
  `normalize`, `sentences`, and `chunk`. Reads from an argument, `--input`
  file, or stdin; writes to stdout or `--output`. All normalization options are
  exposed as flags, and `chunk` supports `--json` output with offsets. Pure
  standard library — the core stays dependency-free.

## [0.1.0] - 2026-07-02

### Added
- `normalize()`, `Normalizer`, and `NormalizerConfig` for Arabic text
  normalization: diacritic/tashkeel removal, tatweel stripping, alef/hamza/
  ta-marbuta/alef-maqsura folding, Arabic-Indic and Eastern Arabic-Indic digit
  conversion, zero-width/bidi control-character stripping, and whitespace
  collapsing. Pure standard library.
- Composable single-purpose helpers (`remove_diacritics`, `normalize_alef`,
  `convert_digits`, `strip_control_chars`, …).
- `split_sentences()` — Arabic- and Latin-aware sentence splitting that does
  not break on decimals or common abbreviations.
- `chunk_text()` and the `Chunk` dataclass — recursive, sentence-aware
  character chunking with configurable overlap and exact character offsets.
- `VectorIndex` — a provider-agnostic, numpy-backed cosine-similarity index
  that takes a caller-supplied `embed_fn` (optional `[search]` extra).
- `sentence_transformers_embedder()` — optional helper returning an `embed_fn`
  backed by sentence-transformers (optional `[embeddings]` extra).
- `load_txt`, `load_pdf`, `load_docx` document loaders (optional `[docs]`
  extra for PDF/DOCX).
- Full pytest suite, ruff configuration, and GitHub Actions CI + PyPI Trusted
  Publishing workflows.

[Unreleased]: https://github.com/GBMUAE/arabic-rag-kit/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/GBMUAE/arabic-rag-kit/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/GBMUAE/arabic-rag-kit/releases/tag/v0.1.0
