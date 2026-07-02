# Contributing to arabic-rag-kit

Thanks for your interest in improving **arabic-rag-kit**! Contributions of all
kinds are welcome — bug reports, documentation, tests, and code.

## Development setup

```bash
git clone https://github.com/GBMUAE/arabic-rag-kit.git
cd arabic-rag-kit

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -e ".[dev]"          # editable install with dev tools
```

## Running the checks

Before opening a pull request, make sure both the linter and the test suite
pass:

```bash
ruff check .
pytest
```

Please add tests for any new behavior. The suite must stay green on Python
3.11 and 3.12 (this is what CI runs).

## Design principles

- **Core stays dependency-free.** The base install must have zero required
  third-party dependencies. Anything heavier (numpy, sentence-transformers,
  pypdf, python-docx) belongs behind an optional extra and must be imported
  lazily with a clear `ImportError` pointing at the right extra.
- **Correctness first for Arabic.** Normalization and sentence splitting
  should be justified against real Arabic text. Include examples in the PR.
- **Small, composable functions.** Prefer pure helpers that do one thing.

## Commit messages

Use clear, imperative commit messages (e.g. "Add hamza folding option"). Group
related changes into logical commits.

## Reporting bugs

Open an issue with a minimal reproducible example, the input text (copy the
exact Unicode), the expected output, and the actual output.

## License

By contributing, you agree that your contributions will be licensed under the
MIT License that covers the project.
