"""Document loaders for common file types.

``load_txt`` is pure standard library. ``load_pdf`` and ``load_docx`` rely on
optional third-party packages that are imported only when the function is
called, so importing this module never fails on a bare install.

Install the extras with::

    pip install "arabic-rag-kit[docs]"
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["load_txt", "load_pdf", "load_docx"]


def load_txt(path: str | Path, encoding: str = "utf-8") -> str:
    """Read a plain-text file and return its contents.

    Args:
        path: Path to a ``.txt`` (or any UTF-8 text) file.
        encoding: Text encoding, ``utf-8`` by default.
    """
    return Path(path).read_text(encoding=encoding)


def load_pdf(path: str | Path) -> str:
    """Extract text from a PDF using ``pypdf``.

    Pages are joined with blank lines. Requires the ``docs`` extra::

        pip install "arabic-rag-kit[docs]"
    """
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError(
            "load_pdf requires pypdf. Install it with:\n"
            '    pip install "arabic-rag-kit[docs]"'
        ) from exc

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def load_docx(path: str | Path) -> str:
    """Extract text from a Word ``.docx`` file using ``python-docx``.

    Paragraphs are joined with newlines. Requires the ``docs`` extra::

        pip install "arabic-rag-kit[docs]"
    """
    try:
        import docx  # python-docx
    except ImportError as exc:
        raise ImportError(
            "load_docx requires python-docx. Install it with:\n"
            '    pip install "arabic-rag-kit[docs]"'
        ) from exc

    document = docx.Document(str(path))
    return "\n".join(p.text for p in document.paragraphs)
