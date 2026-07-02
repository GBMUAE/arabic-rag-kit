"""Command-line interface for arabic-rag-kit.

Pure standard library (``argparse``) — the CLI keeps the core dependency-free.

Three subcommands, each reading from a positional argument, ``--input FILE``,
or standard input, and writing to standard output or ``--output FILE``:

* ``normalize``  — normalize Arabic text (all options exposed as flags).
* ``sentences``  — split text into sentences, one per line.
* ``chunk``      — chunk text; plain text separated by a rule, or ``--json``.

Examples::

    echo "الْعَرَبِيَّةُ ١٢٣" | arabic-rag-kit normalize
    arabic-rag-kit chunk -i doc.txt --size 500 --overlap 100 --normalize --json
    arabic-rag-kit sentences "جملة أولى. جملة ثانية؟"
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from . import __version__
from .chunk import chunk_text, split_sentences
from .normalize import Normalizer, NormalizerConfig

__all__ = ["main", "build_parser"]


def _read_text(args: argparse.Namespace) -> str:
    """Resolve input text from positional arg, --input file, or stdin."""
    if getattr(args, "text", None):
        return args.text
    if getattr(args, "input", None):
        with open(args.input, encoding=args.encoding) as fh:
            return fh.read()
    return sys.stdin.read()


def _write_output(args: argparse.Namespace, text: str) -> None:
    """Write ``text`` to --output file or stdout."""
    if getattr(args, "output", None):
        with open(args.output, "w", encoding=args.encoding) as fh:
            fh.write(text)
    else:
        sys.stdout.write(text)
        if not text.endswith("\n"):
            sys.stdout.write("\n")


def _config_from_args(args: argparse.Namespace) -> NormalizerConfig:
    """Build a NormalizerConfig from the normalize/chunk flags."""
    return NormalizerConfig(
        remove_diacritics=not args.no_diacritics,
        remove_tatweel=not args.no_tatweel,
        normalize_alef=not args.no_alef,
        normalize_hamza=args.hamza,
        normalize_ta_marbuta=args.ta_marbuta,
        normalize_alef_maqsura=args.alef_maqsura,
        convert_digits=not args.no_digits,
        strip_control_chars=not args.no_control,
        collapse_whitespace=not args.no_whitespace,
    )


def _add_normalize_flags(parser: argparse.ArgumentParser) -> None:
    """Attach the shared normalization flags to a subparser."""
    group = parser.add_argument_group("normalization options")
    # Enable folds that are off by default.
    group.add_argument("--hamza", action="store_true",
                       help="fold hamza carriers (ؤ→و, ئ→ي)")
    group.add_argument("--ta-marbuta", action="store_true",
                       help="fold ta marbuta (ة→ه)")
    group.add_argument("--alef-maqsura", action="store_true",
                       help="fold alef maqsura (ى→ي)")
    # Disable steps that are on by default.
    group.add_argument("--no-diacritics", action="store_true",
                       help="keep diacritics/tashkeel")
    group.add_argument("--no-tatweel", action="store_true",
                       help="keep tatweel/kashida")
    group.add_argument("--no-alef", action="store_true",
                       help="keep alef variants (أ إ آ ٱ)")
    group.add_argument("--no-digits", action="store_true",
                       help="keep Arabic-Indic digits")
    group.add_argument("--no-control", action="store_true",
                       help="keep zero-width/bidi control characters")
    group.add_argument("--no-whitespace", action="store_true",
                       help="do not collapse whitespace")


def _add_io_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("text", nargs="?", help="input text (else --input or stdin)")
    parser.add_argument("-i", "--input", help="read input from this file")
    parser.add_argument("-o", "--output", help="write output to this file")
    parser.add_argument("--encoding", default="utf-8", help="file encoding (utf-8)")


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="arabic-rag-kit",
        description="Prepare Arabic text for RAG and search: normalize, "
                    "split sentences, and chunk — from the command line.",
    )
    parser.add_argument("-V", "--version", action="version",
                        version=f"arabic-rag-kit {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    # normalize
    p_norm = sub.add_parser("normalize", help="normalize Arabic text")
    _add_io_args(p_norm)
    _add_normalize_flags(p_norm)
    p_norm.set_defaults(func=_cmd_normalize)

    # sentences
    p_sent = sub.add_parser("sentences", help="split text into sentences (one per line)")
    _add_io_args(p_sent)
    p_sent.set_defaults(func=_cmd_sentences)

    # chunk
    p_chunk = sub.add_parser("chunk", help="chunk text for embedding")
    _add_io_args(p_chunk)
    p_chunk.add_argument("--size", type=int, default=1000,
                         help="max characters per chunk (default 1000)")
    p_chunk.add_argument("--overlap", type=int, default=200,
                         help="overlap characters between chunks (default 200)")
    p_chunk.add_argument("--normalize", action="store_true",
                         help="normalize before chunking (default settings)")
    p_chunk.add_argument("--json", action="store_true",
                         help="emit JSON objects with offsets instead of plain text")
    p_chunk.set_defaults(func=_cmd_chunk)

    return parser


def _cmd_normalize(args: argparse.Namespace) -> int:
    normalizer = Normalizer(_config_from_args(args))
    _write_output(args, normalizer.normalize(_read_text(args)))
    return 0


def _cmd_sentences(args: argparse.Namespace) -> int:
    sentences = split_sentences(_read_text(args))
    _write_output(args, "\n".join(sentences))
    return 0


def _cmd_chunk(args: argparse.Namespace) -> int:
    chunks = chunk_text(
        _read_text(args),
        chunk_size=args.size,
        chunk_overlap=args.overlap,
        normalize=args.normalize,
    )
    if args.json:
        payload = [
            {
                "index": c.index,
                "start_char": c.start_char,
                "end_char": c.end_char,
                "text": c.text,
            }
            for c in chunks
        ]
        _write_output(args, json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        # Human-readable: each chunk preceded by a header rule.
        blocks = [f"--- chunk {c.index} [{c.start_char}:{c.end_char}] ---\n{c.text}"
                  for c in chunks]
        _write_output(args, "\n".join(blocks))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"arabic-rag-kit: error: {exc}\n")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
