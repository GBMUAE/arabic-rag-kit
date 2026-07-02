"""Tests for the arabic_rag_kit command-line interface."""

import io
import json

import pytest

from arabic_rag_kit import __version__
from arabic_rag_kit.cli import main


def run(argv, capsys):
    """Run the CLI, returning (exit_code, stdout)."""
    code = main(argv)
    out = capsys.readouterr().out
    return code, out


def test_normalize_positional(capsys):
    code, out = run(["normalize", "الْعَرَبِيَّةُ ١٢٣ كتـــاب"], capsys)
    assert code == 0
    assert out.strip() == "العربية 123 كتاب"


def test_normalize_from_stdin(capsys, monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO("أحمد إلى المدرسة"))
    code, out = run(["normalize"], capsys)
    assert code == 0
    assert out.strip() == "احمد الى المدرسة"


def test_normalize_optional_folds(capsys):
    code, out = run(
        ["normalize", "مؤسسة على", "--hamza", "--ta-marbuta", "--alef-maqsura"],
        capsys,
    )
    assert out.strip() == "موسسه علي"


def test_normalize_disable_default_step(capsys):
    # --no-diacritics keeps the tashkeel.
    code, out = run(["normalize", "مُحَمَّد", "--no-diacritics"], capsys)
    assert out.strip() == "مُحَمَّد"


def test_sentences(capsys):
    code, out = run(["sentences", "جملة أولى. جملة ثانية؟ ثالثة"], capsys)
    assert code == 0
    assert out.strip().splitlines() == ["جملة أولى.", "جملة ثانية؟", "ثالثة"]


def test_chunk_plain_output(capsys):
    text = "جملة اولى. جملة ثانية. جملة ثالثة."
    code, out = run(["chunk", text, "--size", "20", "--overlap", "5"], capsys)
    assert code == 0
    assert "--- chunk 0 [" in out
    assert "جملة اولى." in out


def test_chunk_json_output(capsys):
    text = "جملة اولى. جملة ثانية. جملة ثالثة."
    code, out = run(["chunk", text, "--size", "20", "--overlap", "5", "--json"], capsys)
    assert code == 0
    payload = json.loads(out)
    assert isinstance(payload, list)
    assert payload[0]["index"] == 0
    assert {"index", "start_char", "end_char", "text"} <= set(payload[0])
    # Offsets are internally consistent.
    for item in payload:
        assert item["end_char"] > item["start_char"]


def test_chunk_normalize_flag(capsys):
    code, out = run(["chunk", "الْعَرَبِيَّةُ ١٢٣.", "--normalize", "--json"], capsys)
    payload = json.loads(out)
    assert payload[0]["text"] == "العربية 123."


def test_input_and_output_files(tmp_path, capsys):
    src = tmp_path / "in.txt"
    dst = tmp_path / "out.txt"
    src.write_text("كتـــاب ٩", encoding="utf-8")
    code, _ = run(["normalize", "-i", str(src), "-o", str(dst)], capsys)
    assert code == 0
    assert dst.read_text(encoding="utf-8").strip() == "كتاب 9"


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_no_command_errors(capsys):
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code != 0


def test_invalid_chunk_params_exit_code(capsys):
    # chunk_overlap >= chunk_size raises ValueError -> handled as exit code 2.
    with pytest.raises(SystemExit) as exc:
        main(["chunk", "نص", "--size", "10", "--overlap", "10"])
    assert exc.value.code == 2
    assert "error" in capsys.readouterr().err
