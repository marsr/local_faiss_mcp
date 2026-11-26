from pathlib import Path
import sys
import types

import pytest
from packaging.requirements import Requirement

sentence_transformers = types.ModuleType("sentence_transformers")
sentence_transformers.SentenceTransformer = object
sentence_transformers.CrossEncoder = object
sys.modules.setdefault("sentence_transformers", sentence_transformers)

from local_faiss_mcp import cli
from local_faiss_mcp import document_parser


def test_collect_files_discovers_only_supported_parser_types(tmp_path):
    (tmp_path / ".mcp.json").write_text("{}", encoding="utf-8")
    (tmp_path / "data.json").write_text("{}", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("notes", encoding="utf-8")
    (tmp_path / "readme.md").write_text("# readme", encoding="utf-8")

    files = cli.collect_files([str(tmp_path)])

    assert [path.name for path in files] == ["notes.txt", "readme.md"]


def test_find_mcp_config_skips_inaccessible_home_candidate(tmp_path, monkeypatch, capsys):
    work = tmp_path / "work"
    home = tmp_path / "home"
    bad = home / ".bad_mount"
    good = home / ".good_config"
    work.mkdir()
    bad.mkdir(parents=True)
    good.mkdir()
    (good / "mcp.json").write_text('{"mcpServers": {}}', encoding="utf-8")
    monkeypatch.chdir(work)

    class FakeHome:
        def iterdir(self):
            return iter([bad, good])

        def __truediv__(self, name):
            return home / name

    monkeypatch.setattr(cli.Path, "home", classmethod(lambda cls: FakeHome()))

    original_exists = Path.exists

    def exists_with_inaccessible_candidate(path):
        if path == bad / "mcp.json":
            raise OSError(6, "Device not configured")
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", exists_with_inaccessible_candidate)

    assert cli.find_mcp_config() == good / "mcp.json"
    assert "Skipping inaccessible MCP config candidate" in capsys.readouterr().err


def test_parse_pdf_pages_warns_on_suspicious_gid_text_without_dropping_page(monkeypatch):
    class FakePage:
        def __init__(self, text):
            self.text = text

        def extract_text(self):
            return self.text

    class FakeReader:
        trailer = {"/Root": {}}
        pages = [
            FakePage("normal page text"),
            FakePage("/gid00030/gid00035/gid00032/gid00030 extracted text"),
        ]

    monkeypatch.setattr(document_parser, "PdfReader", lambda path: FakeReader())

    with pytest.warns(RuntimeWarning, match="page 2"):
        pages = document_parser.parse_pdf_pages(Path("broken-font.pdf"))

    assert pages == [
        {"text": "normal page text", "metadata": {"pdf_page": 1}},
        {
            "text": "/gid00030/gid00035/gid00032/gid00030 extracted text",
            "metadata": {"pdf_page": 2},
        },
    ]


def test_pdf_fonttools_dependency_is_declared():
    root = Path(__file__).resolve().parents[1]
    requirements = [
        Requirement(line)
        for line in (root / "requirements.txt").read_text().splitlines()
        if line and not line.startswith("#")
    ]
    assert any(req.name == "fonttools" for req in requirements)
    assert '"fonttools>=' in (root / "pyproject.toml").read_text()
