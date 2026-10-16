import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check_links.py"
spec = importlib.util.spec_from_file_location("check_links", SCRIPT)
check_links = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_links)


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_valid_links_pass(tmp_path):
    _write(tmp_path, "other.md", "# Other\n\n## A heading, with `code`\n")
    _write(tmp_path, "img/x.png", "png")
    doc = _write(
        tmp_path,
        "doc.md",
        "# Title\n[a](other.md) [b](other.md#a-heading-with-code) ![i](img/x.png)\n"
        "[c](#title) [d](https://example.com/missing) [e](mailto:a@b.c)\n",
    )
    assert check_links.check_file(doc) == []


def test_broken_links_are_reported(tmp_path):
    _write(tmp_path, "other.md", "# Other\n")
    doc = _write(
        tmp_path,
        "doc.md",
        "# Title\n[a](missing.md)\n[b](other.md#nope)\n![i](missing.png)\n"
        '<img src="also-missing.png">\n[c](#nowhere)\n',
    )
    errors = check_links.check_file(doc)
    assert len(errors) == 5
    assert any("missing file: missing.md" in e for e in errors)
    assert any("missing anchor #nope" in e for e in errors)
    assert any("missing anchor #nowhere" in e for e in errors)


def test_code_is_ignored(tmp_path):
    doc = _write(
        tmp_path,
        "doc.md",
        "# T\nSee `[x](nope.md)` here.\n```\n[y](nope2.md)\n```\n",
    )
    assert check_links.check_file(doc) == []


def test_repository_docs_have_no_broken_links():
    errors = []
    for f in check_links.default_files():
        errors += check_links.check_file(f)
    assert errors == []
