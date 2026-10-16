"""Check that relative links in the project's Markdown files resolve.

Usage:
    python3 scripts/check_links.py            # README.md and docs/**/*.md
    python3 scripts/check_links.py FILE...    # explicit files

For every Markdown link or image ``[text](target)`` / ``![alt](target)`` and every HTML
``src=`` / ``href=`` attribute, outside fenced code blocks and inline code:

* external targets (``http:``, ``https:``, ``mailto:``, ...) are skipped (no network access);
* the relative path (with any ``#fragment`` and ``?query`` removed) must exist, resolved
  against the directory of the file that contains the link;
* a ``#fragment`` pointing into a Markdown file must match a heading in that file
  (GitHub-style slug; ``#`` alone and ``#fragment`` within the same file are checked too).

Exits 1 and lists every broken link if there is any, 0 otherwise. Standard library only.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parent.parent

LINK_RE = re.compile(r"!?\[[^\]\n]*\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
HTML_ATTR_RE = re.compile(r"""\b(?:src|href)\s*=\s*["']([^"']+)["']""", re.IGNORECASE)
INLINE_CODE_RE = re.compile(r"`[^`\n]*`")
HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$")
EXTERNAL_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")


def default_files() -> list[Path]:
    files = [ROOT / "README.md"]
    files += sorted((ROOT / "docs").rglob("*.md"))
    return [f for f in files if f.exists()]


def slugify(heading: str) -> str:
    """GitHub-style anchor: lowercase, drop punctuation, spaces to hyphens."""
    text = re.sub(r"`([^`]*)`", r"\1", heading)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = text.strip().lower()
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


def anchors_of(path: Path, _cache: dict[Path, set[str]] = {}) -> set[str]:
    if path in _cache:
        return _cache[path]
    seen: dict[str, int] = {}
    anchors: set[str] = set()
    in_fence = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith(("```", "~~~")):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = HEADING_RE.match(line)
        if not m:
            continue
        slug = slugify(m.group(2))
        n = seen.get(slug, 0)
        seen[slug] = n + 1
        anchors.add(slug if n == 0 else f"{slug}-{n}")
    _cache[path] = anchors
    return anchors


def iter_targets(text: str):
    """Yield (line_number, target) for links outside fenced code and inline code."""
    in_fence = False
    for lineno, line in enumerate(text.splitlines(), start=1):
        if line.lstrip().startswith(("```", "~~~")):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        stripped = INLINE_CODE_RE.sub("", line)
        for m in LINK_RE.finditer(stripped):
            yield lineno, m.group(1)
        for m in HTML_ATTR_RE.finditer(stripped):
            yield lineno, m.group(1)


def check_file(path: Path) -> list[str]:
    errors: list[str] = []
    text = path.read_text(encoding="utf-8")
    rel = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
    for lineno, target in iter_targets(text):
        if EXTERNAL_RE.match(target):
            continue
        base, _, fragment = target.partition("#")
        base = unquote(base.partition("?")[0])
        dest = path if base == "" else (path.parent / base).resolve()
        if base != "" and not dest.exists():
            errors.append(f"{rel}:{lineno}: missing file: {target}")
            continue
        if fragment and dest.suffix.lower() == ".md" and dest.is_file():
            if unquote(fragment).lower() not in anchors_of(dest):
                errors.append(f"{rel}:{lineno}: missing anchor #{fragment} in {dest.name}")
    return errors


def main(argv: list[str]) -> int:
    files = [Path(a).resolve() for a in argv] if argv else default_files()
    errors: list[str] = []
    n_links = 0
    for f in files:
        n_links += sum(1 for _ in iter_targets(f.read_text(encoding="utf-8")))
        errors += check_file(f)
    for e in errors:
        print(e)
    print(f"checked {len(files)} files, {n_links} link targets, {len(errors)} broken")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
