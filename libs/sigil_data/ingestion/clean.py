"""Parse, clean, quality-filter, and canonically hash a document. §3.1 I1-I3, I6."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from html.parser import HTMLParser

_WS = re.compile(r"[ \t\f\v]+")
_BLANKS = re.compile(r"\n{3,}")
_CTRL = dict.fromkeys(c for c in range(32) if c not in (9, 10))


class _Text(HTMLParser):
    SKIP = {"script", "style", "noscript", "template"}
    BLOCK = {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "section", "article"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip:
            self._skip -= 1
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def html_to_text(raw: str) -> str:
    p = _Text()
    p.feed(raw)
    return "".join(p.parts)


def clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(_CTRL)
    lines = [_WS.sub(" ", ln).strip() for ln in text.splitlines()]
    return _BLANKS.sub("\n\n", "\n".join(lines)).strip()


def content_hash(text: str) -> str:
    """Hash of the canonical form. Re-ingesting identical content is a no-op (idempotent
    by content hash, §20.2 S5)."""
    return hashlib.sha256(clean(text).casefold().encode()).hexdigest()


def quality(text: str, min_chars: int = 40, max_symbol_ratio: float = 0.5) -> tuple[bool, str]:
    if len(text) < min_chars:
        return False, "too_short"
    letters = sum(ch.isalnum() for ch in text)
    if letters / max(len(text), 1) < 1 - max_symbol_ratio:
        return False, "mostly_symbols"
    return True, "ok"
