"""Text-side tokenization and input formatting. The identifier vocabulary itself is
defined in ``sigil_core.ids`` so that serving code can use it without torch.

§4.2 [FIXED]: the tenant marker is part of the input string, never a separate
embedding, so one checkpoint serves every tenant in a shard.
"""

from __future__ import annotations

import re
import unicodedata

from sigil_core.ids import BASE, BOS_ID, VOCAB_SIZE, token_id  # noqa: F401  (re-exported)

_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Gateway normalization (§3.2): NFKC, casefold, collapse whitespace."""
    return _WS.sub(" ", unicodedata.normalize("NFKC", text).casefold()).strip()


def format_query(query: str, tenant_id: str | None = None) -> str:
    prefix = f"tenant: {tenant_id} | " if tenant_id else ""
    return f"{prefix}query: {normalize(query)}"


def format_document(text: str, tenant_id: str | None = None) -> str:
    """Content-prefix indexing examples (§7.2) share the encoder with queries."""
    prefix = f"tenant: {tenant_id} | " if tenant_id else ""
    return f"{prefix}document: {normalize(text)}"


def load_text_tokenizer(backbone: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(backbone)
