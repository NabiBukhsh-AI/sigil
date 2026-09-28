"""SemanticId: packing, parsing, ESCAPE handling, and the level-scoped token mapping.

§5.3 [FIXED]:

    docid := c1 . c2 . c3 . c4 . u          c_i in [0,255],  u in [0,254]
             u = 255 is ESCAPE, meaning "one more level follows"

The part after the four routing codes is the *tail*. It is ``(u,)`` normally,
``(255, e)`` after one ESCAPE, ``(255, 255, e)`` after two. Every tail ends in a byte
below 255, so a packed id is self-delimiting and its length is 5, 6, or 7 bytes.

§4.3 [FIXED]: decoder token for code ``k`` at level ``l`` is ``BASE + (l-1)*K + k``.
Level scoping means the softmax at step ``l`` cannot emit another level's code even
before masking, which makes trie masks exact.
"""

from __future__ import annotations

from dataclasses import dataclass

LEVELS = 4  # routing levels the decoder generates
K = 256  # codes per level
ESCAPE = 255
MAX_ESCAPE_DEPTH = 2
ORDINALS_PER_PREFIX = (K - 1) * (MAX_ESCAPE_DEPTH + 1)  # 765 documents per 4-level prefix

# Decoder vocabulary: 8 specials, then 5 levels x 256 codes = 1288 tokens.
PAD, BOS_ID, ESC, UNK_ROUTE = 0, 1, 2, 3  # 4..7 reserved
BASE = 8
TOKEN_LEVELS = LEVELS + 1  # the terminal ordinal has its own scoped block
VOCAB_SIZE = BASE + TOKEN_LEVELS * K


def _check_byte(value: int, what: str) -> None:
    if not 0 <= value < K:
        raise ValueError(f"{what}={value} outside [0, {K - 1}]")


@dataclass(frozen=True, slots=True, order=True)
class SemanticId:
    codes: tuple[int, ...]
    tail: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.codes) != LEVELS:
            raise ValueError(f"expected {LEVELS} routing codes, got {len(self.codes)}")
        for c in self.codes:
            _check_byte(c, "code")
        if not 1 <= len(self.tail) <= MAX_ESCAPE_DEPTH + 1:
            raise ValueError(f"tail length {len(self.tail)} outside [1, {MAX_ESCAPE_DEPTH + 1}]")
        for b in self.tail:
            _check_byte(b, "tail byte")
        # Every byte but the last must be ESCAPE; the last must not be.
        if any(b != ESCAPE for b in self.tail[:-1]) or self.tail[-1] == ESCAPE:
            raise ValueError(f"malformed ESCAPE chain {self.tail}")

    # -- constructors -----------------------------------------------------------------

    @classmethod
    def from_ordinal(cls, codes: tuple[int, ...], ordinal: int) -> SemanticId:
        """The ``ordinal``-th identifier under a prefix: 0..254, then ESCAPE levels."""
        if not 0 <= ordinal < ORDINALS_PER_PREFIX:
            raise ValueError(f"ordinal {ordinal} exceeds prefix capacity {ORDINALS_PER_PREFIX}")
        depth, u = divmod(ordinal, K - 1)
        return cls(tuple(codes), (ESCAPE,) * depth + (u,))

    @classmethod
    def unpack(cls, raw: bytes) -> SemanticId:
        return cls(tuple(raw[:LEVELS]), tuple(raw[LEVELS:]))

    @classmethod
    def parse(cls, text: str) -> SemanticId:
        """Parse the human form, ``"37.210.8.155.0"``."""
        try:
            parts = tuple(int(p) for p in text.strip().split("."))
        except ValueError as e:
            raise ValueError(f"not a semantic id: {text!r}") from e
        return cls(parts[:LEVELS], parts[LEVELS:])

    # -- views --------------------------------------------------------------------------

    @property
    def u(self) -> int:
        return self.tail[0]

    @property
    def escaped(self) -> bool:
        return len(self.tail) > 1

    @property
    def ordinal(self) -> int:
        return (len(self.tail) - 1) * (K - 1) + self.tail[-1]

    def pack(self) -> bytes:
        return bytes(self.codes + self.tail)

    def prefix(self, length: int) -> tuple[int, ...]:
        return self.codes[:length]

    def target_tokens(self) -> list[int]:
        """Decoder targets: 4 routing codes plus ``u``. ESCAPE extension bytes are not
        learned; fan-out resolves the terminal level (ADR 0002)."""
        return [token_id(level, c) for level, c in enumerate((*self.codes, self.u), start=1)]

    def __str__(self) -> str:
        return ".".join(map(str, self.codes + self.tail))


def token_id(level: int, code: int) -> int:
    if not 1 <= level <= TOKEN_LEVELS:
        raise ValueError(f"level {level} outside [1, {TOKEN_LEVELS}]")
    _check_byte(code, "code")
    return BASE + (level - 1) * K + code


def token_to_level_code(token: int) -> tuple[int, int]:
    if not BASE <= token < VOCAB_SIZE:
        raise ValueError(f"token {token} is not an identifier token")
    level, code = divmod(token - BASE, K)
    return level + 1, code
