"""§33 unit: SemanticId packing, parsing, ESCAPE, token mapping."""

import pytest
from sigil_core.ids import (
    BASE,
    ESCAPE,
    ORDINALS_PER_PREFIX,
    VOCAB_SIZE,
    SemanticId,
    token_id,
    token_to_level_code,
)


def test_packs_to_five_bytes_and_round_trips():
    sid = SemanticId.parse("37.210.8.155.0")
    raw = sid.pack()
    assert raw == bytes([37, 210, 8, 155, 0])
    assert SemanticId.unpack(raw) == sid
    assert str(sid) == "37.210.8.155.0"


def test_escape_produces_six_bytes():
    sid = SemanticId.from_ordinal((1, 2, 3, 4), 255)
    assert sid.tail == (ESCAPE, 0)
    assert len(sid.pack()) == 6
    assert sid.escaped and sid.ordinal == 255


def test_ordinals_are_dense_and_bijective():
    seen = set()
    for n in range(ORDINALS_PER_PREFIX):
        sid = SemanticId.from_ordinal((0, 0, 0, 0), n)
        assert sid.ordinal == n
        seen.add(sid.pack())
    assert len(seen) == ORDINALS_PER_PREFIX
    assert SemanticId.from_ordinal((0, 0, 0, 0), 254).tail == (254,)
    with pytest.raises(ValueError):
        SemanticId.from_ordinal((0, 0, 0, 0), ORDINALS_PER_PREFIX)


@pytest.mark.parametrize(
    "bad",
    ["1.2.3.4", "1.2.3.4.255", "1.2.3.4.0.0", "1.2.3.256.0", "a.b.c.d.e", "1.2.3.4.255.255.255.0"],
)
def test_rejects_malformed(bad):
    with pytest.raises(ValueError):
        SemanticId.parse(bad)


def test_token_ids_are_level_scoped():
    assert token_id(1, 0) == BASE
    assert token_id(5, 255) == VOCAB_SIZE - 1 == 1287
    for level in range(1, 6):
        for code in (0, 17, 255):
            assert token_to_level_code(token_id(level, code)) == (level, code)
    # The same code at different levels is a different token.
    assert len({token_id(level, 7) for level in range(1, 6)}) == 5


def test_target_tokens_cover_routing_and_u():
    sid = SemanticId.parse("37.210.8.155.3")
    assert [token_to_level_code(t) for t in sid.target_tokens()] == [
        (1, 37), (2, 210), (3, 8), (4, 155), (5, 3)
    ]
