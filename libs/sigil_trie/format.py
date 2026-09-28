"""Packed on-disk trie layout, format version 1.

    [8B magic "SGLTRIE1"][4B little-endian header length][JSON header][pad to 64][arrays]

The header names every array's dtype, offset, and length, so the file maps straight into
numpy views with no parsing. Arrays, CSR per depth (depth 0 is the root, depth 4 is a full
4-level prefix):

    child_off_d   uint32[n_nodes[d] + 1]   children of node i at depth d are
    child_code_d  uint8[n_nodes[d + 1]]    child_code_d[child_off_d[i] : child_off_d[i+1]]
                                            and the child's node id at depth d+1 is its
                                            position in child_code_d
    leaf_off      uint32[n_nodes[4] + 1]    tails under depth-4 node i are
    leaf_tail     uint8[n_leaves, 3]        leaf_tail[leaf_off[i] : leaf_off[i+1]]

Tails are stored in 3 bytes, zero-padded. The ESCAPE chain makes each tail self-delimiting,
so padding is unambiguous. At 10M documents this is roughly 120 to 200 MB, which is the
§1.2 footprint figure.

The file is content-addressed: its SHA-256 is its identity, recorded in a ``.sha256``
sidecar and in the bundle manifest, and verified on write and on load (§24 F16).
"""

from __future__ import annotations

MAGIC = b"SGLTRIE1"
FORMAT_VERSION = 1
ALIGN = 64
TAIL_WIDTH = 3


def array_names(levels: int) -> list[str]:
    names = []
    for d in range(levels):
        names += [f"child_off_{d}", f"child_code_{d}"]
    return names + ["leaf_off", "leaf_tail"]


def align(n: int) -> int:
    return -(-n // ALIGN) * ALIGN


def ids_digest(packed_sorted: list[bytes]) -> str:
    """Order-independent identity of an id set. The registry computes the same digest over
    its snapshot, so trie and registry agreement at publication is a string compare (§34)."""
    import hashlib

    h = hashlib.sha256()
    for raw in packed_sorted:
        h.update(len(raw).to_bytes(1, "little") + raw)
    return h.hexdigest()
