"""Error types. Each maps to a failure mode in §24 so alerts and traces can name it."""


class SigilError(Exception):
    """Base class."""


class InvalidIdentifier(SigilError):
    """F1. A decoded identifier is not a trie path. Must never happen: valid_id_rate is 1.0
    by construction, so raising this is a P1 bug, not a recoverable condition."""


class PrefixCapacityExceeded(SigilError):
    """F20. A 4-level prefix has used every ordinal including ESCAPE levels. The quantizer
    no longer fits the corpus; refit is a schema migration."""


class BundleIncompatible(SigilError):
    """F15. §17.3 compatibility rule violated. The pod refuses to become ready."""


class SnapshotIntegrityError(SigilError):
    """F16. Trie snapshot or codebook hash mismatch. Refuse to map; keep the previous one."""


class CapExceeded(SigilError):
    """§22.4 / §27. Request exceeds a server-side budget. Maps to HTTP 429."""
