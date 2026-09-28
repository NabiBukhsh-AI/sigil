"""Compiled prefix trie: build, packed format, mmap reader, validation."""

from sigil_trie.build import write
from sigil_trie.mmap_reader import TrieSnapshot

__all__ = ["TrieSnapshot", "write"]
