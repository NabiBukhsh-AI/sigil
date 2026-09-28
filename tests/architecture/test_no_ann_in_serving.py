"""ADR 0006 / §9.2 [FIXED]: no ANN library and no vector database client anywhere in the
serving call graph. Enforced by CI, not by discipline.

Walks the static import graph from every module under services/, following first-party
imports (libs/sigil_*, services.*) transitively, and collects every third-party root it
reaches. Imports inside functions count: a lazily imported FAISS is still FAISS.
"""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BANNED = {
    "faiss", "scann", "hnswlib", "annoy", "usearch", "nmslib", "voyager", "pynndescent",
    "pinecone", "weaviate", "qdrant_client", "pymilvus", "chromadb", "lancedb", "vespa",
}
FIRST_PARTY = ("sigil_", "services")


def module_file(name: str, roots: list[Path]) -> Path | None:
    parts = name.split(".")
    for root in roots:
        base = root.joinpath(*parts)
        for cand in (base.with_suffix(".py"), base / "__init__.py"):
            if cand.exists():
                return cand
    return None


def imports_of(path: Path) -> set[str]:
    out = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            out.add(node.module)
            out |= {f"{node.module}.{a.name}" for a in node.names}
    return out


def reachable_third_party(entry_dir: Path, roots: list[Path]) -> dict[str, set[str]]:
    """third-party root -> first-party modules that import it."""
    todo = list(entry_dir.rglob("*.py"))
    seen: set[Path] = set()
    found: dict[str, set[str]] = {}
    while todo:
        f = todo.pop()
        if f in seen:
            continue
        seen.add(f)
        for name in imports_of(f):
            root = name.split(".")[0]
            if root.startswith(FIRST_PARTY):
                parts = name.split(".")
                for i in range(1, len(parts) + 1):  # the module and every parent package
                    if (m := module_file(".".join(parts[:i]), roots)) is not None:
                        todo.append(m)
            else:
                found.setdefault(root, set()).add(f"{f.parent.name}/{f.name}")
    return found


def violations(entry_dir: Path, roots: list[Path]) -> dict[str, set[str]]:
    return {k: v for k, v in reachable_third_party(entry_dir, roots).items() if k in BANNED}


def test_serving_graph_has_no_vector_index():
    bad = violations(ROOT / "services", [ROOT / "libs", ROOT])
    assert not bad, f"ANN / vector DB reachable from services/ (ADR 0006): {bad}"


def test_guard_catches_a_violation_hidden_two_modules_deep(tmp_path):
    (tmp_path / "libs" / "sigil_evil").mkdir(parents=True)
    (tmp_path / "libs" / "sigil_evil" / "__init__.py").write_text("")
    (tmp_path / "libs" / "sigil_evil" / "inner.py").write_text("def f():\n    import faiss\n")
    (tmp_path / "libs" / "sigil_evil" / "outer.py").write_text("from sigil_evil.inner import f\n")
    (tmp_path / "services").mkdir()
    (tmp_path / "services" / "svc.py").write_text("import sigil_evil.outer\n")
    assert "faiss" in violations(tmp_path / "services", [tmp_path / "libs", tmp_path])


def test_offline_ann_exists_only_outside_serving():
    """The offline dense baseline is allowed, and must stay unreachable from services/."""
    assert "hnswlib" in imports_of(ROOT / "libs" / "sigil_eval" / "baselines" / "dense.py")


@pytest.mark.parametrize("svc", sorted(p.name for p in (ROOT / "services").iterdir() if p.is_dir()))
def test_each_service_is_clean(svc):
    assert not violations(ROOT / "services" / svc, [ROOT / "libs", ROOT])
