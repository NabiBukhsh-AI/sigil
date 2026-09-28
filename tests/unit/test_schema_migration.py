"""The migration plan produces a consistent, verifiable set of artefacts and never executes."""

import json
import uuid

from sigil_core.config import load_yaml
from sigil_core.ids import SemanticId
from sigil_trie import TrieSnapshot

from pipelines.schema_migration import plan
from tests.helpers import clustered_embeddings


def test_plan_artefacts_agree(tmp_path):
    emb = clustered_embeddings(600)
    corpus = [{"doc_uid": str(uuid.uuid5(uuid.NAMESPACE_URL, str(i))), "semantic_id": f"{i % 7}.0.0.0.0"}
              for i in range(600)]
    cfg = load_yaml("configs/identifiers/ids_v1.yaml")
    cfg["schema_version"] = "ids_v2"
    cfg["identifiers"]["codes_per_level"] = 16
    cfg["quantizer"].update(kmeans_iters=10, kmeans_restarts=1)
    rep = plan(corpus, emb, cfg, "cb_v2.0", tmp_path / "m")
    mapping = [json.loads(x) for x in (tmp_path / "m" / "mapping.jsonl").read_text().splitlines()]
    assert len(mapping) == 600 and len({m["new"] for m in mapping}) == 600
    with TrieSnapshot(rep["trie"], rep["trie_sha256"]) as t:
        assert t.id_schema == "ids_v2" and t.n_leaves == 600
        assert all(SemanticId.parse(m["new"]) in t for m in mapping[:50])
    sql = (tmp_path / "m" / "cutover.sql").read_text()
    assert sql.startswith("--") and sql.rstrip().endswith("COMMIT;") and sql.count("UPDATE documents") == 600
    assert rep["gates"]["l1_balance_ok"]
