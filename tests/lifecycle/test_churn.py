"""§33 corpus lifecycle: the riskiest behaviour in the system. A short Experiment 8 replay
at a moderate churn rate must hold every invariant; a high churn rate must trip the hot-set
cap rather than let BM25 quietly become the system."""

from scripts.simulate_churn import simulate


def test_moderate_churn_holds_every_invariant(tmp_path):
    r = simulate(days=4, n_docs=320, churn=0.02, workdir=tmp_path, refresh_every=2)
    assert r["violations"] == []
    # E7: fan-out makes new documents reachable without retraining. Reach is a rate over a
    # handful of adds per day, so gate the aggregate, not the noisiest day.
    reach = [(d["new_doc_generative_reach"], d["adds"]) for d in r["days"] if d["adds"]]
    assert sum(x * n for x, n in reach) / sum(n for _, n in reach) >= 0.8
    assert all(d["trie_ids"] == d["live"] for d in r["days"])
    assert all(d["adds_blocked_by_hot_cap"] == 0 for d in r["days"])
    assert r["old_doc_recall_regression_points"] <= 5.0


def test_high_churn_trips_the_hot_set_cap(tmp_path):
    r = simulate(days=5, n_docs=320, churn=0.06, workdir=tmp_path, refresh_every=100)
    assert r["violations"] == []
    assert any(d["adds_blocked_by_hot_cap"] > 0 for d in r["days"])
