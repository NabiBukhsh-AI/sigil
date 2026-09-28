"""Scheduled trie snapshot: build from the registry, validate, publish, notify pods. §18.3.

    python pipelines/trie_snapshot.py --config configs/serving/prod.yaml --out s3-mount/tries --once

Thin wrapper so the orchestrator (Argo) schedules the same code the trie builder service runs.
"""

from services.trie_builder.main import main

if __name__ == "__main__":
    main()
