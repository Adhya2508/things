"""
Session Store
-------------
Minimal in-memory store for the hackathon. Swap the dict for a MySQL table
later if your teammate wants n8n to read/write it directly (schema noted
at the bottom).

Every pipeline run writes one record per session_id. Query Agent and
Explainability Agent read from here instead of re-running the pipeline.
"""

from datetime import datetime, timezone

_STORE: dict[str, dict] = {}


def save_session(session_id: str, result: dict) -> None:
    _STORE[session_id] = {
        **result,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def load_session(session_id: str) -> dict | None:
    return _STORE.get(session_id)


def session_exists(session_id: str) -> bool:
    return session_id in _STORE


"""
MySQL equivalent schema (for teammate, if needed):

CREATE TABLE session_results (
    session_id      VARCHAR(64) PRIMARY KEY,
    method          VARCHAR(16),          -- 'rule' or 'kmeans'
    segmented_df    LONGTEXT,             -- JSON-serialized dataframe
    rules_or_centroids LONGTEXT,          -- JSON
    eval_metrics    LONGTEXT,             -- JSON
    edge_cases      LONGTEXT,             -- JSON list of customer_ids
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""