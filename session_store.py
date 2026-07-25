"""
Session Store (MySQL-backed)
-----------------------------
Persistent session storage using MySQL. Same interface as the original
in-memory version so main.py and all agents require zero changes.

Environment variables (set in .env, loaded by docker-compose):
    DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME
"""

import os
import json
from datetime import datetime, timezone

import pymysql
import pandas as pd


def _get_conn():
    return pymysql.connect(
        host=os.getenv("DB_HOST", "127.0.0.1"),
        port=int(os.getenv("DB_PORT", 3306)),
        user=os.getenv("DB_USER", "root"),
        password=os.getenv("DB_PASSWORD", ""),
        database=os.getenv("DB_NAME", "segmentation"),
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True,
    )


def save_session(session_id: str, result: dict) -> None:
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO session_results
                    (session_id, method, segmented_df, rules_or_centroids,
                     eval_metrics, edge_cases, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    method              = VALUES(method),
                    segmented_df        = VALUES(segmented_df),
                    rules_or_centroids  = VALUES(rules_or_centroids),
                    eval_metrics        = VALUES(eval_metrics),
                    edge_cases          = VALUES(edge_cases),
                    created_at          = VALUES(created_at)
                """,
                (
                    session_id,
                    result.get("method"),
                    result["segmented_df"].to_json(orient="split"),
                    json.dumps(_to_serializable(result.get("rules_or_centroids"))),
                    json.dumps(_to_serializable(result.get("eval_metrics"))),
                    json.dumps(result.get("edge_cases", [])),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
    finally:
        conn.close()


def load_session(session_id: str) -> dict | None:
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM session_results WHERE session_id = %s",
                (session_id,),
            )
            row = cur.fetchone()
    finally:
        conn.close()

    if row is None:
        return None

    return {
        "segmented_df": pd.read_json(row["segmented_df"], orient="split"),
        "method": row["method"],
        "rules_or_centroids": json.loads(row["rules_or_centroids"]),
        "eval_metrics": json.loads(row["eval_metrics"]),
        "edge_cases": json.loads(row["edge_cases"]),
        "timestamp": row["created_at"].isoformat() if row["created_at"] else None,
    }


def session_exists(session_id: str) -> bool:
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM session_results WHERE session_id = %s LIMIT 1",
                (session_id,),
            )
            return cur.fetchone() is not None
    finally:
        conn.close()


def _to_serializable(obj):
    """Convert numpy types to native Python for JSON serialization."""
    import numpy as np
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
