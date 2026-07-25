CREATE DATABASE IF NOT EXISTS segmentation;
USE segmentation;

CREATE TABLE IF NOT EXISTS session_results (
    session_id          VARCHAR(64) PRIMARY KEY,
    method              VARCHAR(16),
    segmented_df        LONGTEXT,
    rules_or_centroids  LONGTEXT,
    eval_metrics        LONGTEXT,
    edge_cases          LONGTEXT,
    created_at          DATETIME DEFAULT CURRENT_TIMESTAMP
);
