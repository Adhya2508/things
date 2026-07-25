"""
Query Agent
-----------
Answers follow-up questions against an EXISTING session's segmented_df.
Never re-runs Data/Feature/Segmentation -- that's the whole point of the
Session Store.

Two supported actions (extend this list as your demo needs more):

  action="compare_segments"
     params: {"column": "AVG_TRANSACTION_SIZE", "stat": "mean"}
     -> matches example query: "average transaction size for priority vs regular"

  action="near_priority_candidates"
     params: {"top_n": 10}
     -> matches example query: "which regular customers can become priority"
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class QueryAgent:
    def run(self, session: dict, action: str, params: dict | None = None) -> dict:
        params = params or {}
        df = session["segmented_df"]

        if action == "compare_segments":
            return self._compare_segments(df, params)
        elif action == "near_priority_candidates":
            return self._near_priority_candidates(df, session, params)
        else:
            return {"error": f"Unknown action '{action}'. Use 'compare_segments' or 'near_priority_candidates'."}

    def _compare_segments(self, df: pd.DataFrame, params: dict) -> dict:
        column = params.get("column", "AVG_TRANSACTION_SIZE")
        stat = params.get("stat", "mean")

        if column not in df.columns:
            return {"error": f"Column '{column}' not found."}

        result = df.groupby("SEGMENT")[column].agg(stat).round(2).to_dict()
        return {"action": "compare_segments", "column": column, "stat": stat, "result": result}

    def _near_priority_candidates(self, df: pd.DataFrame, session: dict, params: dict) -> dict:
        top_n = params.get("top_n", 10)
        regular = df[df["SEGMENT"] == "Regular"].copy()

        if session["method"] == "rule":
            rules = session["rules_or_centroids"]
            bal_gap = (rules["priority_balance"] - regular["EFFECTIVE_BALANCE"]).clip(lower=0) / rules["priority_balance"]
            freq_gap = (rules["priority_freq"] - regular["MONTHLY_TRX_FREQUENCY"]).clip(lower=0) / rules["priority_freq"]
            regular["conversion_distance"] = (bal_gap + freq_gap).round(4)
        else:
            meta = session["rules_or_centroids"]
            features = meta["features"]
            centroids = np.array(meta["centroids"])
            priority_cluster_id = int(
                [k for k, v in meta["label_map"].items() if v == "Priority"][0]
            )
            priority_centroid = centroids[priority_cluster_id]

            scaler = StandardScaler().fit(df[features])
            X_scaled = scaler.transform(regular[features])
            regular["conversion_distance"] = np.round(
                np.linalg.norm(X_scaled - priority_centroid, axis=1), 4
            )

        ranked = regular.sort_values("conversion_distance").head(top_n)
        candidates = [
            {
                "customer_id": idx,
                "conversion_distance": row["conversion_distance"],
                "effective_balance": round(row["EFFECTIVE_BALANCE"], 2),
                "monthly_trx_frequency": round(row["MONTHLY_TRX_FREQUENCY"], 2),
            }
            for idx, row in ranked.iterrows()
        ]

        return {
            "action": "near_priority_candidates",
            "count": len(candidates),
            "candidates": candidates,
        }


if __name__ == "__main__":
    from clean import DataAgent
    from features import FeatureAgent
    from segment import SegmentationAgent
    from session_store import save_session, load_session

    clean_df = DataAgent().run("CC_GENERAL.csv")
    feat_df = FeatureAgent().run(clean_df)
    result = SegmentationAgent().run(feat_df, mode="rule")
    save_session("demo_query", result)
    session = load_session("demo_query")

    agent = QueryAgent()

    print("=== Compare segments (avg transaction size) ===")
    print(agent.run(session, "compare_segments", {"column": "AVG_TRANSACTION_SIZE"}))
    print()
    print("=== Near-priority candidates ===")
    out = agent.run(session, "near_priority_candidates", {"top_n": 3})
    print(out)