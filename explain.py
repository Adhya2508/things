"""
Explainability Agent
----------------------
Answers: "Why is customer C10001 Priority/Regular/Dormant?"

Rule mode -> states which thresholds were satisfied (exact, no ML needed)
ML mode   -> trains a lightweight RandomForest SURROGATE model that mimics
             KMeans' cluster assignment (features -> cluster_label), then
             runs SHAP's TreeExplainer on it. This is the standard technique
             for explaining unsupervised clustering: KMeans itself has no
             notion of "feature contribution," so clustering-explainability
             tooling trains a supervised proxy and explains THAT instead.
             SHAP values are computed once per session (not per customer)
             and cached, then looked up per customer_id.
"""

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier


class ExplainabilityAgent:
    def __init__(self):
        self._shap_cache = {}  # session_id -> (explainer, shap_values, feature_names)

    def run(self, customer_id: str, session: dict, session_id: str = "default") -> dict:
        df = session["segmented_df"]
        if customer_id not in df.index:
            return {"error": f"Customer {customer_id} not found in this session's segmentation."}

        row = df.loc[customer_id]
        segment = row["SEGMENT"]

        if session["method"] == "rule":
            return self._explain_rule(row, segment, session["rules_or_centroids"])
        else:
            return self._explain_ml_shap(row, segment, df, session["rules_or_centroids"], session_id)

    def _explain_rule(self, row, segment, rules) -> dict:
        reasons = []
        if segment == "Priority":
            reasons.append(
                f"Effective balance ₹{row['EFFECTIVE_BALANCE']:.0f} "
                f">= threshold ₹{rules['priority_balance']:.0f}"
            )
            reasons.append(
                f"Monthly transaction frequency {row['MONTHLY_TRX_FREQUENCY']:.2f} "
                f">= threshold {rules['priority_freq']:.2f}"
            )
        elif segment == "Dormant":
            reasons.append(
                f"Monthly transaction frequency {row['MONTHLY_TRX_FREQUENCY']:.2f} "
                f"< dormant threshold {rules['dormant_freq']:.2f}"
            )
        else:
            reasons.append(
                f"Balance and/or frequency fall between Priority and Dormant thresholds "
                f"(balance ₹{row['EFFECTIVE_BALANCE']:.0f}, "
                f"frequency {row['MONTHLY_TRX_FREQUENCY']:.2f})"
            )

        return {
            "customer_id": row.name,
            "segment": segment,
            "method": "rule",
            "reasons": reasons,
            "near_boundary": bool(row.get("NEAR_BOUNDARY", False)),
        }

    def _explain_ml_shap(self, row, segment, df, meta, session_id) -> dict:
        features = meta["features"]

        # fit (or reuse cached) surrogate + explainer for this session
        if session_id not in self._shap_cache:
            X = df[features].fillna(0)
            y = df["SEGMENT"]
            model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
            model.fit(X, y)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)  # (n_samples, n_features, n_classes)
            self._shap_cache[session_id] = {
                "model": model, "explainer": explainer,
                "shap_values": shap_values, "index": X.index, "features": features,
            }

        cache = self._shap_cache[session_id]
        row_pos = cache["index"].get_loc(row.name)
        class_idx = list(cache["model"].classes_).index(segment)

        # shap_values shape: (n_samples, n_features, n_classes) in recent SHAP versions
        sv = cache["shap_values"]
        if sv.ndim == 3:
            customer_shap = sv[row_pos, :, class_idx]
        else:  # older SHAP returns a list of per-class arrays
            customer_shap = sv[class_idx][row_pos]

        contrib = pd.Series(customer_shap, index=features).sort_values(
            key=lambda s: s.abs(), ascending=False
        )
        top_features = contrib.head(3)

        reasons = []
        for feat, val in top_features.items():
            direction = "pushed toward" if val > 0 else "pushed away from"
            reasons.append(
                f"{feat} = {row[feat]:.2f} {direction} '{segment}' "
                f"(SHAP contribution: {val:+.3f})"
            )

        return {
            "customer_id": row.name,
            "segment": segment,
            "method": "kmeans+shap",
            "cluster_id": int(row["CLUSTER"]),
            "reasons": reasons,
            "near_boundary": bool(row.get("NEAR_BOUNDARY", False)),
        }


if __name__ == "__main__":
    from clean import DataAgent
    from features import FeatureAgent
    from segment import SegmentationAgent
    from session_store import save_session, load_session

    clean_df = DataAgent().run("CC GENERAL.csv")
    feat_df = FeatureAgent().run(clean_df)

    # --- Rule mode ---
    rule_result = SegmentationAgent().run(feat_df, mode="rule")
    save_session("rule_session", rule_result)
    session = load_session("rule_session")
    sample_priority = session["segmented_df"][session["segmented_df"]["SEGMENT"] == "Priority"].index[0]
    sample_dormant = session["segmented_df"][session["segmented_df"]["SEGMENT"] == "Dormant"].index[0]

    agent = ExplainabilityAgent()
    print("=== RULE MODE: Priority customer ===")
    print(agent.run(sample_priority, session, session_id="rule_session"))
    print()
    print("=== RULE MODE: Dormant customer ===")
    print(agent.run(sample_dormant, session, session_id="rule_session"))

    # --- ML mode (SHAP) ---
    ml_result = SegmentationAgent().run(feat_df, mode="ml")
    save_session("ml_session", ml_result)
    ml_session = load_session("ml_session")
    ml_priority = ml_session["segmented_df"][ml_session["segmented_df"]["SEGMENT"] == "Priority"].index[0]

    print()
    print("=== ML MODE (SHAP): Priority customer ===")
    print(agent.run(ml_priority, ml_session, session_id="ml_session"))