"""
Segmentation Agent
-------------------
Two modes, matching the problem statement's requirement to support
"rule-based OR ML-based clustering":

  mode="rule" -> uses BALANCE + MONTHLY_TRX_FREQUENCY thresholds to assign
                 priority / regular / dormant (matches the exact example
                 query in the problem statement)
  mode="ml"   -> KMeans with automatic k selection via silhouette score

Both modes attach:
  - a per-customer boundary/edge-case flag (required: "identify edge cases")
  - evaluation metrics (silhouette, Davies-Bouldin, Calinski-Harabasz)
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


class SegmentationAgent:

    RULE_FEATURES = ["EFFECTIVE_BALANCE", "MONTHLY_TRX_FREQUENCY"]
    ML_FEATURES = [
        "BALANCE_LOG", "MONTHLY_TRX_FREQUENCY", "AVG_TRANSACTION_SIZE",
        "CREDIT_UTILIZATION", "PAYMENT_RATIO", "PURCHASES_LOG", "CASH_ADVANCE_LOG"
    ]

    def run(self, df: pd.DataFrame, mode: str = "rule",
            rules: dict | None = None, k: int | None = None) -> dict:
        if mode == "rule":
            return self._run_rule(df, rules)
        elif mode == "ml":
            return self._run_ml(df, k)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    # ---------------------------------------------------------------- RULE
    def _run_rule(self, df: pd.DataFrame, rules: dict | None) -> dict:
        df = df.copy()

        # defaults derived from the dataset's own distribution, so they're
        # not arbitrary magic numbers -- but caller can override via `rules`
        rules = rules or {
            "priority_balance": df["EFFECTIVE_BALANCE"].quantile(0.75),
            "priority_freq": df["MONTHLY_TRX_FREQUENCY"].quantile(0.75),
            "dormant_freq": 0.1,
        }

        def classify(row):
            if (row["EFFECTIVE_BALANCE"] >= rules["priority_balance"] and
                    row["MONTHLY_TRX_FREQUENCY"] >= rules["priority_freq"]):
                return "Priority"
            elif row["MONTHLY_TRX_FREQUENCY"] < rules["dormant_freq"]:
                return "Dormant"
            else:
                return "Regular"

        df["SEGMENT"] = df.apply(classify, axis=1)

        # --- edge case flagging: within 10% margin of a threshold ---
        bal_margin = 0.10 * rules["priority_balance"]
        freq_margin = 0.10 * rules["priority_freq"]
        df["NEAR_BOUNDARY"] = (
            (df["EFFECTIVE_BALANCE"].between(
                rules["priority_balance"] - bal_margin,
                rules["priority_balance"] + bal_margin)) |
            (df["MONTHLY_TRX_FREQUENCY"].between(
                rules["priority_freq"] - freq_margin,
                rules["priority_freq"] + freq_margin))
        )

        metrics = self._evaluate(df, self.RULE_FEATURES, df["SEGMENT"])

        return {
            "segmented_df": df,
            "method": "rule",
            "rules_or_centroids": rules,
            "eval_metrics": metrics,
            "edge_cases": df[df["NEAR_BOUNDARY"]].index.tolist(),
        }

    # ------------------------------------------------------------------ ML
    def _run_ml(self, df: pd.DataFrame, k: int | None) -> dict:
        df = df.copy()
        X = df[self.ML_FEATURES].fillna(0)
        X_scaled = StandardScaler().fit_transform(X)

        if k is None:
            k = self._find_optimal_k(X_scaled)

        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_ids = model.fit_predict(X_scaled)
        df["CLUSTER"] = cluster_ids

        # rank clusters by effective balance -> map to business labels
        cluster_order = (
            df.groupby("CLUSTER")["EFFECTIVE_BALANCE"].mean()
            .sort_values(ascending=False).index.tolist()
        )
        label_map = {cluster_order[0]: "Priority", cluster_order[-1]: "Dormant"}
        for cid in cluster_order[1:-1]:
            label_map[cid] = "Regular"
        df["SEGMENT"] = df["CLUSTER"].map(label_map)

        # distance to own centroid vs distance to nearest other centroid
        centroids = model.cluster_centers_
        dists = np.linalg.norm(X_scaled[:, None, :] - centroids[None, :, :], axis=2)
        own_dist = dists[np.arange(len(df)), cluster_ids]
        sorted_dists = np.sort(dists, axis=1)
        second_nearest = sorted_dists[:, 1]
        df["NEAR_BOUNDARY"] = (second_nearest - own_dist) < (0.15 * own_dist.mean())

        metrics = self._evaluate(df, self.ML_FEATURES, cluster_ids, X_scaled)
        metrics["k"] = k

        return {
            "segmented_df": df,
            "method": "kmeans",
            "rules_or_centroids": {
                "centroids": centroids.tolist(),
                "features": self.ML_FEATURES,
                "label_map": {str(k_): v for k_, v in label_map.items()},
            },
            "eval_metrics": metrics,
            "edge_cases": df[df["NEAR_BOUNDARY"]].index.tolist(),
        }

    def _find_optimal_k(self, X_scaled, k_range=range(2, 7)) -> int:
        best_k, best_score = 2, -1
        for k in k_range:
            labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)
            if score > best_score:
                best_k, best_score = k, score
        return best_k

    def _evaluate(self, df, features, labels, X_scaled=None) -> dict:
        if X_scaled is None:
            X_scaled = StandardScaler().fit_transform(df[features].fillna(0))
        # metrics need numeric label encoding
        label_codes = pd.factorize(labels)[0]
        if len(set(label_codes)) < 2:
            return {"silhouette": None, "davies_bouldin": None, "calinski_harabasz": None}
        return {
            "silhouette": round(float(silhouette_score(X_scaled, label_codes)), 4),
            "davies_bouldin": round(float(davies_bouldin_score(X_scaled, label_codes)), 4),
            "calinski_harabasz": round(float(calinski_harabasz_score(X_scaled, label_codes)), 4),
        }


if __name__ == "__main__":
    from clean import DataAgent
    from features import FeatureAgent

    clean_df = DataAgent().run("CC GENERAL.csv")
    feat_df = FeatureAgent().run(clean_df)

    print("=== RULE MODE ===")
    rule_result = SegmentationAgent().run(feat_df, mode="rule")
    print(rule_result["segmented_df"]["SEGMENT"].value_counts())
    print("Eval metrics:", rule_result["eval_metrics"])
    print("Edge cases:", len(rule_result["edge_cases"]))
    print()

    print("=== ML MODE ===")
    ml_result = SegmentationAgent().run(feat_df, mode="ml")
    print(ml_result["segmented_df"]["SEGMENT"].value_counts())
    print("Chosen k:", ml_result["eval_metrics"]["k"])
    print("Eval metrics:", ml_result["eval_metrics"])
    print("Edge cases:", len(ml_result["edge_cases"]))