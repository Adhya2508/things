"""
EDA Agent
---------
Satisfies Minimum Functional Requirement #4: "Dynamic EDA from user queries
(missing values, distributions, correlations)".

Runs on either raw cleaned data or an existing session's segmented data.
Correlation matrix is restricted to key business columns to keep the JSON
payload small and fast over HTTP.
"""

import pandas as pd


class EDAAgent:
    KEY_COLUMNS = [
        "BALANCE", "PURCHASES", "CASH_ADVANCE", "CREDIT_LIMIT", "PAYMENTS",
        "TENURE", "MONTHLY_TRX_FREQUENCY", "EFFECTIVE_BALANCE",
        "AVG_TRANSACTION_SIZE", "CREDIT_UTILIZATION",
    ]

    def run(self, df: pd.DataFrame) -> dict:
        cols = [c for c in self.KEY_COLUMNS if c in df.columns]

        return {
            "n_rows": len(df),
            "n_columns": df.shape[1],
            "missing_values": {
                c: int(v) for c, v in df.isnull().sum().items() if v > 0
            },
            "summary_stats": df[cols].describe().round(2).to_dict(),
            "correlation": df[cols].corr().round(3).to_dict(),
        }


if __name__ == "__main__":
    from clean import DataAgent
    from features import FeatureAgent

    clean_df = DataAgent().run("CC_GENERAL.csv")
    feat_df = FeatureAgent().run(clean_df)
    result = EDAAgent().run(feat_df)

    print("Rows/cols:", result["n_rows"], result["n_columns"])
    print("Missing:", result["missing_values"])
    print("Balance stats:", result["summary_stats"]["BALANCE"])