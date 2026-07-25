"""
Feature Engineering Agent
--------------------------
Takes the cleaned dataframe (from DataAgent) and derives the features that
actually answer the problem statement's example query:
    "priority / regular / dormant based on balance maintained and
     frequency of transactions"

Raw columns in CC_GENERAL are mostly usable as-is, but a few derived /
transformed features make segmentation and explainability much cleaner.
"""

import pandas as pd
import numpy as np


class FeatureAgent:
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # --- 1. Transaction frequency (combine purchase + cash advance txns) ---
        df["TOTAL_TRX_COUNT"] = df["PURCHASES_TRX"] + df["CASH_ADVANCE_TRX"]

        # Monthly frequency, normalized by tenure so a 6-month customer isn't
        # unfairly compared to a 12-month customer
        df["MONTHLY_TRX_FREQUENCY"] = df["TOTAL_TRX_COUNT"] / df["TENURE"].replace(0, np.nan)
        df["MONTHLY_TRX_FREQUENCY"] = df["MONTHLY_TRX_FREQUENCY"].fillna(0)

        # --- 2. Average transaction size (avoid divide-by-zero) ---
        df["AVG_TRANSACTION_SIZE"] = np.where(
            df["PURCHASES_TRX"] > 0,
            df["PURCHASES"] / df["PURCHASES_TRX"],
            0
        )

        # --- 3. Balance behavior ---
        # BALANCE is already a snapshot; BALANCE_FREQUENCY tells us how
        # consistently it's updated/maintained -> combine into one signal
        df["EFFECTIVE_BALANCE"] = df["BALANCE"] * df["BALANCE_FREQUENCY"]

        # --- 4. Credit utilization (behavioral risk / spending power proxy) ---
        df["CREDIT_UTILIZATION"] = np.where(
            df["CREDIT_LIMIT"] > 0,
            df["BALANCE"] / df["CREDIT_LIMIT"],
            0
        )

        # --- 5. Payment discipline (higher = healthier customer) ---
        df["PAYMENT_RATIO"] = np.where(
            df["MINIMUM_PAYMENTS"] > 0,
            df["PAYMENTS"] / df["MINIMUM_PAYMENTS"],
            0
        )

        # --- 6. Activity flag (dormancy signal) ---
        df["IS_DORMANT"] = (
            (df["PURCHASES"] == 0) &
            (df["CASH_ADVANCE"] == 0) &
            (df["TOTAL_TRX_COUNT"] == 0)
        ).astype(int)

        # --- 7. Log-transform heavily skewed monetary columns ---
        # (raw columns kept too, so Explainability Agent can still show
        #  real rupee values to the user, not log values)
        for col in ["BALANCE", "PURCHASES", "CASH_ADVANCE", "PAYMENTS"]:
            df[f"{col}_LOG"] = np.log1p(df[col])

        return df

    def feature_summary(self, df: pd.DataFrame) -> dict:
        """Quick reference for the Report Agent."""
        return {
            "n_dormant_customers": int(df["IS_DORMANT"].sum()),
            "avg_monthly_trx_frequency": round(df["MONTHLY_TRX_FREQUENCY"].mean(), 2),
            "avg_transaction_size": round(df["AVG_TRANSACTION_SIZE"].mean(), 2),
            "avg_credit_utilization": round(df["CREDIT_UTILIZATION"].mean(), 2),
        }


if __name__ == "__main__":
    from clean import DataAgent

    clean_df = DataAgent().run("CC GENERAL.csv")
    feat_df = FeatureAgent().run(clean_df)

    key_cols = [
        "BALANCE", "EFFECTIVE_BALANCE", "MONTHLY_TRX_FREQUENCY",
        "AVG_TRANSACTION_SIZE", "CREDIT_UTILIZATION", "IS_DORMANT"
    ]
    print(feat_df[key_cols].head())
    print()
    print(FeatureAgent().feature_summary(feat_df))