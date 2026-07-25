"""
Data Agent
----------
Responsibilities (per architecture doc):
  - read CSV
  - fix datatypes
  - handle missing values
  - drop duplicates
  - flag outliers (does NOT transform them — that's Feature Agent's job)

Output: clean dataframe, indexed by CUST_ID
"""

import pandas as pd
import numpy as np


class DataAgent:
    def run(self, source: str) -> pd.DataFrame:
        df = pd.read_csv(source)

        # 1. CUST_ID is an identifier, not a feature -> set as index
        df = df.set_index("CUST_ID")

        # 2. Duplicates
        before = len(df)
        df = df.drop_duplicates()
        dropped = before - len(df)
        if dropped:
            print(f"[DataAgent] Dropped {dropped} duplicate rows")

        # 3. Missing values
        #    MINIMUM_PAYMENTS: missing likely means no minimum was due
        #    -> median imputation is safer than 0 (0 would distort low-balance
        #       customers who legitimately have small minimums)
        if df["MINIMUM_PAYMENTS"].isnull().sum() > 0:
            median_val = df["MINIMUM_PAYMENTS"].median()
            df["MINIMUM_PAYMENTS"] = df["MINIMUM_PAYMENTS"].fillna(median_val)

        if df["CREDIT_LIMIT"].isnull().sum() > 0:
            df["CREDIT_LIMIT"] = df["CREDIT_LIMIT"].fillna(df["CREDIT_LIMIT"].median())

        # 4. Dtype sanity check
        numeric_cols = df.columns.tolist()
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        # any coercion failures become NaN -> fill with median as a safety net
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

        # 5. Outlier flagging (IQR method) -> adds a boolean column per key
        #    field, does not remove/clip rows here
        for col in ["BALANCE", "PURCHASES", "CASH_ADVANCE", "PAYMENTS"]:
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            df[f"{col}_OUTLIER"] = (df[col] < lower) | (df[col] > upper)

        return df

    def report(self, df: pd.DataFrame) -> dict:
        """Quick summary the EDA Agent or Report Agent can reuse."""
        outlier_cols = [c for c in df.columns if c.endswith("_OUTLIER")]
        return {
            "n_customers": len(df),
            "n_features": df.shape[1] - len(outlier_cols),
            "missing_after_clean": int(df.isnull().sum().sum()),
            "outlier_counts": {c: int(df[c].sum()) for c in outlier_cols},
        }


if __name__ == "__main__":
    agent = DataAgent()
    clean_df = agent.run("CC GENERAL.csv")
    print(clean_df.head())
    print()
    print(agent.report(clean_df))