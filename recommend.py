"""
Recommendation Agent
---------------------
Last of the 7 agents. Satisfies:
  - Minimum Functional Requirement #5: rule-based recommendation engine
  - Minimum Functional Requirement #7: cross-sell / up-sell logic
  - Recommended Output #5: recommendations for customer retention

Two intents:
  intent="grow"    -> cross-sell / up-sell products for a segment
  intent="retain"  -> retention actions for a segment

Works at two levels:
  - segment-level (generic advice for "Priority" / "Regular" / "Dormant")
  - customer-level (personalizes the segment-level list using that
    customer's own EFFECTIVE_BALANCE / CREDIT_UTILIZATION / PAYMENT_RATIO,
    so two "Regular" customers don't always get an identical answer)

This is what Query Agent's Q4 ("which regular customers can become
priority + what should be done") calls into for the "what should be done"
half -- it reuses this agent with intent="grow" on each near-miss customer.
"""

import pandas as pd


class RecommendationAgent:

    # ---- static, rule-based playbook (Requirement #5) -------------------
    GROW_PLAYBOOK = {
        "Priority": [
            "Offer premium credit card upgrade with higher credit limit",
            "Introduce dedicated relationship manager / wealth advisory",
            "Pitch investment products (mutual funds, fixed deposits) given high idle balance",
        ],
        "Regular": [
            "Offer cashback / reward-points credit card to increase transaction frequency",
            "Promote auto-pay / bill-pay setup to build stickier usage",
            "Cross-sell a personal loan pre-approval based on payment discipline",
        ],
        "Dormant": [
            "Send reactivation offer (fee waiver, bonus reward points for next 3 transactions)",
            "Trigger low-cost SMS/email nudge campaign highlighting unused credit limit",
            "Offer a short-term low-interest EMI conversion to re-engage spending",
        ],
    }

    RETAIN_PLAYBOOK = {
        "Priority": [
            "Proactive relationship check-in before renewal / annual fee cycle",
            "Loyalty tier benefits (lounge access, concierge) to prevent attrition to competitors",
            "Monitor for early churn signals: sudden drop in balance or transaction frequency",
        ],
        "Regular": [
            "Milestone-based rewards to nudge toward Priority tier",
            "Reduce friction: fee waivers on first missed payment window",
            "Personalized nudges based on spending category to increase engagement",
        ],
        "Dormant": [
            "Win-back call/survey to understand reason for inactivity",
            "Consider account maintenance fee waiver to avoid silent attrition",
            "Last-resort: evaluate for account closure if no response after 2 campaigns",
        ],
    }

    def run(self, segment: str, intent: str = "grow",
            customer_row: pd.Series | None = None) -> dict:
        playbook = self.GROW_PLAYBOOK if intent == "grow" else self.RETAIN_PLAYBOOK
        if segment not in playbook:
            return {"error": f"Unknown segment '{segment}'. Expected Priority/Regular/Dormant."}

        recommendations = list(playbook[segment])  # copy, so personalization doesn't mutate the template

        if customer_row is not None:
            recommendations = self._personalize(recommendations, segment, intent, customer_row)

        return {
            "segment": segment,
            "intent": intent,
            "recommendations": recommendations,
        }

    def run_for_segment_summary(self, segmented_df: pd.DataFrame, intent: str = "grow") -> dict:
        """Report-Agent-friendly: one recommendation block per segment present in the df."""
        out = {}
        for seg in segmented_df["SEGMENT"].unique():
            out[seg] = self.run(seg, intent)["recommendations"]
        return out

    def _personalize(self, recs: list[str], segment: str, intent: str, row: pd.Series) -> list[str]:
        extra = []
        if "CREDIT_UTILIZATION" in row and row["CREDIT_UTILIZATION"] > 0.8:
            extra.append(
                f"High credit utilization ({row['CREDIT_UTILIZATION']:.0%}) — "
                f"consider credit-limit increase offer to ease strain and prevent churn"
            )
        if "PAYMENT_RATIO" in row and 0 < row["PAYMENT_RATIO"] < 1:
            extra.append(
                f"Payment ratio below 1 ({row['PAYMENT_RATIO']:.2f}) — "
                f"flag for EMI conversion outreach before it affects credit standing"
            )
        if "EFFECTIVE_BALANCE" in row and segment == "Regular" and row["EFFECTIVE_BALANCE"] > 0:
            extra.append(
                f"Effective balance ₹{row['EFFECTIVE_BALANCE']:.0f} — "
                f"close to Priority range, prioritize this customer in the next campaign batch"
            )
        return recs + extra


if __name__ == "__main__":
    from clean import DataAgent
    from features import FeatureAgent
    from segment import SegmentationAgent
    from session_store import save_session, load_session

    clean_df = DataAgent().run("CC GENERAL.csv")
    feat_df = FeatureAgent().run(clean_df)
    result = SegmentationAgent().run(feat_df, mode="rule")
    save_session("demo_recommend", result)
    session = load_session("demo_recommend")

    agent = RecommendationAgent()

    print("=== Segment-level: Priority, grow ===")
    print(agent.run("Priority", intent="grow"))
    print()
    print("=== Segment-level: Dormant, retain ===")
    print(agent.run("Dormant", intent="retain"))
    print()
    print("=== Customer-level personalized (Regular, grow) ===")
    sample_regular = session["segmented_df"][session["segmented_df"]["SEGMENT"] == "Regular"].iloc[0]
    print(agent.run("Regular", intent="grow", customer_row=sample_regular))
    print()
    print("=== Summary across all segments (grow) ===")
    print(agent.run_for_segment_summary(session["segmented_df"], intent="grow"))