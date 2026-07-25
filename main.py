"""
main.py — FastAPI wrapper around all 7 agents.
Run with: uvicorn main:app --reload --port 8000
Then expose with: ngrok http 8000
Interactive test UI (auto-generated): http://127.0.0.1:8000/docs

============================================================================
ENDPOINTS (8 total) — this is the exact contract your teammate wires into
n8n's HTTP Request nodes / the "AI Agent" tool list.
============================================================================

  GET  /health
       -> {"status": "ok"}  (n8n pings this first to confirm the API is up)

  POST /agents/segment
       body: {"session_id": str, "mode": "rule"|"ml",
              "csv_path": str (optional, default "CC GENERAL.csv"),
              "rules": dict (optional, only used if mode="rule"),
              "k": int (optional, only used if mode="ml")}
       -> runs Data -> Feature -> Segmentation, SAVES the session,
          returns counts + eval metrics + rules/centroids

  POST /agents/eda
       body: {"session_id": str (optional), "csv_path": str (optional)}
       -> if session_id given and exists, runs EDA on that session's
          feature-engineered df; else reads csv_path fresh
       -> returns missing values, summary stats, correlation matrix

  POST /agents/explain
       body: {"session_id": str, "customer_id": str}
       -> why this customer is in their segment (rule reasons, or
          SHAP top-3 features if the session was ML-mode)

  POST /agents/query
       body: {"session_id": str, "action": "compare_segments"|"near_priority_candidates",
              "params": dict (optional, action-specific)}
       -> answers a follow-up question against an EXISTING session,
          never re-runs the pipeline

  POST /agents/recommend
       body: {"session_id": str (optional), "segment": str, "intent": "grow"|"retain",
              "customer_id": str (optional, personalizes the recommendation
              using that customer's own numbers)}
       -> cross-sell/upsell ("grow") or retention ("retain") actions

  GET  /agents/session/{session_id}
       -> session metadata: method, segment counts, eval metrics, timestamp

  GET  /agents/session/{session_id}/export
       -> downloads the full segmented dataframe as CSV
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import io
import numpy as np

from clean import DataAgent
from features import FeatureAgent
from eda import EDAAgent
from segment import SegmentationAgent
from explain import ExplainabilityAgent
from query import QueryAgent
from recommend import RecommendationAgent
from session_store import save_session, load_session, session_exists

app = FastAPI(title="Bank Segmentation Agent API")

data_agent = DataAgent()
feature_agent = FeatureAgent()
eda_agent = EDAAgent()
segmentation_agent = SegmentationAgent()
explain_agent = ExplainabilityAgent()     # single instance so its internal SHAP cache persists
query_agent = QueryAgent()
recommend_agent = RecommendationAgent()


# ---------------------------------------------------------------- schemas
class SegmentRequest(BaseModel):
    session_id: str
    mode: str = "rule"                 # "rule" or "ml"
    csv_path: str = "CC GENERAL.csv"
    rules: dict | None = None
    k: int | None = None


class EDARequest(BaseModel):
    session_id: str | None = None
    csv_path: str = "CC GENERAL.csv"


class ExplainRequest(BaseModel):
    session_id: str
    customer_id: str


class QueryRequest(BaseModel):
    session_id: str
    action: str                        # "compare_segments" | "near_priority_candidates"
    params: dict | None = None


class RecommendRequest(BaseModel):
    session_id: str | None = None
    segment: str                       # "Priority" | "Regular" | "Dormant"
    intent: str = "grow"               # "grow" | "retain"
    customer_id: str | None = None     # optional, personalizes the output


# ---------------------------------------------------------------- routes
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/agents/segment")
def run_segmentation(req: SegmentRequest):
    try:
        clean_df = data_agent.run(req.csv_path)
        feat_df = feature_agent.run(clean_df)
        result = segmentation_agent.run(
            feat_df, mode=req.mode, rules=req.rules, k=req.k
        )
        save_session(req.session_id, result)

        counts = result["segmented_df"]["SEGMENT"].value_counts().to_dict()
        return {
            "session_id": req.session_id,
            "method": result["method"],
            "counts": counts,
            "eval_metrics": result["eval_metrics"],
            "edge_case_count": len(result["edge_cases"]),
            "rules_or_centroids": _safe_serialize(result["rules_or_centroids"]),
        }
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail=f"CSV not found: {req.csv_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agents/eda")
def run_eda(req: EDARequest):
    try:
        if req.session_id and session_exists(req.session_id):
            df = load_session(req.session_id)["segmented_df"]
        else:
            clean_df = data_agent.run(req.csv_path)
            df = feature_agent.run(clean_df)

        result = eda_agent.run(df)
        return _safe_serialize(result)
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail=f"CSV not found: {req.csv_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agents/explain")
def explain_customer(req: ExplainRequest):
    if not session_exists(req.session_id):
        raise HTTPException(status_code=404, detail=f"No session '{req.session_id}'. Run /agents/segment first.")

    session = load_session(req.session_id)
    result = explain_agent.run(req.customer_id, session, session_id=req.session_id)

    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return _safe_serialize(result)


@app.post("/agents/query")
def run_query(req: QueryRequest):
    if not session_exists(req.session_id):
        raise HTTPException(status_code=404, detail=f"No session '{req.session_id}'. Run /agents/segment first.")

    session = load_session(req.session_id)
    result = query_agent.run(session, req.action, req.params)

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return _safe_serialize(result)


@app.post("/agents/recommend")
def run_recommend(req: RecommendRequest):
    customer_row = None
    if req.customer_id:
        if not req.session_id or not session_exists(req.session_id):
            raise HTTPException(status_code=404, detail=f"No session '{req.session_id}' to look up customer_id in.")
        df = load_session(req.session_id)["segmented_df"]
        if req.customer_id not in df.index:
            raise HTTPException(status_code=404, detail=f"Customer {req.customer_id} not found in session.")
        customer_row = df.loc[req.customer_id]

    result = recommend_agent.run(req.segment, intent=req.intent, customer_row=customer_row)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return _safe_serialize(result)


@app.get("/agents/session/{session_id}")
def get_session_info(session_id: str):
    if not session_exists(session_id):
        raise HTTPException(status_code=404, detail=f"No session '{session_id}'")

    session = load_session(session_id)
    counts = session["segmented_df"]["SEGMENT"].value_counts().to_dict()
    return {
        "session_id": session_id,
        "method": session["method"],
        "counts": counts,
        "eval_metrics": session["eval_metrics"],
        "timestamp": session["timestamp"],
    }


@app.get("/agents/session/{session_id}/export")
def export_session_csv(session_id: str):
    if not session_exists(session_id):
        raise HTTPException(status_code=404, detail=f"No session '{session_id}'")

    df = load_session(session_id)["segmented_df"]
    stream = io.StringIO()
    df.to_csv(stream)
    stream.seek(0)
    return StreamingResponse(
        iter([stream.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={session_id}_segmented.csv"},
    )


def _safe_serialize(obj):
    """rules_or_centroids / numpy types can't go straight to JSON; clean them up."""
    if isinstance(obj, dict):
        return {str(k): _safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe_serialize(v) for v in obj]
    if isinstance(obj, tuple):
        return [_safe_serialize(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj