import json, time
from pathlib import Path
from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field, create_model

BASE = Path(__file__).resolve().parent.parent
cfg = json.loads((BASE / "models/deploy_config.json").read_text())
FEATURES = json.loads((BASE / "models/feature_order.json").read_text())
preprocessor = joblib.load(BASE / "models/preprocessor.joblib")
model = joblib.load(BASE / cfg["model_file"])

THRESHOLD = cfg["threshold"]
ELEVATED = cfg["elevated_cutoff"]
RAW_COLS = ["Amount"] + [f"V{i}" for i in range(1, 29)]

sample_file = BASE / "app/samples/sample_payloads.json"
EXAMPLE = (json.loads(sample_file.read_text())["sample_2_caught_fraud"]["payload"]
           if sample_file.exists() else None)

# Input format: Amount + V1..V28
fields = {"Amount": (float, Field(..., ge=0, allow_inf_nan=False,
                                  description="Amount of the transaction (original value, not scaled)"))}
fields.update({f"V{i}": (float, Field(..., allow_inf_nan=False,
                                      description=f"PCA feature V{i} from the dataset"))
               for i in range(1, 29)})
Transaction = create_model(
    "Transaction",
    __config__=ConfigDict(extra="forbid",
                          json_schema_extra={"example": EXAMPLE} if EXAMPLE else {}),
    **fields,
)


class BatchRequest(BaseModel):
    transactions: List[Transaction] = Field(..., min_length=1, max_length=1000)


class Prediction(BaseModel):
    risk_score: float
    flagged: bool
    risk_band: str
    threshold: float


class BatchResponse(BaseModel):
    count: int
    flagged_count: int
    latency_ms: float
    results: List[Prediction]


class SingleResponse(Prediction):
    latency_ms: float


def band(score: float) -> str:
    if score >= THRESHOLD:
        return "HIGH"
    if score >= ELEVATED:
        return "ELEVATED"
    return "LOW"


def score_rows(rows: list) -> list:
    df = pd.DataFrame(rows, columns=RAW_COLS)
    X = preprocessor.transform(df)[FEATURES]
    return model.predict_proba(X)[:, 1].tolist()


def to_prediction(s: float) -> Prediction:
    return Prediction(risk_score=round(s, 5), flagged=s >= THRESHOLD,
                      risk_band=band(s), threshold=round(THRESHOLD, 5))


app = FastAPI(title="Fraud Detection API",
              version="1.0",
              description="A simple API that checks whether a credit card transaction "
                          "looks like fraud. Made for my AI/ML capstone project.")


@app.get("/")
def root():
    return {"message": "Fraud detection API is running", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok", "model": cfg["model_name"], "threshold": THRESHOLD,
            "n_features": len(RAW_COLS), "score_note": cfg["score_note"]}


@app.post("/predict", response_model=SingleResponse)
def predict(tx: Transaction):
    t = time.perf_counter()
    s = score_rows([tx.model_dump()])[0]
    ms = round((time.perf_counter() - t) * 1000, 2)
    return SingleResponse(**to_prediction(s).model_dump(), latency_ms=ms)


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(req: BatchRequest):
    t = time.perf_counter()
    scores = score_rows([x.model_dump() for x in req.transactions])
    ms = round((time.perf_counter() - t) * 1000, 2)
    preds = [to_prediction(s) for s in scores]
    return BatchResponse(count=len(preds), flagged_count=sum(p.flagged for p in preds),
                         latency_ms=ms, results=preds)