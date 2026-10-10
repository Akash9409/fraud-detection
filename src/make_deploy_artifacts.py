import json, os
import numpy as np, pandas as pd, joblib
from sklearn.metrics import precision_recall_curve

os.makedirs("app/samples", exist_ok=True)

val = pd.read_csv("data/processed/val.csv")
test = pd.read_csv("data/processed/test.csv")
xgb = joblib.load("models/xgb_tuned.joblib")
pre = joblib.load("models/preprocessor.joblib")

Xv, yv = val.drop(columns="Class"), val["Class"].values
Xte, yte = test.drop(columns="Class"), test["Class"].values

# Threshold: same rule as Step 7 (max recall with val precision >= 0.90), val only
pv = xgb.predict_proba(Xv)[:, 1]
prec, rec, thr = precision_recall_curve(yv, pv)
ok = prec[:-1] >= 0.90
threshold = float(thr[np.argmax(np.where(ok, rec[:-1], -1))])
print(f"Deployment threshold: {threshold:.4f}  (Step 7 showed 0.1474)")

cfg = {
    "model_name": "XGBoost (tuned)",
    "model_file": "models/xgb_tuned.joblib",
    "threshold": threshold,
    "threshold_rule": "max recall with val precision >= 0.90",
    "elevated_cutoff": threshold * 0.5,
    "score_note": ("risk_score is an uncalibrated model score (trained with class "
                   "weighting), not a true fraud probability."),
}
json.dump(cfg, open("models/deploy_config.json", "w"), indent=2)

# ---- sample payloads, converted back to RAW units (what an API caller sends) ----
pt = xgb.predict_proba(Xte)[:, 1]
scaler = pre.named_transformers_["amount"].named_steps["scale"]
amount = np.clip(np.expm1(scaler.inverse_transform(Xte[["Amount"]].values)).ravel(), 0, None)
raw = Xte.copy()
raw["Amount"] = amount.round(2)
raw = raw[["Amount"] + [f"V{i}" for i in range(1, 29)]]


def payload(i):
    return {c: float(raw.iloc[i][c]) for c in raw.columns}


legit = np.where(yte == 0)[0]
fraud = np.where(yte == 1)[0]
caught = fraud[pt[fraud] >= threshold]
missed = fraud[pt[fraud] < threshold]
false_alarm = legit[pt[legit] >= threshold]


def closest(idx, target):
    return int(idx[np.argmin(np.abs(pt[idx] - target))])


picks = {
    "sample_1_typical_legit": (closest(legit, np.median(pt[legit])),
        "Typical legitimate transaction. Expected: LOW, not flagged."),
    "sample_2_caught_fraud": (closest(caught, np.median(pt[caught])),
        "Fraud with the typical V17/V14/V12 pattern. Expected: HIGH, flagged."),
    "sample_3_borderline": (int(np.argmin(np.abs(pt - threshold))),
        "Score sits right at the threshold, so the flag can go either way."),
    "failure_missed_fraud": (int(missed[np.argmin(pt[missed])]),
        "KNOWN FAILURE (false negative): real fraud that looks normal in PCA space."),
    "failure_false_alarm": (int(false_alarm[np.argmax(pt[false_alarm])]),
        "KNOWN FAILURE (false positive): legit transaction with fraud-like features."),
}

samples = {}
print(f"\n{'sample':26s} true  offline_score  flagged")
for name, (i, desc) in picks.items():
    samples[name] = {"description": desc, "true_label": int(yte[i]),
                     "offline_score": round(float(pt[i]), 4), "payload": payload(i)}
    print(f"{name:26s} {int(yte[i]):>4}  {pt[i]:>13.4f}  {pt[i] >= threshold}")

json.dump(samples, open("app/samples/sample_payloads.json", "w"), indent=2)
json.dump(samples["sample_2_caught_fraud"]["payload"],
          open("app/samples/example_request.json", "w"), indent=2)
print("\nSaved models/deploy_config.json and app/samples/*.json")