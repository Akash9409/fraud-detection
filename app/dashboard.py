import json
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Fraud Checker", layout="wide")
st.title("Credit Card Fraud Checker")
st.caption("My capstone project: a model that predicts if a transaction is fraud")

api = st.sidebar.text_input("API URL", "http://127.0.0.1:8000")
try:
    h = requests.get(f"{api}/health", timeout=3).json()
    st.sidebar.success("API is running")
    st.sidebar.write(f"Model: {h['model']}")
    st.sidebar.write(f"Threshold: {h['threshold']:.4f}")
except Exception:
    st.sidebar.error("Can't reach the API. Start it first with: uvicorn app.main:app")
    st.stop()

RAW_COLS = ["Amount"] + [f"V{i}" for i in range(1, 29)]
samples = json.loads((Path(__file__).parent / "samples/sample_payloads.json").read_text())

LABELS = {
    "sample_1_typical_legit": "Example 1 - normal transaction",
    "sample_2_caught_fraud": "Example 2 - fraud (model catches it)",
    "sample_3_borderline": "Example 3 - borderline case",
    "failure_missed_fraud": "Failure case - fraud the model misses",
    "failure_false_alarm": "Failure case - normal one flagged as fraud",
}
INFO = {
    "sample_1_typical_legit": "A normal transaction, should come out as LOW.",
    "sample_2_caught_fraud": "A real fraud with the usual V14/V17 pattern, should come out as HIGH.",
    "sample_3_borderline": "The score is almost exactly at the threshold, so it could go either way.",
    "failure_missed_fraud": "This is real fraud but it looks like a normal transaction, so the model misses it.",
    "failure_false_alarm": "This is a normal transaction but it looks like fraud, so the model flags it.",
}

tab1, tab2 = st.tabs(["One transaction", "Upload CSV"])

with tab1:
    name = st.selectbox("Pick an example", list(samples), format_func=lambda k: LABELS.get(k, k))
    s = samples[name]
    st.write(INFO.get(name, ""))
    st.write(f"Actual label: **{'FRAUD' if s['true_label'] else 'not fraud'}**")
    text = st.text_area("Transaction data (you can edit this)",
                        json.dumps(s["payload"], indent=2), height=250)
    if st.button("Check transaction"):
        try:
            r = requests.post(f"{api}/predict", json=json.loads(text), timeout=10)
        except Exception as e:
            st.error(f"Something went wrong: {e}")
        else:
            if r.status_code != 200:
                st.error(f"API error {r.status_code}: {r.text}")
            else:
                j = r.json()
                c1, c2, c3 = st.columns(3)
                c1.metric("Fraud score", j["risk_score"])
                c2.metric("Risk level", j["risk_band"])
                c3.metric("Time (ms)", j["latency_ms"])
                msg = ("Flagged as fraud" if j["flagged"] else "Not flagged") + \
                      f" (threshold = {j['threshold']})"
                {"HIGH": st.error, "ELEVATED": st.warning, "LOW": st.success}[j["risk_band"]](msg)
                if name.startswith("failure"):
                    st.warning("Heads up: this is one of the cases my model gets wrong.")

with tab2:
    st.write("Upload a CSV with the columns Amount, V1 ... V28 (original values, max 1000 rows).")
    f = st.file_uploader("Choose a CSV file", type="csv")
    if f is not None:
        df = pd.read_csv(f)
        missing = [c for c in RAW_COLS if c not in df.columns]
        if missing:
            st.error(f"These columns are missing: {missing}")
        else:
            df = df[RAW_COLS].head(1000)
            r = requests.post(f"{api}/predict/batch",
                              json={"transactions": df.to_dict("records")}, timeout=60)
            if r.status_code != 200:
                st.error(f"API error {r.status_code}: {r.text[:500]}")
            else:
                j = r.json()
                st.write(f"Checked {j['count']} transactions, "
                         f"{j['flagged_count']} flagged ({j['latency_ms']} ms)")
                out = pd.DataFrame(j["results"])[["risk_score", "flagged", "risk_band"]]
                st.dataframe(pd.concat([df.reset_index(drop=True), out], axis=1)
                             .sort_values("risk_score", ascending=False))