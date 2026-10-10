import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)
samples = json.loads((ROOT / "app/samples/sample_payloads.json").read_text())

r = client.get("/health")
assert r.status_code == 200 and r.json()["status"] == "ok"
print("health OK:", r.json()["model"], "| threshold", round(r.json()["threshold"], 4))

print(f"\n{'sample':26s} true  offline  api_score  flagged  band")
results = {}
for name, s in samples.items():
    r = client.post("/predict", json=s["payload"])
    assert r.status_code == 200, r.text
    j = r.json()
    results[name] = j
    print(f"{name:26s} {s['true_label']:>4}  {s['offline_score']:>7}  "
          f"{j['risk_score']:>9}  {str(j['flagged']):7s}  {j['risk_band']}  ({j['latency_ms']} ms)")

# Behaviour checks (borderline sample is deliberately NOT asserted)
assert not results["sample_1_typical_legit"]["flagged"]
assert results["sample_2_caught_fraud"]["flagged"]
assert not results["failure_missed_fraud"]["flagged"]     # known false negative
assert results["failure_false_alarm"]["flagged"]          # known false positive

# Validation checks
bad = dict(samples["sample_1_typical_legit"]["payload"])
del bad["V5"]
assert client.post("/predict", json=bad).status_code == 422
neg = dict(samples["sample_1_typical_legit"]["payload"]); neg["Amount"] = -5
assert client.post("/predict", json=neg).status_code == 422
extra = dict(samples["sample_1_typical_legit"]["payload"]); extra["Hour"] = 3
assert client.post("/predict", json=extra).status_code == 422
print("\nvalidation OK (missing field, negative amount, extra field -> 422)")

# Batch
batch = {"transactions": [s["payload"] for s in samples.values()]}
r = client.post("/predict/batch", json=batch)
assert r.status_code == 200 and r.json()["count"] == len(samples)
print("batch OK:", r.json()["count"], "scored,", r.json()["flagged_count"], "flagged,",
      r.json()["latency_ms"], "ms")
print("\nALL CHECKS PASSED")