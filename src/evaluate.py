import os, json, time
import numpy as np, pandas as pd, joblib
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             precision_recall_curve, confusion_matrix,
                             ConfusionMatrixDisplay)

os.makedirs("reports/figures", exist_ok=True)

val = pd.read_csv("data/processed/val.csv")
test = pd.read_csv("data/processed/test.csv")
Xv, yv = val.drop(columns="Class"), val["Class"].values
Xte, yte = test.drop(columns="Class"), test["Class"].values

# ---------- load models ----------
mlps = [keras.models.load_model(f"models/mlp_tuned_seed{s}.keras") for s in (42, 43, 44)]
xgb = joblib.load("models/xgb_tuned.joblib")
rf_tuned = joblib.load("models/rf_tuned.joblib")
rf_base = joblib.load("models/baselines/RandForest_plus_class_weight.joblib")
thr_json = json.load(open("models/threshold.json"))


def mlp_proba(X):
    a = X.values.astype("float32")
    return np.mean([m.predict(a, batch_size=4096, verbose=0).ravel() for m in mlps], axis=0)


def mlp_fast(X):  # direct call: lower overhead for single rows
    a = X.values.astype("float32")
    return np.mean([m(a, training=False).numpy().ravel() for m in mlps], axis=0)


predictors = {
    "MLP (3-seed ensemble)": mlp_proba,
    "XGBoost (tuned)": lambda X: xgb.predict_proba(X)[:, 1],
    "RandForest (tuned)": lambda X: rf_tuned.predict_proba(X)[:, 1],
    "RandForest (baseline)": lambda X: rf_base.predict_proba(X)[:, 1],
}
CHAMP = "MLP (3-seed ensemble)"


def thr_rule(y, p, min_prec=0.90):
    prec, rec, thr = precision_recall_curve(y, p)
    ok = prec[:-1] >= min_prec
    if not ok.any():
        return 0.5
    return float(thr[np.argmax(np.where(ok, rec[:-1], -1))])


# ---------- probabilities + frozen thresholds ----------
pv = {n: f(Xv) for n, f in predictors.items()}
pt = {n: f(Xte) for n, f in predictors.items()}
thr = {n: thr_rule(yv, pv[n]) for n in predictors}
thr[CHAMP] = thr_json["threshold"]          # frozen in Step 6
print("Frozen thresholds (from val):", {k: round(v, 4) for k, v in thr.items()})

# ---------- paired bootstrap on PR-AUC ----------
rng = np.random.default_rng(0)
B = 1000
boot = {n: [] for n in predictors}
for _ in range(B):
    i = rng.integers(0, len(yte), len(yte))
    if yte[i].sum() == 0:
        continue
    for n in predictors:
        boot[n].append(average_precision_score(yte[i], pt[n][i]))
boot = {n: np.array(v) for n, v in boot.items()}

# ---------- metrics table ----------
rows = []
for n in predictors:
    pred = (pt[n] >= thr[n]).astype(int)
    tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
    prec, rec = tp / max(tp + fp, 1), tp / (tp + fn)
    lo, hi = np.percentile(boot[n], [2.5, 97.5])
    diff = boot[n] - boot[CHAMP]
    d_lo, d_hi = np.percentile(diff, [2.5, 97.5])
    rows.append({"model": n,
                 "PR_AUC": average_precision_score(yte, pt[n]),
                 "PR_AUC_95CI": f"[{lo:.3f}, {hi:.3f}]",
                 "diff_vs_MLP_95CI": "-" if n == CHAMP else f"[{d_lo:+.3f}, {d_hi:+.3f}]",
                 "ROC_AUC": roc_auc_score(yte, pt[n]),
                 "threshold": thr[n], "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                 "precision": prec, "recall": rec,
                 "F1": 2 * prec * rec / (prec + rec + 1e-12)})
res = pd.DataFrame(rows)
res.round(4).to_csv("reports/test_results.csv", index=False)
pd.set_option("display.width", 250, "display.max_columns", None)
print("\n=== TEST RESULTS ===")
print(res.round(4).set_index("model").T)

# ---------- error analysis (champion) ----------
p = pt[CHAMP]
pred = (p >= thr[CHAMP]).astype(int)
pre = joblib.load("models/preprocessor.joblib")
scaler = pre.named_transformers_["amount"].named_steps["scale"]
raw_amount = np.expm1(scaler.inverse_transform(Xte[["Amount"]].values)).ravel()

ea = Xte.copy()
ea["Amount_raw"] = raw_amount.round(2)
ea["proba"] = p
ea["Class"] = yte
fn_df = ea[(yte == 1) & (pred == 0)].sort_values("proba")
tp_df = ea[(yte == 1) & (pred == 1)]
fp_df = ea[(yte == 0) & (pred == 1)].sort_values("proba", ascending=False)
leg_df = ea[yte == 0]

print(f"\n=== ERROR ANALYSIS ({CHAMP}) ===")
print(f"Missed frauds: {len(fn_df)} | caught: {len(tp_df)} | false alarms: {len(fp_df)}")
print("\nMissed-fraud probabilities (how confident was the model they were legit?):")
print(fn_df["proba"].round(3).tolist())
cols = ["Amount_raw", "V17", "V14", "V12", "V10", "V4", "V11"]
print("\nMedian feature values:")
print(pd.DataFrame({"missed fraud": fn_df[cols].median(),
                    "caught fraud": tp_df[cols].median(),
                    "legit": leg_df[cols].median()}).round(2))
print("\nFalse alarms (legit flagged as fraud):")
print(fp_df[["Amount_raw", "proba", "V17", "V14"]].round(3).to_string())
fn_df.to_csv("reports/error_analysis_missed_frauds.csv")

fn_sets = {n: set(np.where((yte == 1) & (pt[n] < thr[n]))[0]) for n in predictors}
common = set.intersection(*fn_sets.values())
print("\nMissed per model:", {n: len(s) for n, s in fn_sets.items()})
print(f"Frauds missed by ALL four models: {len(common)} of {int(yte.sum())}")

# ---------- latency ----------
def lat(fn, X, n=100):
    row = X.iloc[[0]]
    fn(row)
    t = time.perf_counter()
    for _ in range(n):
        fn(row)
    return (time.perf_counter() - t) / n * 1000

batch = Xte.iloc[:1000]
lat_fns = {"MLP (3-seed ensemble)": mlp_fast, **{k: v for k, v in predictors.items() if k != CHAMP}}
print("\n=== LATENCY ===")
lat_rows = []
for n, f in lat_fns.items():
    single = lat(f, Xte)
    t = time.perf_counter(); f(batch); b = time.perf_counter() - t
    lat_rows.append({"model": n, "single_row_ms": round(single, 2),
                     "rows_per_sec_batch1000": int(1000 / b)})
print(pd.DataFrame(lat_rows).to_string(index=False))
pd.DataFrame(lat_rows).to_csv("reports/latency.csv", index=False)

# ---------- plots ----------
tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
ConfusionMatrixDisplay(np.array([[tn, fp], [fn, tp]]),
                       display_labels=["Legit", "Fraud"]).plot(cmap="Blues", values_format="d")
plt.title(f"Test confusion matrix\n{CHAMP}, thr={thr[CHAMP]:.3f}")
plt.tight_layout(); plt.savefig("reports/figures/test_confusion_matrix.png", dpi=150); plt.close()

plt.figure(figsize=(7, 5))
for n in predictors:
    pr, rc, _ = precision_recall_curve(yte, pt[n])
    plt.plot(rc, pr, label=f"{n} (AP={average_precision_score(yte, pt[n]):.3f})")
plt.axhline(yte.mean(), ls="--", c="gray", label="No-skill")
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("TEST PR curves")
plt.legend(fontsize=8); plt.tight_layout()
plt.savefig("reports/figures/test_pr_curves.png", dpi=150); plt.close()

pr, rc, th = precision_recall_curve(yte, p)
plt.figure(figsize=(7, 4))
plt.plot(th, pr[:-1], label="precision"); plt.plot(th, rc[:-1], label="recall")
plt.axvline(thr[CHAMP], ls="--", c="red", label=f"frozen thr={thr[CHAMP]:.3f}")
plt.xlabel("Threshold"); plt.title(f"Threshold trade-off on test ({CHAMP})")
plt.legend(); plt.tight_layout()
plt.savefig("reports/figures/test_threshold_tradeoff.png", dpi=150); plt.close()
print("\nSaved: reports/test_results.csv, error_analysis_missed_frauds.csv, latency.csv + 3 figures")