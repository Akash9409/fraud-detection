import os, time
import numpy as np, pandas as pd, joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             precision_recall_curve, precision_score,
                             recall_score, f1_score, confusion_matrix)
from imblearn.over_sampling import SMOTE

SEED = 42
os.makedirs("models/baselines", exist_ok=True)
os.makedirs("reports/figures", exist_ok=True)

train = pd.read_csv("data/processed/train.csv")
val = pd.read_csv("data/processed/val.csv")
X_train, y_train = train.drop(columns="Class"), train["Class"]
X_val, y_val = val.drop(columns="Class"), val["Class"]

# SMOTE on train only
X_sm, y_sm = SMOTE(sampling_strategy=0.1, random_state=SEED).fit_resample(X_train, y_train)
print(f"Train: {len(y_train)} rows, {y_train.sum()} fraud -> "
      f"SMOTE: {len(y_sm)} rows, {int(y_sm.sum())} fraud")

def make_lr():
    return LogisticRegression(max_iter=2000, random_state=SEED)

def make_rf():
    return RandomForestClassifier(n_estimators=200, min_samples_leaf=3,
                                  n_jobs=-1, random_state=SEED)

experiments = [
    ("LogReg + class_weight", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED), X_train, y_train),
    ("LogReg + SMOTE",        make_lr(), X_sm, y_sm),
    ("RandForest + class_weight", RandomForestClassifier(n_estimators=200, min_samples_leaf=3, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED), X_train, y_train),
    ("RandForest + SMOTE",    make_rf(), X_sm, y_sm),
]

def latency_ms(model, X, n=100):
    row = X.iloc[[0]]
    t = time.perf_counter()
    for _ in range(n):
        model.predict_proba(row)
    return (time.perf_counter() - t) / n * 1000

results, curves = [], {}
for name, model, Xf, yf in experiments:
    t0 = time.time()
    model.fit(Xf, yf)
    fit_s = time.time() - t0
    proba = model.predict_proba(X_val)[:, 1]

    prec, rec, thr = precision_recall_curve(y_val, proba)
    f1s = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
    best_thr = float(thr[np.argmax(f1s)])
    curves[name] = (rec, prec)

    row = {"model": name,
           "PR_AUC": average_precision_score(y_val, proba),
           "ROC_AUC": roc_auc_score(y_val, proba),
           "fit_seconds": round(fit_s, 1),
           "latency_ms": round(latency_ms(model, X_val), 2),
           "best_thr": round(best_thr, 4)}
    for label, t in [("@0.5", 0.5), ("@best", best_thr)]:
        pred = (proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, pred).ravel()
        row.update({f"precision{label}": precision_score(y_val, pred, zero_division=0),
                    f"recall{label}": recall_score(y_val, pred),
                    f"F1{label}": f1_score(y_val, pred),
                    f"TP{label}": tp, f"FP{label}": fp, f"FN{label}": fn})
    results.append(row)
    joblib.dump(model, f"models/baselines/{name.replace(' ', '_').replace('+', 'plus')}.joblib")
    print(f"done: {name}  PR-AUC={row['PR_AUC']:.4f}")

res = pd.DataFrame(results).round(4)
res.to_csv("reports/baseline_results.csv", index=False)
pd.set_option("display.width", 250, "display.max_columns", None)
print("\n", res.T)

# PR curves
plt.figure(figsize=(7, 5))
for name, (r, p) in curves.items():
    plt.plot(r, p, label=name)
plt.axhline(y_val.mean(), ls="--", c="gray", label="No-skill (fraud rate)")
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Validation PR curves: baselines")
plt.legend(); plt.tight_layout()
plt.savefig("reports/figures/baseline_pr_curves.png", dpi=150)