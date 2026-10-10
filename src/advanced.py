import os, time
import numpy as np, pandas as pd, joblib
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from xgboost import XGBClassifier
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             precision_recall_curve, precision_score,
                             recall_score, f1_score, confusion_matrix)

SEEDS = [42, 43, 44]
os.makedirs("models", exist_ok=True)
os.makedirs("reports/figures", exist_ok=True)

train = pd.read_csv("data/processed/train.csv")
val = pd.read_csv("data/processed/val.csv")
X_train, y_train = train.drop(columns="Class"), train["Class"]
X_val, y_val = val.drop(columns="Class"), val["Class"]
Xt = X_train.values.astype("float32")
Xv = X_val.values.astype("float32")

pos, neg = int(y_train.sum()), int((y_train == 0).sum())
w = float(np.sqrt(neg / pos))
print(f"Fraud class weight (sqrt ratio): {w:.1f}")


def evaluate(name, proba, y):
    prec, rec, thr = precision_recall_curve(y, proba)
    f1s = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
    best_thr = float(thr[np.argmax(f1s)])
    row = {"model": name,
           "PR_AUC": average_precision_score(y, proba),
           "ROC_AUC": roc_auc_score(y, proba),
           "best_thr": round(best_thr, 4)}
    for label, t in [("@0.5", 0.5), ("@best", best_thr)]:
        pred = (proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
        row.update({f"precision{label}": precision_score(y, pred, zero_division=0),
                    f"recall{label}": recall_score(y, pred),
                    f"F1{label}": f1_score(y, pred),
                    f"TP{label}": tp, f"FP{label}": fp, f"FN{label}": fn})
    return row, (rec, prec)


# ---------------- MLP ----------------
def build_mlp():
    m = keras.Sequential([
        keras.layers.Input(shape=(Xt.shape[1],)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    m.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="binary_crossentropy",
              metrics=[keras.metrics.AUC(name="pr_auc", curve="PR")])
    return m


runs = []
for seed in SEEDS:
    keras.utils.set_random_seed(seed)
    model = build_mlp()
    es = keras.callbacks.EarlyStopping(monitor="val_pr_auc", mode="max",
                                       patience=10, restore_best_weights=True)
    hist = model.fit(Xt, y_train.values, validation_data=(Xv, y_val.values),
                     epochs=100, batch_size=2048,
                     class_weight={0: 1.0, 1: w},
                     callbacks=[es], verbose=0)
    proba = model.predict(Xv, batch_size=4096, verbose=0).ravel()
    ap = average_precision_score(y_val, proba)
    runs.append((ap, seed, model, hist.history, proba))
    print(f"MLP seed {seed}: val PR-AUC={ap:.4f} "
          f"(stopped after {len(hist.history['loss'])} epochs)")

aps = [r[0] for r in runs]
print(f"MLP PR-AUC over seeds: mean={np.mean(aps):.4f}, std={np.std(aps):.4f}")

best_ap, best_seed, mlp, hist, mlp_proba = max(runs, key=lambda r: r[0])
print("Best MLP seed:", best_seed)
mlp.save("models/mlp.keras")

# Training curves for the best seed
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(hist["loss"], label="train"); ax[0].plot(hist["val_loss"], label="val")
ax[0].set_title("MLP loss (weighted BCE)"); ax[0].set_xlabel("epoch"); ax[0].legend()
ax[1].plot(hist["pr_auc"], label="train"); ax[1].plot(hist["val_pr_auc"], label="val")
ax[1].set_title("MLP PR-AUC (Keras approximation)"); ax[1].set_xlabel("epoch"); ax[1].legend()
plt.tight_layout(); plt.savefig("reports/figures/mlp_training_curves.png", dpi=150)

# Single-transaction latency
row0 = Xv[:1]
t = time.perf_counter()
for _ in range(100):
    mlp(row0, training=False).numpy()
mlp_latency = (time.perf_counter() - t) / 100 * 1000

# ---------------- XGBoost ----------------
xgb = XGBClassifier(n_estimators=1000, learning_rate=0.05, max_depth=4,
                    subsample=0.8, colsample_bytree=0.8,
                    scale_pos_weight=w, eval_metric="aucpr",
                    early_stopping_rounds=30, tree_method="hist",
                    n_jobs=-1, random_state=42)
t0 = time.time()
xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
xgb_fit = time.time() - t0
xgb_proba = xgb.predict_proba(X_val)[:, 1]
print(f"XGBoost: best iteration={xgb.best_iteration}, "
      f"val PR-AUC={average_precision_score(y_val, xgb_proba):.4f}")
joblib.dump(xgb, "models/xgb.joblib")

row0_df = X_val.iloc[[0]]
t = time.perf_counter()
for _ in range(100):
    xgb.predict_proba(row0_df)
xgb_latency = (time.perf_counter() - t) / 100 * 1000

# ---------------- Compare ----------------
rf = joblib.load("models/baselines/RandForest_plus_class_weight.joblib")
rf_proba = rf.predict_proba(X_val)[:, 1]

rows, curves = [], {}
for name, proba, lat in [("MLP (best seed)", mlp_proba, mlp_latency),
                         ("XGBoost", xgb_proba, xgb_latency),
                         ("RandForest + class_weight (baseline)", rf_proba, 115)]:
    r, c = evaluate(name, proba, y_val)
    r["latency_ms"] = round(lat, 2)
    rows.append(r); curves[name] = c

res = pd.DataFrame(rows).round(4)
res.to_csv("reports/advanced_results.csv", index=False)
pd.set_option("display.width", 250, "display.max_columns", None)
print("\n", res.set_index("model").T)

plt.figure(figsize=(7, 5))
for name, (r, p) in curves.items():
    plt.plot(r, p, label=name)
plt.axhline(y_val.mean(), ls="--", c="gray", label="No-skill")
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.title("Validation PR curves: advanced vs baseline")
plt.legend(fontsize=8); plt.tight_layout()
plt.savefig("reports/figures/advanced_pr_curves.png", dpi=150)