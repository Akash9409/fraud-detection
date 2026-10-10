import sys, os, json, time, random, datetime
import numpy as np, pandas as pd, joblib
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             precision_recall_curve, confusion_matrix)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)
LOG = "reports/experiments.csv"

train = pd.read_csv("data/processed/train.csv")
val = pd.read_csv("data/processed/val.csv")
X_train, y_train = train.drop(columns="Class"), train["Class"]
X_val, y_val = val.drop(columns="Class"), val["Class"]


def log(model, config, seed, metric_name, value, **extra):
    row = {"time": datetime.datetime.now().isoformat(timespec="seconds"),
           "model": model, "config": json.dumps(config, default=str),
           "seed": seed, metric_name: round(float(value), 4), **extra}
    pd.DataFrame([row]).to_csv(LOG, mode="a", header=not os.path.exists(LOG), index=False)


# ---------------- XGBoost ----------------
def run_xgb():
    from xgboost import XGBClassifier
    base = XGBClassifier(tree_method="hist", eval_metric="aucpr",
                         n_jobs=-1, random_state=42)
    params = dict(n_estimators=[200, 400, 600], max_depth=[3, 5, 7],
                  learning_rate=[0.03, 0.05, 0.1],
                  scale_pos_weight=[1, 5, 24.5],
                  subsample=[0.7, 0.9], colsample_bytree=[0.6, 0.8, 1.0],
                  min_child_weight=[1, 3, 5])
    search = RandomizedSearchCV(
        base, params, n_iter=15, scoring="average_precision",
        cv=StratifiedKFold(3, shuffle=True, random_state=42),
        random_state=42, refit=True, verbose=1)
    search.fit(X_train, y_train)

    cv = search.cv_results_
    for p, m, s in zip(cv["params"], cv["mean_test_score"], cv["std_test_score"]):
        log("XGBoost", p, "cv3", "cv_pr_auc", m, cv_std=round(float(s), 4))

    best = search.best_estimator_
    proba = best.predict_proba(X_val)[:, 1]
    ap = average_precision_score(y_val, proba)
    log("XGBoost (best, refit on train)", search.best_params_, 42,
        "val_pr_auc", ap, val_roc_auc=round(roc_auc_score(y_val, proba), 4))
    print("Best params:", search.best_params_)
    print(f"Best CV PR-AUC={search.best_score_:.4f} | val PR-AUC={ap:.4f}")
    joblib.dump(best, "models/xgb_tuned.joblib")
    np.save("reports/val_proba_xgb.npy", proba)


# ---------------- Random Forest ----------------
def run_rf():
    from sklearn.ensemble import RandomForestClassifier
    base = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    params = dict(min_samples_leaf=[1, 3, 5], max_features=["sqrt", 0.3],
                  max_depth=[None, 20],
                  class_weight=["balanced_subsample", {0: 1, 1: 10}, {0: 1, 1: 50}])
    search = RandomizedSearchCV(
        base, params, n_iter=6, scoring="average_precision",
        cv=StratifiedKFold(3, shuffle=True, random_state=42),
        random_state=42, refit=False, verbose=1)
    search.fit(X_train, y_train)

    cv = search.cv_results_
    for p, m, s in zip(cv["params"], cv["mean_test_score"], cv["std_test_score"]):
        log("RandomForest", p, "cv3", "cv_pr_auc", m, cv_std=round(float(s), 4))

    best = RandomForestClassifier(**{**search.best_params_, "n_estimators": 200,
                                     "n_jobs": -1, "random_state": 42})
    best.fit(X_train, y_train)
    proba = best.predict_proba(X_val)[:, 1]
    ap = average_precision_score(y_val, proba)
    log("RandomForest (best, refit on train)", search.best_params_, 42,
        "val_pr_auc", ap, val_roc_auc=round(roc_auc_score(y_val, proba), 4))
    print("Best params:", search.best_params_)
    print(f"Best CV PR-AUC={search.best_score_:.4f} | val PR-AUC={ap:.4f}")
    joblib.dump(best, "models/rf_tuned.joblib")
    np.save("reports/val_proba_rf.npy", proba)


# ---------------- MLP ----------------
def run_mlp():
    from tensorflow import keras
    Xt = X_train.values.astype("float32")
    Xv = X_val.values.astype("float32")
    pos, neg = int(y_train.sum()), int((y_train == 0).sum())
    sqrt_w = round(float(np.sqrt(neg / pos)), 1)

    def build(units, dropout, lr):
        layers = [keras.layers.Input(shape=(Xt.shape[1],))]
        for u in units:
            layers += [keras.layers.Dense(u, activation="relu"),
                       keras.layers.Dropout(dropout)]
        layers.append(keras.layers.Dense(1, activation="sigmoid"))
        m = keras.Sequential(layers)
        m.compile(optimizer=keras.optimizers.Adam(lr), loss="binary_crossentropy",
                  metrics=[keras.metrics.AUC(name="pr_auc", curve="PR")])
        return m

    space = dict(lr=[1e-3, 5e-4, 3e-4], batch=[2048, 4096],
                 units=[(64, 32), (128, 64), (128, 64, 32)],
                 dropout=[0.2, 0.3, 0.4], cw=[10, sqrt_w, 50, 100])
    rng = random.Random(0)
    configs = [dict(lr=1e-3, batch=2048, units=(64, 32), dropout=0.3, cw=sqrt_w)]  # Step 5 config
    configs += [{k: rng.choice(v) for k, v in space.items()} for _ in range(8)]

    best_mean, best_models, best_probas, best_cfg = -1, None, None, None
    for i, cfg in enumerate(configs, 1):
        aps, models, probas = [], [], []
        for seed in [42, 43, 44]:
            keras.utils.set_random_seed(seed)
            m = build(cfg["units"], cfg["dropout"], cfg["lr"])
            es = keras.callbacks.EarlyStopping(monitor="val_pr_auc", mode="max",
                                               patience=30, restore_best_weights=True)
            t0 = time.time()
            h = m.fit(Xt, y_train.values, validation_data=(Xv, y_val.values),
                      epochs=150, batch_size=cfg["batch"],
                      class_weight={0: 1.0, 1: cfg["cw"]}, callbacks=[es], verbose=0)
            p = m.predict(Xv, batch_size=4096, verbose=0).ravel()
            ap = average_precision_score(y_val, p)
            aps.append(ap); models.append(m); probas.append(p)
            log("MLP", cfg, seed, "val_pr_auc", ap,
                val_roc_auc=round(roc_auc_score(y_val, p), 4),
                epochs=len(h.history["loss"]), fit_seconds=round(time.time() - t0, 1))
        mean, std = np.mean(aps), np.std(aps)
        print(f"[{i}/{len(configs)}] mean={mean:.4f} std={std:.4f} "
              f"seeds={[round(a, 3) for a in aps]} cfg={cfg}")
        if mean > best_mean:
            best_mean, best_models, best_probas, best_cfg = mean, models, probas, cfg

    ens = np.mean(best_probas, axis=0)
    ens_ap = average_precision_score(y_val, ens)
    print(f"\nBest config: {best_cfg}\n  mean single-seed PR-AUC={best_mean:.4f}"
          f"\n  3-seed ENSEMBLE PR-AUC={ens_ap:.4f}")
    log("MLP (best config, 3-seed ensemble)", best_cfg, "ens3", "val_pr_auc", ens_ap,
        val_roc_auc=round(roc_auc_score(y_val, ens), 4))
    for s, m in zip([42, 43, 44], best_models):
        m.save(f"models/mlp_tuned_seed{s}.keras")
    json.dump({k: list(v) if isinstance(v, tuple) else v for k, v in best_cfg.items()},
              open("models/mlp_tuned_config.json", "w"))
    np.save("reports/val_proba_mlp.npy", ens)


# ---------------- Compare + freeze threshold ----------------
def run_compare():
    cands = {"XGBoost (tuned)": "xgb", "RandomForest (tuned)": "rf",
             "MLP (3-seed ensemble)": "mlp"}
    probas = {n: np.load(f"reports/val_proba_{k}.npy") for n, k in cands.items()
              if os.path.exists(f"reports/val_proba_{k}.npy")}
    if not probas:
        sys.exit("Run xgb / rf / mlp first.")
    print(f"{'model':28s} PR-AUC   ROC-AUC")
    for n, p in probas.items():
        print(f"{n:28s} {average_precision_score(y_val, p):.4f}   {roc_auc_score(y_val, p):.4f}")

    champ = max(probas, key=lambda n: average_precision_score(y_val, probas[n]))
    p = probas[champ]
    prec, rec, thr = precision_recall_curve(y_val, p)
    f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
    thr_f1 = float(thr[np.argmax(f1)])
    ok = prec[:-1] >= 0.90
    thr_p90 = float(thr[np.argmax(np.where(ok, rec[:-1], -1))]) if ok.any() else thr_f1

    print(f"\nChampion on val PR-AUC: {champ}")
    for label, t in [("max-F1", thr_f1), ("recall @ precision>=0.90", thr_p90)]:
        tn, fp, fn, tp = confusion_matrix(y_val, (p >= t).astype(int)).ravel()
        print(f"  {label:26s} thr={t:.4f}  TP={tp} FP={fp} FN={fn}  "
              f"precision={tp/(tp+fp+1e-12):.3f} recall={tp/(tp+fn):.3f}")
    json.dump({"champion": champ, "threshold": thr_p90,
               "threshold_max_f1": thr_f1, "rule": "max recall with val precision >= 0.90"},
              open("models/threshold.json", "w"), indent=2)
    print("Saved models/threshold.json (frozen: do not re-tune on test)")


if __name__ == "__main__":
    {"xgb": run_xgb, "rf": run_rf, "mlp": run_mlp, "compare": run_compare}[sys.argv[1]]()