"""Rewrite reports/experiments.csv into one consistent table.

tune.py appended rows with different columns under the first header, so the raw
file is misaligned. This script fixes it (a backup of the raw file is kept).

Run from the project root:  python src/clean_experiments_log.py
"""
import csv, os, shutil

PATH = "reports/experiments.csv"
BACKUP = "reports/experiments_raw_backup.csv"
COLS = ["time", "model", "config", "seed", "cv_pr_auc", "cv_std",
        "val_pr_auc", "val_roc_auc", "epochs", "fit_seconds"]

with open(PATH, newline="") as f:
    rows = list(csv.reader(f))

if rows[0] == COLS:
    print("Already clean, nothing to do.")
    raise SystemExit

shutil.copy(PATH, BACKUP)
out = []
for r in rows[1:]:
    if not r:
        continue
    d = dict.fromkeys(COLS, "")
    d.update(time=r[0], model=r[1], config=r[2], seed=r[3])
    if len(r) == 8:                                   # MLP single run
        d.update(val_pr_auc=r[4], val_roc_auc=r[5], epochs=r[6], fit_seconds=r[7])
    elif "(best" in r[1]:                             # refit / ensemble rows
        d.update(val_pr_auc=r[4], val_roc_auc=r[5])
    else:                                             # CV search rows
        d.update(cv_pr_auc=r[4], cv_std=r[5])
    out.append(d)

with open(PATH, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=COLS)
    w.writeheader()
    w.writerows(out)
print(f"Cleaned {len(out)} rows. Raw copy saved to {BACKUP}")
