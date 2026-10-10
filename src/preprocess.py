import json, os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

RAW = "data/creditcard.csv"
OUT = "data/processed"
SEED = 42

os.makedirs(OUT, exist_ok=True)
os.makedirs("models", exist_ok=True)

# 1. Load, dedupe, drop Time
df = pd.read_csv(RAW)
print("Raw:", df.shape)
df = df.drop_duplicates().drop(columns="Time").reset_index(drop=True)
print("After dedup:", df.shape, "| fraud:", df["Class"].sum())

X = df.drop(columns="Class")
y = df["Class"]

# 2. Stratified split: 70% train, 15% val, 15% test
X_train, X_tmp, y_train, y_tmp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=SEED)
X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED)

# 3. Preprocessing pipeline (fit on train only)
amount_pipe = Pipeline([
    ("log", FunctionTransformer(np.log1p)),
    ("scale", StandardScaler()),
])
preprocessor = ColumnTransformer(
    [("amount", amount_pipe, ["Amount"])],
    remainder="passthrough",
    verbose_feature_names_out=False,
).set_output(transform="pandas")

X_train_p = preprocessor.fit_transform(X_train)
X_val_p = preprocessor.transform(X_val)
X_test_p = preprocessor.transform(X_test)

# 4. Save everything
for name, Xp, yy in [("train", X_train_p, y_train),
                     ("val", X_val_p, y_val),
                     ("test", X_test_p, y_test)]:
    out = Xp.copy()
    out["Class"] = yy.values
    out.to_csv(f"{OUT}/{name}.csv", index=False)
    print(f"{name}: {len(out):>7} rows | fraud: {int(yy.sum()):>3} "
          f"({yy.mean()*100:.3f}%)")

joblib.dump(preprocessor, "models/preprocessor.joblib")
with open("models/feature_order.json", "w") as f:
    json.dump(list(X_train_p.columns), f)

print("Feature order:", list(X_train_p.columns))