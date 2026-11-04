"""Train the model and save it for the live server."""
import os
import sys

sys.path.insert(0, os.getcwd())

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from app.features import FEATURES

ds = pd.read_csv("data/lap_dataset.csv")
ds = ds[ds.groupby(["race", "lap"])["win"].transform("max") == 1]
if len(sys.argv) > 1:
    ds = ds[ds["race"] != sys.argv[1]]
    print(f"leaving out {sys.argv[1]}")
clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=4, random_state=0)
clf.fit(ds[FEATURES], ds["win"])
os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/win_model.joblib")
print(f"trained on {ds['race'].nunique()} races, {len(ds)} rows -> models/win_model.joblib")