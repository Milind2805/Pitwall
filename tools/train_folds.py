"""Train the 8 cross-validation models and record which one left out which race.
Run from the project root:   python tools/train_folds.py
"""
import json
import os
import sys

sys.path.insert(0, os.getcwd())

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from app.features import FEATURES

ds = pd.read_csv("data/lap_dataset.csv")
ds = ds[ds.groupby(["race", "lap"])["win"].transform("max") == 1].reset_index(drop=True)
os.makedirs("models", exist_ok=True)
race_fold = {}
for k, (tr, te) in enumerate(GroupKFold(n_splits=8).split(ds, groups=ds["race"])):
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=4, random_state=0)
    clf.fit(ds.loc[tr, FEATURES], ds.loc[tr, "win"])
    joblib.dump(clf, f"models/fold_{k}.joblib")
    for race in ds.loc[te, "race"].unique():
        race_fold[race] = k
    print(f"fold {k}: trained on {ds.loc[tr, 'race'].nunique()} races, left out {ds.loc[te, 'race'].nunique()}")
with open("models/race_fold.json", "w") as f:
    json.dump(race_fold, f, indent=1)
print(f"{len(race_fold)} races mapped -> models/race_fold.json")