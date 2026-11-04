"""Training a lap-by-lap win model on some races, scoring it on races it has never seen, and comparing with the heuristic in app/predictor.py."""
import os
import sys

sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from app.predictor import win_probs

FEATURES = ["laps_remaining", "frac_done", "position", "gap", "interval",
            "tyre_age", "compound", "pits_done", "pace_delta", "lapped"]
KEY = ["race", "lap"]

ds = pd.read_csv("data/lap_dataset.csv")
ds = ds[ds.groupby(KEY)["win"].transform("max") == 1].reset_index(drop=True)   # laps where the winner is in the data

# 1. Model, out of fold: every race is predicted by a model that never saw that race.
ds["p_model"] = 0.0
for tr, te in GroupKFold(n_splits=8).split(ds, groups=ds["race"]):
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=4, random_state=0)
    clf.fit(ds.loc[tr, FEATURES], ds.loc[tr, "win"])
    ds.loc[te, "p_model"] = clf.predict_proba(ds.loc[te, FEATURES])[:, 1]
ds["p_model"] = ds["p_model"] / (ds.groupby(KEY)["p_model"].transform("sum") + 1e-12)   # sum to 1 per lap

# 2. Heuristic from predictor.py, on the same snapshots.
parts = []
for (race, lap), g in ds.groupby(KEY, sort=False):
    total = lap + int(g["laps_remaining"].iloc[0])
    pr = win_probs(dict(zip(g["driver"], g["position"])),
                   {d: {"gap_to_leader": x} for d, x in zip(g["driver"], g["gap"])}, lap, total)
    parts.append(g["driver"].map(pr))
ds["p_heur"] = pd.concat(parts)

# 3. Do-nothing baseline: everyone equally likely.
ds["p_uniform"] = 1 / ds.groupby(KEY)["driver"].transform("count")


def scores(col):
    p = ds[col].clip(1e-4, 1 - 1e-4)
    brier = ((p - ds["win"]) ** 2).groupby([ds["race"], ds["lap"]]).sum()
    logloss = (-np.log(p)).where(ds["win"] == 1).groupby([ds["race"], ds["lap"]]).sum()
    return brier, logloss


phase = pd.cut(ds.groupby(KEY)["frac_done"].first(), [0, .25, .5, .75, 1.0],
               labels=["0-25% of race", "25-50%", "50-75%", "75-100%"])
out = {}
for name, col in [("uniform", "p_uniform"), ("heuristic", "p_heur"), ("model", "p_model")]:
    b, l = scores(col)
    out[(name, "logloss")] = l.groupby(phase,observed=False).mean()
    out[(name, "brier")] = b.groupby(phase,observed=False).mean()
    out[(name, "logloss")]["ALL"] = l.mean()
    out[(name, "brier")]["ALL"] = b.mean()
print(f"{ds['race'].nunique()} races, {len(ds)} rows (lower is better)\n")
print(pd.DataFrame(out).round(3).to_string())

# 4. The Monza 2024 case from the dashboard (LEC won).
for lap in (20, 33, 45):
    g = ds[(ds["race"] == "2024_Monza") & (ds["lap"] == lap)].set_index("driver")
    if len(g):
        print(f"\nMonza 2024, after lap {lap}:  P(LEC wins)  heuristic {g.loc['LEC','p_heur']:.1%}   model {g.loc['LEC','p_model']:.1%}")