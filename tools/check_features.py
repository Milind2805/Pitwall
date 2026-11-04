import os
import sys

sys.path.insert(0, os.getcwd())

import pandas as pd

from app.features import FEATURES, build_rows
from app.replayer import all_events
from app.server import RaceState, _num

ds = pd.read_csv("data/lap_dataset.csv")
ds = ds[ds["race"] == "2024_Monza"]
events = list(all_events("data/2024_Monza"))
total = max(e["lap_number"] for e in events if e["type"] == "lap")

state, diffs = RaceState(), []
for e in events:
    if e["type"] == "gap":
        e["gap_to_leader"], e["interval"] = _num(e["gap_to_leader"]), _num(e["interval"])
    state.update(e)
    if state.lap < total and state.ready_for_prediction():
        state.pred_lap = state.lap
        built = build_rows(state.positions, state.gaps, state.last_lap, state.durs, state.lap, total)
        if built is None:
            continue
        drivers, rows = built
        live = pd.DataFrame(rows, columns=FEATURES, index=drivers).astype(float)
        ref = ds[ds["lap"] == state.lap].set_index("driver")[FEATURES].astype(float)
        common = live.index.intersection(ref.index)
        if len(common):
            diffs.append((live.loc[common] - ref.loc[common]).abs())

d = pd.concat(diffs)
print(f"{len(diffs)} laps compared, {len(d)} driver-rows\n")
print(pd.DataFrame({"mean abs diff": d.mean().round(3), "exact match": (d < 1e-6).mean().round(3)}))