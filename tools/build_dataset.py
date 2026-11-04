"""Gives one row per (race, lap, driver) with the features the model will see, plus the win label."""
import glob
import json
import os

import numpy as np
import pandas as pd

COMPOUND = {"SOFT": 0, "MEDIUM": 1, "HARD": 2}      # anything else (wet tyres, unknown) -> 3


def build_race(d):
    name = os.path.basename(d)
    laps = pd.read_csv(os.path.join(d, "laps.csv"))
    pos = pd.read_csv(os.path.join(d, "positions.csv")).sort_values("t")
    gaps = pd.read_csv(os.path.join(d, "intervals.csv")).sort_values("t")
    with open(os.path.join(d, "session.json")) as f:
        t0 = pd.Timestamp(json.load(f)["date_start"])

    last_pos = pos.groupby("driver")["position"].last()
    winners = last_pos[last_pos == 1].index.tolist()
    if len(winners) != 1:
        return None, name, None
    winner = winners[0]

    laps = laps.dropna(subset=["lap_duration", "date_start"]).sort_values(["driver", "lap_number"]).copy()
    started = (pd.to_datetime(laps["date_start"], utc=True, format="ISO8601") - t0).dt.total_seconds()
    laps["t_end"] = started + laps["lap_duration"]
    laps["pace3"] = laps.groupby("driver")["lap_duration"].transform(lambda s: s.rolling(3, min_periods=1).median())
    total_laps = int(laps["lap_number"].max())

    rows = []
    for L, g in laps.groupby("lap_number"):
        if L >= total_laps or len(g) < 10:
            continue
        T = g["t_end"].median()                                   # one snapshot time for the whole field
        p = pos[pos["t"] <= T].groupby("driver")["position"].last()
        gp = gaps[gaps["t"] <= T].groupby("driver")[["gap_to_leader", "interval"]].last()
        df = g.set_index("driver").join(p, how="inner").join(gp, how="left")
        if df.empty:
            continue

        lapped = df["gap_to_leader"].astype(str).str.contains("LAP")
        gl = pd.to_numeric(df["gap_to_leader"], errors="coerce").where(df["position"] > 1, 0.0)
        fallback = pd.Series(np.where(lapped, 90.0, (df["position"] - 1) * 3.0), index=df.index)
        gl = gl.fillna(fallback).clip(upper=120)
        iv = pd.to_numeric(df["interval"], errors="coerce").fillna(0).clip(upper=60)

        out = pd.DataFrame({
            "race": name, "lap": L, "driver": df.index,
            "laps_remaining": total_laps - L, "frac_done": round(L / total_laps, 4),
            "position": df["position"].values, "gap": gl.values, "interval": iv.values,
            "tyre_age": df["tyre_age"].fillna(0).values,
            "compound": df["compound"].map(COMPOUND).fillna(3).values,
            "pits_done": (df["stint"].fillna(1) - 1).values,
            "pace_delta": (df["pace3"] - df["pace3"].median()).clip(-10, 10).values,
            "lapped": lapped.astype(int).values,
        })
        out["win"] = (out["driver"] == winner).astype(int)
        rows.append(out)
    return (pd.concat(rows) if rows else None), name, winner


def main():
    parts = []
    for d in sorted(glob.glob("data/20*_*")):
        need = ("laps.csv", "positions.csv", "intervals.csv", "session.json")
        if not all(os.path.exists(os.path.join(d, f)) for f in need):
            continue
        try:
            df, name, winner = build_race(d)
        except Exception as ex:
            print(f"{os.path.basename(d):28s} failed: {ex}")
            continue
        if df is None:
            print(f"{name:28s} skipped (no clear winner or no laps)")
            continue
        parts.append(df)
        print(f"{name:28s} winner {winner}  {len(df)} rows")
    ds = pd.concat(parts, ignore_index=True)
    ds.to_csv("data/lap_dataset.csv", index=False)
    print(f"\n{len(ds)} rows, {ds['race'].nunique()} races -> data/lap_dataset.csv")


if __name__ == "__main__":
    main()