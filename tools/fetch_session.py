"""Download one F1 race with FastF1 and save it as plain files.

Usage:  python tools/fetch_session.py 2024 Monza
Output: data/<year>_<gp>/laps.csv and data/<year>_<gp>/telemetry.csv.gz
"""
import os
import sys

import pandas as pd


def build_outputs(session, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # ---- laps: one row per driver per lap ----
    cols = ["Driver", "DriverNumber", "LapNumber", "LapTime", "Compound", "TyreLife",
            "Stint", "PitInTime", "PitOutTime", "Position", "Time"]
    laps = session.laps[cols].copy()
    for c in ["LapTime", "PitInTime", "PitOutTime", "Time"]:
        laps[c] = laps[c].dt.total_seconds()          # timedeltas -> seconds
    laps.to_csv(os.path.join(out_dir, "laps.csv"), index=False)

    # ---- telemetry: car data + position, merged per driver, one big timeline ----
    frames = []
    for num in session.drivers:
        if num not in session.car_data or num not in session.pos_data:
            continue
        abbr = session.get_driver(num)["Abbreviation"]
        car = session.car_data[num][["SessionTime", "Speed", "RPM", "nGear",
                                     "Throttle", "Brake", "DRS"]].copy()
        pos = session.pos_data[num][["SessionTime", "X", "Y"]].copy()
        car["t"] = car["SessionTime"].dt.total_seconds()
        pos["t"] = pos["SessionTime"].dt.total_seconds()
        merged = pd.merge_asof(car.sort_values("t"), pos.sort_values("t")[["t", "X", "Y"]],
                               on="t", direction="nearest", tolerance=0.5)
        merged["driver"] = abbr
        frames.append(merged[["t", "driver", "Speed", "RPM", "nGear", "Throttle",
                              "Brake", "DRS", "X", "Y"]])

    telemetry = pd.concat(frames).sort_values("t").reset_index(drop=True)
    telemetry["Brake"] = telemetry["Brake"].astype(int)
    telemetry.to_csv(os.path.join(out_dir, "telemetry.csv.gz"), index=False)
    return len(laps), len(telemetry)


def main():
    import fastf1

    year = int(sys.argv[1]) if len(sys.argv) > 1 else 2024
    gp = sys.argv[2] if len(sys.argv) > 2 else "Monza"
    out_dir = os.path.join("data", f"{year}_{gp.replace(' ', '_')}")

    os.makedirs("cache", exist_ok=True)
    fastf1.Cache.enable_cache("cache")          # so the second run is instant

    print(f"Loading {year} {gp} (race). The first time this takes a few minutes...")
    session = fastf1.get_session(year, gp, "R")
    session.load(laps=True, telemetry=True, weather=False, messages=False)

    n_laps, n_rows = build_outputs(session, out_dir)
    print(f"Saved {n_laps} laps and {n_rows} telemetry rows to {out_dir}/")


if __name__ == "__main__":
    main()