"""FetchesS laps, stints, positions and gaps for every race of a season (no car telemetry)."""
import json
import os
import sys

import pandas as pd

from fetch_openf1 import get_json, q, fetch_windowed, build_laps, timed, fetch_positions


def fetch_one(sess):
    circuit, sk = sess["circuit_short_name"], sess["session_key"]
    out_dir = os.path.join("data", f"{sess['year']}_{circuit.replace(' ', '_')}")
    files = [os.path.join(out_dir, f) for f in ("positions.csv", "intervals.csv", "laps.csv")]
    if all(os.path.exists(f) for f in files) and (pd.read_csv(files[0])["t"] < 0).any():
        print(f"skip {out_dir} (already fetched)")
        return
    start = pd.to_datetime(sess["date_start"], utc=True)
    end = pd.to_datetime(sess["date_end"], utc=True)

    drivers = pd.DataFrame(get_json("drivers", q(session_key=sk)))
    acr = dict(zip(drivers["driver_number"], drivers["name_acronym"]))
    laps = build_laps(get_json("laps", q(session_key=sk)), get_json("stints", q(session_key=sk)), acr)
    pos = fetch_positions(sk, acr, start, end)
    gaps = timed(fetch_windowed("intervals", q(session_key=sk), start, end), acr, start,
                 ["gap_to_leader", "interval"])
    if laps.empty or pos.empty or gaps.empty:
        print(f"  incomplete data for {out_dir}, not saved")
        return

    os.makedirs(out_dir, exist_ok=True)
    drivers[["driver_number", "name_acronym", "full_name", "team_name", "team_colour"]].to_csv(
        os.path.join(out_dir, "drivers.csv"), index=False)
    laps.to_csv(os.path.join(out_dir, "laps.csv"), index=False)
    pos.to_csv(os.path.join(out_dir, "positions.csv"), index=False)
    gaps.to_csv(os.path.join(out_dir, "intervals.csv"), index=False)
    with open(os.path.join(out_dir, "session.json"), "w") as f:
        json.dump({"session_key": sk, "year": sess["year"], "circuit": circuit,
                   "date_start": str(start), "date_end": str(end)}, f, indent=2)
    print(f"saved {out_dir}: {len(laps)} laps")


def main():
    year = int(sys.argv[1]) if len(sys.argv) > 1 else 2024
    races = sorted(get_json("sessions", q(year=year, session_name="Race")), key=lambda s: s["date_start"])
    print(f"{len(races)} races in {year}")
    for s in races:
        try:
            fetch_one(s)
        except Exception as ex:                       # one bad race should not stop the season
            print(f"  failed {s['circuit_short_name']}: {ex}")


if __name__ == "__main__":
    main()