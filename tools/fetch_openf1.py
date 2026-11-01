import hashlib
import json
import os
import sys
import time
from urllib.parse import quote

import pandas as pd
import requests

BASE = "https://api.openf1.org/v1"
CACHE_DIR = os.path.join("cache", "openf1")
MIN_GAP_S = 0.4          # pause between requests, to be polite to a free service
_last_request = [0.0]


def get_json(path, query):
    """GET one endpoint. Cached on disk, throttled, retries when told to slow down."""
    url = f"{BASE}/{path}?{query}"
    cache_file = os.path.join(CACHE_DIR, hashlib.md5(url.encode()).hexdigest() + ".json")
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            return json.load(f)
    os.makedirs(CACHE_DIR, exist_ok=True)

    wait = 5
    for _ in range(10):
        gap = time.time() - _last_request[0]
        if gap < MIN_GAP_S:
            time.sleep(MIN_GAP_S - gap)
        _last_request[0] = time.time()
        r = requests.get(url, timeout=60)
        if r.status_code == 200:
            data = r.json()
        elif r.status_code == 404:
            data = []                                  # OpenF1 answers 404 when nothing matches
        elif r.status_code in (429, 500, 502, 503, 504):
            delay = float(r.headers.get("Retry-After", wait))
            print(f"  server answered {r.status_code}, waiting {delay:.0f}s ...")
            time.sleep(delay)
            wait = min(wait * 2, 60)
            continue
        else:
            print(f"  request failed: {r.status_code} {r.text[:200]}  ({url})")
            r.raise_for_status()
        with open(cache_file, "w") as f:
            json.dump(data, f)
        return data
    raise RuntimeError(f"gave up after repeated errors: {url}")


def q(**kw):
    return "&".join(f"{k}={quote(str(v))}" for k, v in kw.items())


def fmt(ts):
    return ts.strftime("%Y-%m-%dT%H:%M:%S")             # UTC, no timezone suffix


def windows(start, end, minutes=15):
    t = start
    while t < end:
        nxt = min(t + pd.Timedelta(minutes=minutes), end)
        yield t, nxt
        t = nxt


def fetch_windowed(path, base_query, start, end):
    rows = []
    for a, b in windows(start, end):
        rows += get_json(path, f"{base_query}&date>={fmt(a)}&date<{fmt(b)}")
    return rows


def find_session(year, circuit):
    rows = get_json("sessions", q(year=year, circuit_short_name=circuit, session_name="Race"))
    if not rows:
        known = get_json("sessions", q(year=year, session_name="Race"))
        names = sorted({r["circuit_short_name"] for r in known})
        print(f"No race found for circuit '{circuit}' in {year}.")
        print("Circuits available that year:", ", ".join(names) if names else "(none: is the year 2023 or later?)")
        sys.exit(1)
    return rows[0]


def build_driver_telemetry(session_key, dn, acronym, start, end, t0):
    car = pd.DataFrame(fetch_windowed("car_data", q(session_key=session_key, driver_number=dn), start, end))
    loc = pd.DataFrame(fetch_windowed("location", q(session_key=session_key, driver_number=dn), start, end))
    if car.empty or loc.empty:
        return None
    car["date"] = pd.to_datetime(car["date"], utc=True, format="ISO8601")
    loc["date"] = pd.to_datetime(loc["date"], utc=True, format="ISO8601")
    car = car.sort_values("date")
    loc = loc.sort_values("date")[["date", "x", "y"]]
    m = pd.merge_asof(car, loc, on="date", direction="nearest", tolerance=pd.Timedelta("500ms"))
    out = pd.DataFrame({
        "t": (m["date"] - t0).dt.total_seconds().round(3),
        "driver": acronym,
        "speed": m["speed"], "rpm": m["rpm"], "gear": m["n_gear"],
        "throttle": m["throttle"], "brake": (m["brake"] > 0).astype(int),
        "drs": m["drs"], "x": m["x"], "y": m["y"],
    })
    return out


def build_laps(laps_rows, stints_rows, acronyms):
    laps = pd.DataFrame(laps_rows)
    if laps.empty:
        return laps
    laps["driver"] = laps["driver_number"].map(acronyms)
    laps["compound"] = None
    laps["stint"] = None
    laps["tyre_age"] = None
    for s in stints_rows:
        last = s["lap_end"] if s.get("lap_end") is not None else 9999
        m = (laps["driver_number"] == s["driver_number"]) & (laps["lap_number"] >= s["lap_start"]) & (laps["lap_number"] <= last)
        laps.loc[m, "compound"] = s["compound"]
        laps.loc[m, "stint"] = s["stint_number"]
        laps.loc[m, "tyre_age"] = (s.get("tyre_age_at_start") or 0) + (laps.loc[m, "lap_number"] - s["lap_start"])
    keep = ["driver", "driver_number", "lap_number", "lap_duration", "duration_sector_1",
            "duration_sector_2", "duration_sector_3", "date_start", "is_pit_out_lap",
            "compound", "stint", "tyre_age"]
    return laps[[c for c in keep if c in laps.columns]].sort_values(["lap_number", "driver"])


def timed(rows, acronyms, t0, cols):
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["t"] = (pd.to_datetime(df["date"], utc=True, format="ISO8601") - t0).dt.total_seconds().round(3)
    df["driver"] = df["driver_number"].map(acronyms)
    return df[["t", "driver"] + cols].sort_values("t")


def main():
    year = int(sys.argv[1]) if len(sys.argv) > 1 else 2024
    circuit = sys.argv[2] if len(sys.argv) > 2 else "Monza"
    out_dir = os.path.join("data", f"{year}_{circuit.replace(' ', '_')}")
    os.makedirs(out_dir, exist_ok=True)

    sess = find_session(year, circuit)
    sk = sess["session_key"]
    start = pd.to_datetime(sess["date_start"], utc=True)
    end = pd.to_datetime(sess["date_end"], utc=True)
    print(f"Found: {year} {circuit} race, session_key {sk}, {start} -> {end}")

    drivers = pd.DataFrame(get_json("drivers", q(session_key=sk)))
    acronyms = dict(zip(drivers["driver_number"], drivers["name_acronym"]))
    drivers[["driver_number", "name_acronym", "full_name", "team_name", "team_colour"]].to_csv(
        os.path.join(out_dir, "drivers.csv"), index=False)
    print(f"{len(drivers)} drivers")

    laps = build_laps(get_json("laps", q(session_key=sk)), get_json("stints", q(session_key=sk)), acronyms)
    laps.to_csv(os.path.join(out_dir, "laps.csv"), index=False)
    print(f"{len(laps)} laps")

    pit = pd.DataFrame(get_json("pit", q(session_key=sk)))
    if not pit.empty:
        pit["driver"] = pit["driver_number"].map(acronyms)
        pit.to_csv(os.path.join(out_dir, "pit.csv"), index=False)

    print("Positions and gaps ...")
    pos = timed(fetch_windowed("position", q(session_key=sk), start, end), acronyms, start, ["position"])
    pos.to_csv(os.path.join(out_dir, "positions.csv"), index=False)
    gaps = timed(fetch_windowed("intervals", q(session_key=sk), start, end), acronyms, start,
                 ["gap_to_leader", "interval"])
    gaps.to_csv(os.path.join(out_dir, "intervals.csv"), index=False)

    frames = []
    for i, (dn, acr) in enumerate(acronyms.items(), 1):
        print(f"Telemetry {i}/{len(acronyms)}: {acr} ...")
        part = build_driver_telemetry(sk, dn, acr, start, end, start)
        if part is not None:
            frames.append(part)
    tel = pd.concat(frames).sort_values("t").reset_index(drop=True)
    tel.to_csv(os.path.join(out_dir, "telemetry.csv.gz"), index=False)

    with open(os.path.join(out_dir, "session.json"), "w") as f:
        json.dump({"session_key": sk, "year": year, "circuit": circuit,
                   "date_start": str(start), "date_end": str(end)}, f, indent=2)
    print(f"Done: {len(tel)} telemetry rows saved to {out_dir}/")


if __name__ == "__main__":
    main()