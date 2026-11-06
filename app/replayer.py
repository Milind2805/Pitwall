import argparse
import heapq
import json
import os
import time
from collections import Counter

import pandas as pd


def _rows(df):
    """DataFrame -> dict per row, with NaN turned into None (valid JSON)."""
    df = df.astype(object).where(df.notna(), None)
    return df.to_dict("records")


def car_events(data_dir):
    path = os.path.join(data_dir, "telemetry.csv.gz")
    if not os.path.exists(path):          # races fetched with fetch_summary.py have no car telemetry
        return
    for r in _rows(pd.read_csv(path)):
        r["type"] = "car"
        yield r


def lap_events(data_dir):
    with open(os.path.join(data_dir, "session.json")) as f:
        t0 = pd.Timestamp(json.load(f)["date_start"])
    laps = pd.read_csv(os.path.join(data_dir, "laps.csv")).dropna(subset=["lap_duration", "date_start"])
    started = (pd.to_datetime(laps["date_start"], utc=True, format="ISO8601") - t0).dt.total_seconds()
    laps = laps.assign(t=(started + laps["lap_duration"]).round(3)).sort_values("t")
    cols = ["t", "driver", "lap_number", "lap_duration", "compound", "stint", "tyre_age"]
    for r in _rows(laps[[c for c in cols if c in laps.columns]]):
        r["type"] = "lap"
        yield r


def table_events(data_dir, filename, kind):
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        return
    for r in _rows(pd.read_csv(path)):
        r["type"] = kind
        yield r


def all_events(data_dir):
    """Every event of the race, in time order."""
    streams = [
        car_events(data_dir),
        lap_events(data_dir),
        table_events(data_dir, "positions.csv", "position"),
        table_events(data_dir, "intervals.csv", "gap"),
    ]
    return heapq.merge(*streams, key=lambda e: e["t"])


def play(events, emit, speed=1.0, start_at=0.0, duration=None):
    """Call emit(event) for each event, at `speed` times real time."""
    begin = time.monotonic()
    for e in events:
        if e["t"] < start_at:
            continue
        if duration is not None and e["t"] > start_at + duration:
            break
        delay = (e["t"] - start_at) / speed - (time.monotonic() - begin)
        if delay > 0:
            time.sleep(delay)
        emit(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--speed", type=float, default=20.0, help="1 = real time")
    ap.add_argument("--start", type=float, default=0.0, help="start at this many seconds into the race")
    ap.add_argument("--seconds", type=float, default=None, help="stop after this many race seconds")
    ap.add_argument("--raw", action="store_true", help="print every event as JSON")
    args = ap.parse_args()

    print(f"Loading {args.data_dir} ...")
    events = all_events(args.data_dir)

    counts, state = Counter(), {"last_print": time.monotonic(), "leader": "-", "lap": 0, "n": 0}

    def emit(e):
        state["n"] += 1
        counts[e["type"]] += 1
        if e["type"] == "position" and e["position"] == 1:
            state["leader"] = e["driver"]
        if e["type"] == "lap":
            state["lap"] = max(state["lap"], e["lap_number"])
        if args.raw:
            print(json.dumps(e))
            return
        now = time.monotonic()
        if now - state["last_print"] >= 1.0:
            rate = sum(counts.values()) / (now - state["last_print"])
            print(f"race t={e['t']:7.1f}s | {rate:6.0f} events/s | leader {state['leader']} | "
                  f"lap {state['lap']} | car={counts['car']} lap={counts['lap']} "
                  f"pos={counts['position']} gap={counts['gap']}")
            counts.clear()
            state["last_print"] = now

    t_wall = time.monotonic()
    play(events, emit, speed=args.speed, start_at=args.start, duration=args.seconds)
    print(f"Done: {state['n']} events in {time.monotonic() - t_wall:.1f}s")


if __name__ == "__main__":
    main()