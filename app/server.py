"""PitWall server: replays a race and pushes the events to browsers over a WebSocket."""

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from app.replayer import all_events
from app.predictor import win_probs

DATA_DIR = os.environ.get("PITWALL_DATA", "data/2024_Monza")
SPEED = float(os.environ.get("PITWALL_SPEED", "20"))
START = float(os.environ.get("PITWALL_START", "0"))
TICK = 0.05            # seconds between batches
QUEUE_SIZE = 100       # messages a slow client may fall behind before we start dropping


def _num(v):
    """'4.146' -> 4.146, but '+1 LAP' stays text."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


class RaceState:
    """Latest known facts, so a browser that joins late can be brought up to date."""

    def __init__(self):
        self.reset()

    def reset(self):
            self.race_t = 0.0
            self.lap = 0
            self.prediction = None
            self.cars, self.positions, self.gaps, self.laps = {}, {}, {}, []

    def update(self, e):
        kind = e["type"]
        if kind == "car":
            self.cars[e["driver"]] = e
        elif kind == "position":
            self.positions[e["driver"]] = e["position"]
        elif kind == "gap":
            self.gaps[e["driver"]] = {"gap_to_leader": e["gap_to_leader"], "interval": e["interval"]}
        elif kind == "lap":
            self.laps.append(e)
            self.lap = max(self.lap, e.get("lap_number") or 0)
        elif kind == "prediction":
            self.prediction = e

    def snapshot(self, drivers):
        return {"type": "snapshot", "race_t": round(self.race_t, 2), "drivers": drivers,
                "cars": self.cars, "positions": self.positions, "gaps": self.gaps,
                "laps": self.laps, "prediction": self.prediction}


state = RaceState()
clients = set()        # one queue per connected browser
stats = {"events_sent": 0, "batches": 0, "dropped_messages": 0}
drivers_info = {}


def load_drivers():
    df = pd.read_csv(os.path.join(DATA_DIR, "drivers.csv"))
    return {r.name_acronym: {"name": r.full_name, "team": r.team_name, "colour": f"#{r.team_colour}"}
            for r in df.itertuples()}


def publish(message):
    for q in clients:
        try:
            q.put_nowait(message)
        except asyncio.QueueFull:
            stats["dropped_messages"] += 1
            while not q.empty():                       # backlog is useless now
                q.get_nowait()
            q.put_nowait(json.dumps(state.snapshot(drivers_info)))   # resync
            stats["resyncs"] = stats.get("resyncs", 0) + 1

async def race_loop():
    events = await asyncio.to_thread(lambda: list(all_events(DATA_DIR)))
    for e in events:
        if e["type"] == "gap":
            e["gap_to_leader"], e["interval"] = _num(e["gap_to_leader"]), _num(e["interval"])
    total_laps = max((e["lap_number"] for e in events if e["type"] == "lap"), default=0)   # NEW
    print(f"Loaded {len(events)} events ({total_laps} laps). Replaying at {SPEED}x from t={START}s")

    while True:
        state.reset()
        i = 0
        while i < len(events) and events[i]["t"] < START:      # catch up silently
            state.update(events[i])
            i += 1
        state.race_t = START
        publish(json.dumps(state.snapshot(drivers_info)))     # every browser starts fresh when the race (re)starts
        begin = time.monotonic()
        while i < len(events):
            race_t = START + (time.monotonic() - begin) * SPEED
            batch = []
            while i < len(events) and events[i]["t"] <= race_t:
                state.update(events[i])
                batch.append(events[i])
                i += 1
            if any(e["type"] == "lap" for e in batch):                                    
                pred = {"type": "prediction", "t": round(race_t, 2), "lap": state.lap,   
                        "probs": win_probs(state.positions, state.gaps, state.lap, total_laps)}  
                state.update(pred)                                                        
                batch.append(pred)                                                        
            state.race_t = race_t
            if batch:
                publish(json.dumps({"type": "batch", "race_t": round(race_t, 2), "events": batch}))
                stats["events_sent"] += len(batch)
                stats["batches"] += 1
            await asyncio.sleep(TICK)
        print("Race finished. Restarting in 5 s.")
        await asyncio.sleep(5)


@asynccontextmanager
async def lifespan(_app):
    drivers_info.update(load_drivers())
    task = asyncio.create_task(race_loop())
    yield
    task.cancel()


app = FastAPI(title="PitWall", lifespan=lifespan)


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    q = asyncio.Queue(maxsize=QUEUE_SIZE)
    clients.add(q)
    try:
        await websocket.send_text(json.dumps(state.snapshot(drivers_info)))
        while True:
            await websocket.send_text(await q.get())
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        clients.discard(q)


@app.get("/api/status")
def status():
    return {"clients": len(clients), "race_t": round(state.race_t, 1), "speed": SPEED, **stats}


if os.path.isdir("app/static"):                      # the dashboard page (Phase 3)
    app.mount("/", StaticFiles(directory="app/static", html=True), name="static")