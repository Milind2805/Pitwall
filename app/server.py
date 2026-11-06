"""PitWall server: replays a race and pushes the events to browsers over a WebSocket."""

import asyncio
import json
import glob
import os
import time
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from app.replayer import all_events
from app.predictor import live_probs
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from app.features import build_rows
from app.predictor import live_probs
from app.qf_client import QueueForge

RACES_DIR = "data"
current = {"name": os.path.basename(os.path.normpath(os.environ.get("PITWALL_DATA", "data/2024_Monza"))),
           "task": None}
SPEED = float(os.environ.get("PITWALL_SPEED", "20"))
START = float(os.environ.get("PITWALL_START", "0"))
TICK = 0.05            # seconds between batches
QUEUE_SIZE = 100       # messages a slow client may fall behind before we start dropping
USE_QF = os.environ.get("PITWALL_QF") == "1"      # off by default: the old in-process path still works
qf = QueueForge()
_tasks = set()


def _num(v):
    """'4.146' -> 4.146, but '+1 LAP' stays text."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


class RaceState:
    """Latest known facts, so a browser that joins late can be brought up to date."""

    def __init__(self):
        self.race = ""
        self.total_laps = 0
        self.reset()
    
    def reset(self):
            self.run_id = time.time_ns()
            self.race_t = 0.0
            self.lap = 0
            self.pred_lap = 0
            self.prediction = None
            self.pred_history = []
            self.cars, self.positions, self.gaps, self.laps = {}, {}, {}, []
            self.last_lap, self.durs = {}, {}

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
            d = e["driver"]
            self.last_lap[d] = e
            if e.get("lap_duration") is not None:
                self.durs.setdefault(d, []).append(e["lap_duration"])
                del self.durs[d][:-3]                      # keep the last 3, for the pace feature
            elif kind == "prediction":
                self.prediction = e
                self.pred_history.append({"lap": e["lap"], "probs": e["probs"]})

    def ready_for_prediction(self):
        """Once per lap, when half the field has completed it (training snapshots the same moment)."""
        if self.lap <= self.pred_lap:
            return False
        done = sum(1 for e in self.last_lap.values() if (e.get("lap_number") or 0) >= self.lap)
        return done * 2 >= len(self.positions)

    def snapshot(self, drivers):
        return {"type": "snapshot", "race_t": round(self.race_t, 2), "drivers": drivers,
                "cars": self.cars, "positions": self.positions, "gaps": self.gaps,
                "laps": self.laps, "prediction": self.prediction, "race": self.race, "total_laps": self.total_laps, "pred_history": self.pred_history}


state = RaceState()
clients = set()        # one queue per connected browser
stats = {"events_sent": 0, "batches": 0, "dropped_messages": 0, "resyncs": 0,
         "jobs_submitted": 0, "submit_failures": 0, "predictions_via_queue": 0, "stale_predictions": 0, "duplicate_submits": 0}
drivers_info = {}


def load_drivers(name):
    df = pd.read_csv(os.path.join(RACES_DIR, name, "drivers.csv"))
    return {r.name_acronym: {"name": r.full_name, "team": r.team_name, "colour": f"#{r.team_colour}"}
            for r in df.itertuples()}


def list_races():
    out = []
    for d in glob.glob(os.path.join(RACES_DIR, "20*_*")):
        need = ("drivers.csv", "laps.csv", "positions.csv", "intervals.csv", "session.json")
        if not all(os.path.exists(os.path.join(d, f)) for f in need):
            continue
        with open(os.path.join(d, "session.json")) as f:
            s = json.load(f)
        out.append({"name": os.path.basename(d), "label": f"{s['year']} {s['circuit']}", "date": s["date_start"],
                    "telemetry": os.path.exists(os.path.join(d, "telemetry.csv.gz"))})
    return sorted(out, key=lambda r: r["date"])


async def start_race(name):
    task = current["task"]
    if task:
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
    current["name"] = name
    drivers_info.clear()
    drivers_info.update(load_drivers(name))
    state.reset()                              # also gives the new race a new run id, so old jobs are dropped as stale
    state.race = name
    state.total_laps = 0
    publish(json.dumps(state.snapshot(drivers_info)))     # browsers clear at once; the loop sends the real start
    current["task"] = asyncio.create_task(race_loop(name))


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

_submitted = set()


async def submit_job(run, lap, payload):
    if (run, lap) in _submitted:                      # one job per (run, lap), whatever the caller does
        stats["duplicate_submits"] += 1
        return
    _submitted.add((run, lap))
    try:
        await qf.submit("predict", payload, priority=lap)     # newest lap first; FIFO within a lap
        stats["jobs_submitted"] += 1
    except Exception as ex:
        _submitted.discard((run, lap))
        stats["submit_failures"] += 1
        print(f"could not submit prediction job: {ex}")

async def race_loop(name):
    data_dir = os.path.join(RACES_DIR, name)
    events = await asyncio.to_thread(lambda: list(all_events(data_dir)))
    for e in events:
        if e["type"] == "gap":
            e["gap_to_leader"], e["interval"] = _num(e["gap_to_leader"]), _num(e["interval"])
    total_laps = max((e["lap_number"] for e in events if e["type"] == "lap"), default=0)
    print(f"{name}: loaded {len(events)} events ({total_laps} laps). Replaying at {SPEED}x from t={START}s")

    while True:
        state.reset()
        state.race = name
        state.total_laps = total_laps
        i = 0
        while i < len(events) and events[i]["t"] < START:      # catch up silently
            state.update(events[i])
            i += 1
        state.race_t = START
        publish(json.dumps(state.snapshot(drivers_info)))
        begin = time.monotonic()
        while i < len(events):
            race_t = START + (time.monotonic() - begin) * SPEED
            batch = []
            while i < len(events) and events[i]["t"] <= race_t:
                state.update(events[i])
                batch.append(events[i])
                i += 1
            if state.lap < total_laps and state.ready_for_prediction():
                state.pred_lap = state.lap
                built = build_rows(state.positions, state.gaps, state.last_lap,
                                   state.durs, state.lap, total_laps) if USE_QF else None
                if built:
                    drivers, rows = built
                    payload = json.dumps({"run": state.run_id, "race": name, "lap": state.lap,
                                          "drivers": drivers, "rows": rows}, separators=(",", ":"))
                    t = asyncio.create_task(submit_job(state.run_id, state.lap, payload))
                    _tasks.add(t)
                    t.add_done_callback(_tasks.discard)
                else:
                    probs, source = live_probs(state.positions, state.gaps, state.last_lap,
                                               state.durs, state.lap, total_laps, name)
                    pred = {"type": "prediction", "t": round(race_t, 2), "lap": state.lap,
                            "source": source, "probs": probs}
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
    await start_race(current["name"])
    yield
    if current["task"]:
        current["task"].cancel()
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
@app.post("/internal/prediction")

async def post_prediction(request: Request):
    r = await request.json()
    last = state.prediction["lap"] if state.prediction else 0
    if r["run"] != state.run_id or r["lap"] <= last:     # previous race run, duplicate or out of order
        stats["stale_predictions"] += 1
        return {"ok": False, "reason": "stale"}
    pred = {"type": "prediction", "t": round(state.race_t, 2), "lap": r["lap"],
            "source": r["source"], "probs": r["probs"]}
    state.update(pred)
    publish(json.dumps({"type": "batch", "race_t": round(state.race_t, 2), "events": [pred]}))
    stats["predictions_via_queue"] += 1
    return {"ok": True}

@app.get("/api/races")
def api_races():
    return {"current": current["name"], "races": list_races()}


@app.post("/api/race")
async def select_race(request: Request):
    name = (await request.json()).get("race")
    if name not in {r["name"] for r in list_races()}:
        return JSONResponse({"ok": False, "reason": "unknown race"}, status_code=404)
    if name != current["name"]:
        await start_race(name)
    return {"ok": True, "race": name}


if os.path.isdir("app/static"):                      # the dashboard page (Phase 3)
    app.mount("/", StaticFiles(directory="app/static", html=True), name="static")