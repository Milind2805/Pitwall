"""Test client: connect to the PitWall WebSocket and measure what arrives.

Usage:  python tools/ws_client.py                 # 1 client, 10 seconds
        python tools/ws_client.py --clients 50    # 50 clients at once
"""
import argparse
import asyncio
import json
import time

import websockets


async def one_client(url, seconds):
    msgs = events = 0
    async with websockets.connect(url, max_size=None) as ws:
        snap = json.loads(await ws.recv())
        t0 = time.monotonic()
        last_t = snap["race_t"]
        while time.monotonic() - t0 < seconds:
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
            msgs += 1
            events += len(msg.get("events", []))
            last_t = msg["race_t"]
    return snap, msgs, events, time.monotonic() - t0, last_t


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://localhost:8000/ws")
    ap.add_argument("--seconds", type=float, default=10)
    ap.add_argument("--clients", type=int, default=1)
    a = ap.parse_args()

    results = await asyncio.gather(*[one_client(a.url, a.seconds) for _ in range(a.clients)])
    snap = results[0][0]
    print(f"snapshot: {len(snap['drivers'])} drivers, {len(snap['cars'])} cars, "
          f"{len(snap['laps'])} laps so far, race_t={snap['race_t']}")
    avg = lambda i: sum(r[i] for r in results) / len(results)
    secs = avg(3)
    print(f"{a.clients} client(s) for {secs:.1f}s: {avg(1)/secs:.1f} messages/s, "
          f"{avg(2)/secs:.0f} events/s per client, last race_t={results[0][4]}")


asyncio.run(main())                                       