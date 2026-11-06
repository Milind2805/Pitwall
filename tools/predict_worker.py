"""QueueForge worker: runs the win model for each 'predict' job and sends the result to the PitWall server.
Run from the project root:   python tools/predict_worker.py
"""
import json
import os
import socket
import sys
import time
import urllib.request

sys.path.insert(0, os.getcwd())

import pandas as pd

from app.features import FEATURES
from app.predictor import model_for

HOST, _, PORT = os.environ.get("QUEUEFORGE_ADDR", "127.0.0.1:9000").partition(":")
SERVER = os.environ.get("PITWALL_URL", "http://127.0.0.1:8000")


def run_job(payload):
    job = json.loads(payload)
    time.sleep(float(os.environ.get("WORKER_DELAY", "0")))     
    print(f"predict {job.get('race')} lap {job['lap']}")
    model = model_for(job.get("race"))
    if not model:
        raise RuntimeError("no model file")
    p = model.predict_proba(pd.DataFrame(job["rows"], columns=FEATURES))[:, 1]
    p = p / max(p.sum(), 1e-12)
    result = {"run": job["run"], "lap": job["lap"], "source": "model",
              "probs": {d: round(float(x), 4) for d, x in zip(job["drivers"], p)}}
    req = urllib.request.Request(SERVER + "/internal/prediction", json.dumps(result).encode(),
                                 {"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=5).read()


def main():
    f = socket.create_connection((HOST, int(PORT))).makefile("rw", encoding="utf-8", newline="\n")

    def cmd(line):
        f.write(line + "\n")
        f.flush()
        return f.readline().strip()

    print("predict worker ready")
    while True:
        reply = cmd("FETCH")
        if not reply:
            raise SystemExit("broker closed the connection")
        if reply == "EMPTY":
            time.sleep(0.05)
            continue
        _, job_id, job_type, payload = reply.split(" ", 3)
        try:
            if job_type != "predict":
                raise ValueError(f"unknown job type {job_type}")
            run_job(payload)
            cmd(f"ACK {job_id}")
        except Exception as ex:                       # the broker retries it, then dead-letters after 3 failures
            print(f"job {job_id} failed: {ex}")
            cmd(f"FAIL {job_id}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("worker stopped")