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

import joblib
import pandas as pd

from app.features import FEATURES

HOST, _, PORT = os.environ.get("QUEUEFORGE_ADDR", "127.0.0.1:9000").partition(":")
SERVER = os.environ.get("PITWALL_URL", "http://127.0.0.1:8000")
MODEL_PATH = os.environ.get("PITWALL_MODEL", "models/win_model.joblib")


def run_job(model, payload):
    job = json.loads(payload)
    p = model.predict_proba(pd.DataFrame(job["rows"], columns=FEATURES))[:, 1]
    p = p / max(p.sum(), 1e-12)
    result = {"run": job["run"], "lap": job["lap"], "source": "model",
              "probs": {d: round(float(x), 4) for d, x in zip(job["drivers"], p)}}
    req = urllib.request.Request(SERVER + "/internal/prediction", json.dumps(result).encode(),
                                 {"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=5).read()


def main():
    model = joblib.load(MODEL_PATH)
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
            run_job(model, payload)
            cmd(f"ACK {job_id}")
        except Exception as ex:                       # the broker retries it, then dead-letters after 3 failures
            print(f"job {job_id} failed: {ex}")
            cmd(f"FAIL {job_id}")


if __name__ == "__main__":
    main()