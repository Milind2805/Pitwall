import json
import os

import joblib
import pandas as pd

from app.features import FEATURES, build_rows
import math


def win_probs(positions, gaps, lap, total_laps):
    """Softmax over -gap_to_leader. Early in the race the field is nearly flat;
    with few laps left the leader dominates."""
    remaining = max(0, total_laps - lap)
    sigma = 2.0 + 0.6 * remaining                  # seconds of "race uncertainty"
    scores = {}
    for d, pos in positions.items():
        if pos == 1:
            gap = 0.0
        else:
            g = gaps.get(d, {}).get("gap_to_leader")
            if isinstance(g, (int, float)):
                gap = g
            elif isinstance(g, str):               # "+1 LAP" and similar
                gap = 120.0
            else:                                  # no interval data yet
                gap = (pos - 1) * 3.0
        scores[d] = -gap / sigma
    if not scores:
        return {}
    m = max(scores.values())
    ex = {d: math.exp(s - m) for d, s in scores.items()}
    z = sum(ex.values())
    return {d: round(v / z, 4) for d, v in ex.items()}

MODEL_PATH = os.environ.get("PITWALL_MODEL", "models/win_model.joblib")
_models = {}


def model_for(race=None):
    """The model that never saw this race (its cross-validation fold), else the default model."""
    path = MODEL_PATH
    try:
        with open("models/race_fold.json") as f:
            fold = json.load(f).get(race)
        if fold is not None:
            path = f"models/fold_{fold}.joblib"
    except OSError:
        pass
    if path not in _models:
        _models[path] = joblib.load(path) if os.path.exists(path) else False
        print(f"loaded {path}" if _models[path] else f"no model at {path}: using the heuristic")
    return _models[path]


def live_probs(positions, gaps, last_lap, durs, lap, total_laps, race=None):
    model, built = model_for(race), build_rows(positions, gaps, last_lap, durs, lap, total_laps)
    if not model or built is None:
        return win_probs(positions, gaps, lap, total_laps), "heuristic"
    drivers, rows = built
    p = model.predict_proba(pd.DataFrame(rows, columns=FEATURES))[:, 1]
    p = p / max(p.sum(), 1e-12)
    return {d: round(float(x), 4) for d, x in zip(drivers, p)}, "model"