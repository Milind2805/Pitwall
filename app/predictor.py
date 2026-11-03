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