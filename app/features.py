"""Features for the win model."""
import statistics

FEATURES = ["laps_remaining", "frac_done", "position", "gap", "interval",
            "tyre_age", "compound", "pits_done", "pace_delta", "lapped"]
COMPOUND = {"SOFT": 0, "MEDIUM": 1, "HARD": 2}      # anything else -> 3


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v


def _gap(pos, raw):
    """(gap to leader in seconds, is_lapped), with the same fallbacks as training."""
    if pos == 1:
        return 0.0, False
    lapped = isinstance(raw, str) and "LAP" in raw
    val = raw if _is_num(raw) else (90.0 if lapped else (pos - 1) * 3.0)
    return min(val, 120.0), lapped


def build_rows(positions, gaps, last_lap, durs, lap, total_laps):
    """positions {drv: pos}, gaps {drv: {gap_to_leader, interval}}, last_lap {drv: lap event},
    durs {drv: last 3 lap durations}. Returns (drivers, rows) or None if the model should not run."""
    drivers = [d for d in positions
    if d in last_lap and durs.get(d) and (last_lap[d].get("lap_number") or 0) >= lap - 2]
    if len(drivers) < 10 or lap >= total_laps:
        return None
    pace = {d: statistics.median(durs[d]) for d in drivers}
    mid = statistics.median(pace.values())
    rows = []
    for d in drivers:
        pos, g, ll = positions[d], gaps.get(d, {}), last_lap[d]
        gap, lapped = _gap(pos, g.get("gap_to_leader"))
        iv = g.get("interval")
        rows.append([
            total_laps - lap, round(lap / total_laps, 4), pos, gap,
            min(iv, 60.0) if _is_num(iv) else 0.0,
            ll.get("tyre_age") or 0, COMPOUND.get(ll.get("compound") or "", 3),
            (ll.get("stint") or 1) - 1,
            max(-10.0, min(10.0, pace[d] - mid)), int(lapped),
        ])
    return drivers, rows