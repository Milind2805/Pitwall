import json
import pandas as pd
from fetch_openf1 import get_json, q, fmt

for r in ["2024_Monte_Carlo", "2024_Singapore", "2024_Yas_Marina_Circuit"]:
    d = f"data/{r}"
    drivers = pd.read_csv(f"{d}/drivers.csv")
    pos = pd.read_csv(f"{d}/positions.csv")
    sess = json.load(open(f"{d}/session.json"))
    start = pd.to_datetime(sess["date_start"], utc=True)
    missing = drivers[~drivers["name_acronym"].isin(pos["driver"].unique())]
    print(f"\n{r}: missing from positions.csv -> {missing['name_acronym'].tolist()}")
    for dn, acr in zip(missing["driver_number"], missing["name_acronym"]):
        rows = get_json("position", q(session_key=sess["session_key"], driver_number=dn) + f"&date<{fmt(start)}")
        print(f"  {acr}: {len(rows)} rows BEFORE session start", rows[-1] if rows else "")