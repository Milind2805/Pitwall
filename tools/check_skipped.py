import pandas as pd

for r in ["2024_Monte_Carlo", "2024_Singapore", "2024_Yas_Marina_Circuit"]:
    d = f"data/{r}"
    pos = pd.read_csv(f"{d}/positions.csv").sort_values("t")
    laps = pd.read_csv(f"{d}/laps.csv")
    last = pos.groupby("driver").agg(position=("position", "last"), t=("t", "last")).sort_values("position")
    print(r, "| lap rows:", len(laps), "| max lap:", laps["lap_number"].max(), "| drivers with position data:", len(last))
    print(last.head(4).to_string(), "\n")