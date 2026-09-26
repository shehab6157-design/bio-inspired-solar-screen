import numpy as np
import pandas as pd
from layers import LAYERS, SCENARIOS_LUX

LUX_TO_WM2 = 0.0079          # rough daylight conversion
DEVICE_AREA_CM2 = 30          # e-reader/sensor-class screen area
LOAD_MW_STANDBY = (1, 5)
LOAD_MW_ACTIVE = (50, 100)
N_TRIALS = 5000

def sample(rng, low, high):
    return low + rng.random() * (high - low)

def run():
    rng = np.random.default_rng(42)
    rows = []
    for scenario, lux in SCENARIOS_LUX.items():
        irradiance_wm2 = lux * LUX_TO_WM2
        for _ in range(N_TRIALS):
            eff = sample(rng, *LAYERS["indoor_cell_baseline"]["efficiency"])
            area_m2 = DEVICE_AREA_CM2 / 10000
            harvested_w = irradiance_wm2 * area_m2 * eff
            harvested_mw = harvested_w * 1000
            load_mw = sample(rng, *LOAD_MW_STANDBY)
            rows.append({
                "scenario": scenario,
                "lux": lux,
                "harvested_mw": harvested_mw,
                "load_mw": load_mw,
                "net_mw": harvested_mw - load_mw,
            })
    df = pd.DataFrame(rows)
    summary = df.groupby("scenario")[["harvested_mw", "load_mw", "net_mw"]].agg(["mean", "min", "max"])
    print(summary)
    df.to_csv("../data/simulation_raw.csv", index=False)
    summary.to_csv("../data/simulation_summary.csv")

if __name__ == "__main__":
    run()
