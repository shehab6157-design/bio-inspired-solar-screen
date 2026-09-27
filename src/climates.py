"""
Step 6b: Hive screens in darker climates.

In northern Israel even January light kept every e-paper alive, so the bee
response thresholds (tired screens decline work) never had to act. This test
asks whether that safety layer matters where winters are much darker:
London, Berlin and Oslo (about 6 hours of daylight in January), using real
PVGIS 2020 sunlight for each city, January, daylight only and with a lamp.

Download the data first (from the project folder), one line per city:
  curl -s "https://re.jrc.ec.europa.eu/api/v5_2/seriescalc?lat=51.51&lon=-0.13&startyear=2020&endyear=2020&outputformat=csv" -o data/pvgis_london_2020.csv
  curl -s "https://re.jrc.ec.europa.eu/api/v5_2/seriescalc?lat=52.52&lon=13.40&startyear=2020&endyear=2020&outputformat=csv" -o data/pvgis_berlin_2020.csv
  curl -s "https://re.jrc.ec.europa.eu/api/v5_2/seriescalc?lat=59.91&lon=10.75&startyear=2020&endyear=2020&outputformat=csv" -o data/pvgis_oslo_2020.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import real_data as rd
import placement as pl
import hive as hv

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
DOCS.mkdir(exist_ok=True)

LOCATIONS = {                       # name: (data file, hours ahead of UTC in winter)
    "Northern Israel": ("pvgis_2020.csv", 2),
    "London": ("pvgis_london_2020.csv", 0),
    "Berlin": ("pvgis_berlin_2020.csv", 1),
    "Oslo": ("pvgis_oslo_2020.csv", 1),
}

ORIGINAL_LOADER = rd.load_pvgis


def use_location(fname, offset):
    """Point the real-data loader at one city's file.
    (load_pvgis fixes its default file when Python first reads it, so changing
    rd.DATA alone is not enough - the loader itself is swapped for this city.)"""
    path = ROOT / "data" / fname
    rd.DATA = path
    rd.UTC_OFFSET_H = offset
    rd.load_pvgis = lambda p=None: ORIGINAL_LOADER(path)


def january_light_at_1pct():
    """Average daily light energy (J per m^2) in January, 1% daylight, no lamp."""
    lux, sun, days = pl.winter_light(0.01, lamp=False)
    a = hv.START_DAY
    return rd.daily_energy_per_m2(lux, sun, days)[a:a + 31].mean()


def main():
    saved = (rd.DATA, rd.UTC_OFFSET_H)
    rng = np.random.default_rng(11)
    room, tv = hv.schedule()
    events = hv.arrivals(rng)
    ep, grid = hv.sample(rng)
    rows, light = [], {}
    try:
        for city, (fname, offset) in LOCATIONS.items():
            if not (ROOT / "data" / fname).exists():
                print(f"skip {city}: data/{fname} not found (see the curl lines at the top of this file)")
                continue
            use_location(fname, offset)
            light[city] = january_light_at_1pct()
            for lamp in (True, False):
                lux, sun = hv.room_light(lamp)
                for policy in hv.POLICIES:
                    r = hv.run(policy, room, tv, lux, sun, events, ep, grid)
                    rows.append({"city": city, "home": "evening lamp" if lamp else "daylight only",
                                 "policy": policy,
                                 **{k: round(float(np.median(v)), 1) for k, v in r.items() if k != "delivered"}})
            print(f"done: {city}")
    finally:
        rd.DATA, rd.UTC_OFFSET_H = saved
        rd.load_pvgis = ORIGINAL_LOADER

    base = light.get("Northern Israel")
    print("\n=== January light at a spot 3-4 m from a window (1% daylight) ===")
    for city, j in light.items():
        rel = f"  ({100 * j / base:.0f}% of northern Israel)" if base else ""
        print(f"{city:16s} {j:8.0f} J per m^2 per day{rel}")

    df = pd.DataFrame(rows)
    df.to_csv(DOCS / "climates_results.csv", index=False)
    pd.set_option("display.width", 220)
    cols = ["city", "home", "policy", "phone_wakes_day", "grid_J_day", "on_free_light_%",
            "epaper_dark_time_%", "dashboard_refresh_day"]
    print("\n=== Hive screens across climates (January, median household) ===")
    print(df[cols].to_string(index=False))

    d = df[(df["home"] == "daylight only") & (df["policy"].isin(["nearest", "hive"]))]
    cities = list(dict.fromkeys(d["city"]))
    x = np.arange(len(cities))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, (policy, col) in enumerate((("nearest", "tab:blue"), ("hive", "tab:green"))):
        vals = [d[(d.city == c) & (d.policy == policy)]["epaper_dark_time_%"].iloc[0] for c in cities]
        ax.bar(x + (i - 0.5) * 0.38, vals, 0.38, color=col, label=policy)
    ax.set_xticks(x, cities)
    ax.set_ylabel("Time an e-paper can't update (%)")
    ax.set_title("Do the bee thresholds protect screens in dark winters? (January, daylight only)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(DOCS / "climates.png", dpi=110)
    print(f"\nSaved: {DOCS / 'climates_results.csv'} and {DOCS / 'climates.png'}")


if __name__ == "__main__":
    main()
