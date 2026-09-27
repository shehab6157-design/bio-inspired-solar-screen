"""
Step 5b: Placement map - how far from a window can the device live?

The real-year test showed a device right by a window has plenty of light,
so every controller survived. Deeper in a room the light drops fast.
This script moves the device from the window into the room and finds,
for each controller, where it stops surviving WINTER (the real bottleneck).

Position is given as the daylight factor: the share of outdoor light that
reaches the spot. Rough guide for a room with one ordinary window
(real rooms vary a lot with window size, glass, walls and furniture):
   3%   ~ right by the window
   2%   ~ 1-2 m in
   1%   ~ 3-4 m in
   0.5% ~ back of the room
   0.2% ~ far corner / hallway

Winter season: real PVGIS 2020 data for northern Israel, November-December
followed by January-February (121 days). Tested twice: with the 200 lux
evening lamp from the real-year test, and with daylight only (no lamp),
because a room lamp can hide how little daylight a spot really gets.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import day_sim as ds
import circadian as c
import real_data as rd

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
DOCS.mkdir(exist_ok=True)

POSITIONS = [                      # (daylight factor, rough place in the room)
    (0.03, "by the window"),
    (0.02, "1-2 m in"),
    (0.01, "3-4 m in"),
    (0.005, "back of the room"),
    (0.002, "far corner"),
]
N = 40
SURVIVE_BAR = 95.0                 # % of devices that must never fail


def winter_light(df_value, lamp=True):
    """Indoor minute light for Nov-Dec then Jan-Feb at one daylight factor."""
    old = (rd.DAYLIGHT_FACTOR, rd.EVENING_LED)
    rd.DAYLIGHT_FACTOR = df_value
    if not lamp:
        rd.EVENING_LED = (rd.EVENING_LED[0], rd.EVENING_LED[1], 0)
    try:
        times, watts = rd.load_pvgis()
        lux, sun, days, start = rd.indoor_minutes(times, watts)
    finally:
        rd.DAYLIGHT_FACTOR, rd.EVENING_LED = old
    dates = start + pd.to_timedelta(np.arange(days), unit="D")
    late = np.where(dates.month >= 11)[0]          # November, December
    early = np.where(dates.month <= 2)[0]          # January, February
    order = np.concatenate([late, early])
    pick = (order[:, None] * 1440 + np.arange(1440)).ravel()
    return lux[pick], sun[pick], len(order)


def main():
    p = ds.sample_params(np.random.default_rng(42), N)
    rows = []
    old = (c.DAYS, c.N)
    try:
        for lamp in (True, False):
            for df_value, place in POSITIONS:
                lux, sun, days = winter_light(df_value, lamp)
                c.DAYS, c.N = days, N
                for ctrl in c.CONTROLLERS:
                    r = c.run(ctrl, lux, sun, p)
                    ok = (r["missed_late"] == 0) & (r["dead_late"] == 0)
                    rows.append({
                        "lamp": "evening lamp" if lamp else "daylight only",
                        "daylight_factor_%": 100 * df_value,
                        "place": place,
                        "controller": ctrl,
                        "survived_winter_%": round(100 * ok.mean(), 1),
                        "updates_per_day": round(float(np.median(r["updates_day"])), 1),
                        "longest_gap_h": round(float(np.median(r["longest_gap_h"])), 1),
                    })
                print(f"done: {place} ({100 * df_value:g}% daylight, {'lamp' if lamp else 'no lamp'})")
    finally:
        c.DAYS, c.N = old

    df = pd.DataFrame(rows)
    df.to_csv(DOCS / "placement_results.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"\n=== Winter survival by position ({N} devices, real Nov-Feb light) ===")
    table = df.pivot_table(index=["lamp", "daylight_factor_%", "place"], columns="controller",
                           values="survived_winter_%", sort=False)[list(c.CONTROLLERS)]
    print(table.to_string())

    print(f"\n=== Placement limit: deepest spot where at least {SURVIVE_BAR:g}% of devices survive ===")
    for lamp in ("evening lamp", "daylight only"):
        for ctrl in c.CONTROLLERS:
            d = df[(df["lamp"] == lamp) & (df["controller"] == ctrl) & (df["survived_winter_%"] >= SURVIVE_BAR)]
            if len(d):
                deepest = d.sort_values("daylight_factor_%").iloc[0]
                print(f"{lamp:13s} {ctrl:9s}: works down to {deepest['daylight_factor_%']:g}% daylight "
                      f"({deepest['place']}), about {deepest['updates_per_day']:.0f} updates a day")
            else:
                print(f"{lamp:13s} {ctrl:9s}: does not reach the bar at any tested position")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    colors = {"fixed": "tab:red", "torpor": "tab:blue", "circadian": "tab:green"}
    for ax, lamp in zip(axes, ("evening lamp", "daylight only")):
        for ctrl in c.CONTROLLERS:
            d = df[(df["lamp"] == lamp) & (df["controller"] == ctrl)]
            ax.plot(d["daylight_factor_%"], d["survived_winter_%"], "o-", color=colors[ctrl], label=ctrl)
        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.set_xticks([q[0] * 100 for q in POSITIONS], [f"{q[0]*100:g}%\n{q[1]}" for q in POSITIONS], fontsize=8)
        ax.axhline(SURVIVE_BAR, color="black", ls="--", lw=1)
        ax.set_title(f"Winter, {lamp}")
        ax.legend()
    axes[0].set_ylabel("Devices surviving winter (%)")
    fig.suptitle("How far from the window? Real winter light, northern Israel")
    fig.tight_layout()
    fig.savefig(DOCS / "placement.png", dpi=110)
    print(f"\nSaved: {DOCS / 'placement_results.csv'} and {DOCS / 'placement.png'}")


if __name__ == "__main__":
    main()
