"""
Step 5 (advanced): Testing on real-world sunlight data.

Every light schedule so far was invented. Here the controllers face a
REAL year of sunlight: hourly data for northern Israel (lat 32.69,
lon 35.42) from PVGIS, the European Commission's solar database
(SARAH-2 satellite record, year 2020): winter cloud spells, short
December days, long summer days.

The device: the e-paper device from day_sim.py sitting indoors near a
window. Indoor daylight = outdoor sunlight x a daylight factor (the share
of outdoor light that reaches a spot inside; 3% is typical near a window),
plus LED room light in the evening.

What it checks:
  1. Reality check: real daily light energy per month vs the invented
     "near_window" schedule the earlier steps assumed.
  2. The three controllers (fixed, reactive torpor, circadian) over the
     whole year: failures, updates per day, longest gap, worst month.

Get the data first (from the project folder):
  curl -s "https://re.jrc.ec.europa.eu/api/v5_2/seriescalc?lat=32.69&lon=35.42&startyear=2020&endyear=2020&outputformat=csv" -o data/pvgis_2020.csv
"""
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import day_sim as ds
import circadian as c

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
DATA = ROOT / "data" / "pvgis_2020.csv"
DOCS.mkdir(exist_ok=True)

UTC_OFFSET_H = 2          # Israel standard time (summer time ignored, noted as a limit)
DAYLIGHT_FACTOR = 0.03    # 3% of outdoor light reaches the device near the window
EVENING_LED = (17, 22, 200)   # LED room light: from 17:00 to 22:00 at 200 lux
N = 60                    # simulated devices (fewer than before: a full year is long)
ROW = re.compile(r"^(\d{8}):(\d{2})(\d{2}),([-\d.]+)")


def load_pvgis(path=DATA):
    """Returns hourly outdoor sunlight (W/m^2) and their UTC timestamps."""
    times, watts = [], []
    for line in Path(path).read_text().splitlines():
        m = ROW.match(line)
        if m:
            times.append(pd.Timestamp(f"{m.group(1)} {m.group(2)}:00"))
            watts.append(max(0.0, float(m.group(4))))
    if not watts:
        raise ValueError(f"No hourly rows found in {path}. Re-download it with the curl line in this file.")
    return pd.DatetimeIndex(times), np.array(watts)


def indoor_minutes(times, watts):
    """Minute-by-minute indoor light, in the 'lux' units the controllers use."""
    local = times + pd.Timedelta(hours=UTC_OFFSET_H)
    start = local[0].normalize()
    offset = int((local[0] - start) / pd.Timedelta(hours=1))
    hourly = np.concatenate([np.zeros(offset), watts])
    days = len(hourly) // 24
    hourly = hourly[:days * 24]
    window_w = np.repeat(hourly, 60) * DAYLIGHT_FACTOR
    lux = window_w / ds.SUN_K
    sun = window_w > 0
    h0, h1, led = EVENING_LED
    minute_of_day = np.tile(np.arange(1440), days)
    evening = (minute_of_day >= h0 * 60) & (minute_of_day < h1 * 60)
    use_led = evening & (lux < led)
    lux = np.where(use_led, led, lux)
    sun = np.where(use_led, False, sun)
    return lux, sun, days, start


def daily_energy_per_m2(lux, sun, days):
    k = np.where(sun, ds.SUN_K, ds.LED_K)
    return (lux * k * 60).reshape(days, 1440).sum(axis=1)


def main():
    times, watts = load_pvgis()
    lux, sun, days, start = indoor_minutes(times, watts)
    months = (start + pd.to_timedelta(np.arange(days), unit="D")).month
    print(f"Loaded {len(watts)} real hours ({days} days) starting {start.date()}")

    real = daily_energy_per_m2(lux, sun, days)
    fake_lux, fake_sun = ds.light_per_minute(ds.PROFILES["near_window"])
    assumed = daily_energy_per_m2(fake_lux, fake_sun, 1)[0]
    check = pd.DataFrame({"month": months, "real": real}).groupby("month")["real"].agg(["mean", "min"])
    check["assumed"] = assumed
    check["real_vs_assumed_%"] = 100 * check["mean"] / assumed
    print("\n=== Reality check: light energy reaching the device, J per m^2 per day ===")
    print(check.round(0).to_string())

    old = (c.DAYS, c.N)
    c.DAYS, c.N = days, N
    p = ds.sample_params(np.random.default_rng(42), N)
    rows, traces = [], {}
    try:
        for ctrl in c.CONTROLLERS:
            r = c.run(ctrl, lux, sun, p)
            traces[ctrl] = r["trace"]
            ok = (r["missed_late"] == 0) & (r["dead_late"] == 0)
            daily_soc = r["trace"].reshape(days, 1440).min(axis=1)
            worst_month = pd.Series(daily_soc).groupby(months).mean().idxmin()
            rows.append({
                "controller": ctrl,
                "no_fail_%": round(100 * ok.mean(), 1),
                "updates_per_day": round(float(np.median(r["updates_day"])), 1),
                "worst_day_updates": round(float(np.median(r["worst_day"])), 1),
                "longest_gap_h": round(float(np.median(r["longest_gap_h"])), 1),
                "worst_month": int(worst_month),
            })
            print(f"done: {ctrl}")
    finally:
        c.DAYS, c.N = old

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(f"\n=== Real year of sunlight, {N} devices, scored after the first week of learning ===")
    print(df.to_string(index=False))
    df.to_csv(DOCS / "real_year_results.csv", index=False)
    check.round(1).to_csv(DOCS / "real_vs_assumed_light.csv")

    fig, axes = plt.subplots(2, 1, figsize=(11, 7))
    axes[0].bar(check.index, check["mean"], label="real (PVGIS 2020)")
    axes[0].axhline(assumed, color="black", ls="--", label="what the earlier models assumed")
    axes[0].set_ylabel("J per m^2 per day")
    axes[0].set_title("Light reaching a device near a window, northern Israel")
    axes[0].set_xticks(range(1, 13))
    axes[0].legend()
    x = np.arange(days * 1440) / 1440
    for ctrl, col in (("fixed", "tab:red"), ("torpor", "tab:blue"), ("circadian", "tab:green")):
        axes[1].plot(x, 100 * traces[ctrl], color=col, lw=0.6, label=ctrl)
    axes[1].set_xlabel("Day of year")
    axes[1].set_ylabel("Stored energy % (median device)")
    axes[1].set_ylim(0, 105)
    axes[1].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(DOCS / "real_year.png", dpi=110)
    print(f"\nSaved: real_year_results.csv, real_vs_assumed_light.csv and real_year.png in {DOCS}")


if __name__ == "__main__":
    main()
