"""
Step 11: The waggle dance - screens sharing a light forecast.

A honeybee that finds flowers dances to tell the hive where they are, so
the others don't each have to search alone.

In the hive home (Step 6), each light-powered e-paper plans its own energy.
To plan well it needs to know whether TODAY is darker than normal (a cloudy
day, a dark spell). A screen deep in a room gets so little light that its own
reading is weak and noisy. The screen by the window measures the day's light
best, so it "dances" a weather reading to the others: today's light compared
with a normal day.

Three ways to run the five e-papers of a home (real January 2020 sunlight,
daylight only, 30 homes, 28 days, the first week used for learning):
  torpor         : react only when the store is already low (today's hive)
  self-forecast  : each screen learns its normal day and judges today from
                   its OWN noisy readings, then plans ahead
  waggle dance   : the same planning, but every screen uses the WINDOW
                   screen's reading of today's light

Sensor noise is realistic: +-10% per reading plus a noise floor of about
1.5 lux, which matters most for dim screens.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import day_sim as ds
import hive as hv
import climates as cl

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
DOCS.mkdir(exist_ok=True)

POLICIES = ["torpor", "self-forecast", "waggle dance"]
WINDOW_ROOM = hv.ROOMS.index("kitchen")      # the screen closest to a window (2% daylight)
LEARN_DAYS = 7
HORIZON_H = np.arange(1, 49)                 # plan up to 48 hours ahead
PERSIST = 0.6                                # a dark day tends to be followed by another (weight per day)
NOISE_LUX = 1.5
MIN_I, MAX_I = 5.0, 480.0
TRUST = 0.8


def run(policy, lux, sun, ep, seed=5):
    rng = np.random.default_rng(seed)
    H, R = hv.H, len(hv.ROOMS)
    cap = ds.DEVICE["storage_j"]
    reserve = cap * hv.RESERVE
    minutes = lux.shape[1]
    days = minutes // 1440
    e = np.full((H, R), cap * ds.DEVICE["start_soc"])
    next_dash = np.zeros((H, R))
    dark = np.zeros((H, R)); refresh = np.zeros((H, R))
    memory = np.zeros((24, H, R)); seen = np.zeros(24, dtype=bool)
    hour_true = np.zeros((H, R))
    today_obs = np.zeros((H, R)); today_exp = np.zeros((H, R))
    ratio = np.ones((H, R))
    noise_floor = NOISE_LUX * ds.LED_K * ep["eff_led"] * ep["area_m2"] * 3600 * ep["conv"]
    fcum = np.zeros((len(HORIZON_H), H, R))

    for t in range(minutes):
        day, m = divmod(t, 1440)
        hour = m // 60
        late = day >= LEARN_DAYS
        k = np.where(sun[:, t], ds.SUN_K * ep["eff_sun"], ds.LED_K * ep["eff_led"])
        h = lux[:, t] * k * ep["area_m2"] * 60 * ep["conv"]
        hour_true += h
        e = e + h - ep["sleep_w"] * 60

        if policy != "torpor" and m % 60 == 0:
            # forecast from the learned day, bent by how today compares with normal
            share = ratio if policy == "self-forecast" else np.repeat(ratio[:, WINDOW_ROOM:WINDOW_ROOM + 1], R, axis=1)
            ahead_hours = (hour + HORIZON_H) % 24
            day_weight = PERSIST ** (HORIZON_H / 24.0)
            factor = 1 + (share[None] - 1) * day_weight[:, None, None]
            hourly = memory[ahead_hours] * factor
            fcum = np.cumsum(hourly, axis=0)

        due = t >= next_dash
        if due.any():
            ok = due & (e - ep["cost_j"] >= reserve)
            e = np.where(ok, e - ep["cost_j"], e)
            refresh += ok * late
            soc = e / cap
            reactive = ds.interval_minutes("torpor", soc)
            if policy == "torpor":
                iv = reactive
            else:
                hrs = HORIZON_H[:, None, None]
                budget = e[None] + TRUST * fcum - reserve - ep["sleep_w"][None] * hrs * 3600
                rate = (np.maximum(budget, 0) / ep["cost_j"][None] / (hrs * 60)).min(axis=0)
                pred = np.clip(1 / np.maximum(rate, 1e-9), MIN_I, MAX_I)
                iv = np.where(soc < 0.4, np.maximum(pred, reactive), pred)
            next_dash = np.where(due, t + iv, next_dash)
        e = np.clip(e, 0, cap)
        dark += (e - ep["cost_j"] < reserve) * late

        if m % 60 == 59:                           # end of the hour: measure (noisily) and learn
            measured = np.maximum(hour_true * (1 + rng.normal(0, 0.1, (H, R)))
                                  + rng.normal(0, 1, (H, R)) * noise_floor, 0)
            if seen[hour]:
                today_exp += memory[hour]
                today_obs += measured
                trusted = today_exp > 10 * noise_floor
                ratio = np.where(trusted, np.clip(today_obs / np.maximum(today_exp, 1e-12), 0.1, 2.0), ratio)
                memory[hour] = 0.8 * memory[hour] + 0.2 * measured
            else:
                memory[hour] = measured
                seen[hour] = True
            hour_true = np.zeros((H, R))
            if hour == 23:                         # new day: keep the ratio (dark spells persist)
                today_obs[:] = 0; today_exp[:] = 0

    scored = (days - LEARN_DAYS) * 1440
    return {
        "dark_%": 100 * dark.mean(axis=1) / scored,
        "worst_room_dark_%": 100 * dark.max(axis=1) / scored,
        "refresh_day": refresh.mean(axis=1) / (days - LEARN_DAYS),
    }


def main():
    rng = np.random.default_rng(11)
    ep, _ = hv.sample(rng)
    rows = []
    saved = (cl.rd.DATA, cl.rd.UTC_OFFSET_H)
    try:
        for city, (fname, offset) in cl.LOCATIONS.items():
            if not (ROOT / "data" / fname).exists():
                print(f"skip {city}: data/{fname} not found")
                continue
            cl.use_location(fname, offset)
            lux, sun = hv.room_light(lamp=False)
            for policy in POLICIES:
                r = run(policy, lux, sun, ep)
                rows.append({"city": city, "policy": policy,
                             **{k: round(float(np.median(v)), 1) for k, v in r.items()}})
            print(f"done: {city}")
    finally:
        cl.rd.DATA, cl.rd.UTC_OFFSET_H = saved
        cl.rd.load_pvgis = cl.ORIGINAL_LOADER

    df = pd.DataFrame(rows)
    df.to_csv(DOCS / "waggle_results.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n=== Home e-papers in January, daylight only (median home, after one week of learning) ===")
    print("dark_% = time a screen can't update; worst_room = the home's darkest screen")
    print(df.to_string(index=False))

    cities = list(dict.fromkeys(df["city"]))
    x = np.arange(len(cities))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    colors = {"torpor": "tab:gray", "self-forecast": "tab:blue", "waggle dance": "tab:green"}
    for i, pol in enumerate(POLICIES):
        d = df[df["policy"] == pol].set_index("city").reindex(cities)
        axes[0].bar(x + (i - 1) * 0.27, d["worst_room_dark_%"], 0.27, color=colors[pol], label=pol)
        axes[1].bar(x + (i - 1) * 0.27, d["refresh_day"], 0.27, color=colors[pol], label=pol)
    for ax, title in zip(axes, ["Darkest screen: time it can't update (%)", "Screen updates per day"]):
        ax.set_xticks(x, cities)
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.suptitle("The waggle dance: sharing the window screen's light forecast (real January light)")
    fig.tight_layout()
    fig.savefig(DOCS / "waggle.png", dpi=110)
    print(f"\nSaved: {DOCS / 'waggle_results.csv'} and {DOCS / 'waggle.png'}")


if __name__ == "__main__":
    main()
