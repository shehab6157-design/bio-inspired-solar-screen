"""
Step 2 (advanced): Circadian torpor - a predictive energy controller.

Reactive torpor (day_sim.py) only slows down once the store is already low.
Animals do better: they learn the rhythm of day and night, and even of the
week and the seasons, and prepare BEFORE the lean time comes (a bear builds
fat before winter, a hummingbird enters torpor as night falls).

The circadian controller:
  - learns how much energy each hour of each weekday brings (a 7 x 24 memory)
  - forecasts harvest hour by hour up to 7 days ahead
  - finds the leanest stretch ahead (like a dark weekend) and spends at a
    rate that survives it, so it neither starves before the dark nor
    wastes energy when the store is full

Compared on 28 days of changing conditions:
  fixed      : refresh every 10 minutes
  torpor     : reactive, slows down when the store is low
  circadian  : predictive, plans from the learned daily and weekly rhythm

Honest note: forecast-based energy budgeting is known in energy-harvesting
sensor research ("energy-neutral operation"). What is added here is the
weekday rhythm memory and testing it inside the bio-inspired screen system.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import day_sim as ds

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
DOCS.mkdir(exist_ok=True)

DAYS = 28
N = 150
MIN_I, MAX_I = 5.0, 480.0      # circadian refresh limits, minutes (can hibernate longer
                               # than reactive torpor, because it knows a long dark is coming)
LEARN_DAYS = 8                 # first week + its weekend = learning; scored separately
FORECAST_TRUST = 0.8           # plan with 80% of forecast (safety margin)
LEARN_RATE = 0.5               # how fast the memory adapts to new days
CONTROLLERS = ["fixed", "torpor", "circadian"]


# ---------- 28-day light scenarios ----------

def build_scenario(name, seed=3):
    """Per-minute lux and sun flag for DAYS days."""
    rng = np.random.default_rng(seed)
    lux = np.zeros(DAYS * 1440)
    sun = np.zeros(DAYS * 1440, dtype=bool)
    cloudy = False
    for d in range(DAYS):
        if name == "office_weekends":
            base = ds.PROFILES["office_desk"]
            factor = 0.0 if d % 7 in (5, 6) else 1.0      # office closed at weekends
        elif name == "cloudy_window":
            base = ds.PROFILES["near_window"]
            cloudy = rng.random() < (0.6 if cloudy else 0.3)  # cloudy spells come in runs
            factor = rng.uniform(0.1, 0.35) if cloudy else 1.0
        else:  # dim_home
            base = ds.PROFILES["home_dim"]
            factor = 1.0
        day_lux, day_sun = ds.light_per_minute(base)
        scale = np.where(day_sun, factor, 1.0) if name == "cloudy_window" else factor
        lux[d * 1440:(d + 1) * 1440] = day_lux * scale
        sun[d * 1440:(d + 1) * 1440] = day_sun
    return lux, sun


SCENARIOS = ["office_weekends", "cloudy_window", "dim_home"]


# ---------- controllers ----------

HORIZONS_H = np.arange(1, 169)        # plan windows: every hour up to 7 days ahead


def interval_for(controller, e, cap, cost, sleep_w, fcum):
    """fcum: forecast harvest summed up to each horizon, shape (len(HORIZONS_H), N)."""
    reactive = ds.interval_minutes("torpor", e / cap)
    if controller == "fixed":
        return np.full(e.shape, 10.0)
    if controller == "torpor":
        return reactive
    # circadian: find the leanest window ahead and spend at a rate that survives it,
    # like a bear rationing fat for the whole winter, not just tonight
    hours = HORIZONS_H[:, None]
    budget = (e[None, :] + FORECAST_TRUST * fcum
              - cap * ds.DEVICE["reserve_soc"] - sleep_w[None, :] * hours * 3600)
    per_minute = np.maximum(budget, 0) / cost[None, :] / (hours * 60)
    rate = per_minute.min(axis=0)
    predictive = np.clip(1.0 / np.maximum(rate, 1e-9), MIN_I, MAX_I)
    # reflex floor: never spend faster than reactive torpor would when the store is low
    return np.where(e / cap < 0.4, np.maximum(predictive, reactive), predictive)


def forecast_cumulative(memory, seen, day, hour, ahead):
    """Forecast harvest for the next 168 hours from the weekday x hour memory."""
    abs_h = day * 24 + hour + ahead
    wd, hd = (abs_h // 24) % 7, abs_h % 24
    counts = seen.sum(axis=0)
    fallback = (memory * seen[:, :, None]).sum(axis=0) / np.maximum(counts, 1)[:, None]
    hourly = np.where(seen[wd, hd][:, None], memory[wd, hd], fallback[hd])
    cum = np.cumsum(hourly, axis=0)
    return cum[HORIZONS_H - 1]


def run(controller, lux, sun, p):
    cap = ds.DEVICE["storage_j"]
    reserve = cap * ds.DEVICE["reserve_soc"]
    minutes = len(lux)
    e = np.full(N, cap * ds.DEVICE["start_soc"])
    next_up = np.zeros(N)
    last_ok = np.zeros(N)
    longest_gap = np.zeros(N)
    missed = np.zeros(N)
    dead = np.zeros(N)
    missed_late = np.zeros(N)
    dead_late = np.zeros(N)
    gap_late = np.zeros(N)
    daily_updates = np.zeros((DAYS, N))
    memory = np.zeros((7, 24, N))
    seen = np.zeros((7, 24), dtype=bool)
    hour_harvest = np.zeros(N)
    trace = np.zeros(minutes)
    fcum = np.zeros((len(HORIZONS_H), N))
    ahead = np.arange(1, 169)

    for t in range(minutes):
        day, m = divmod(t, 1440)
        hour = m // 60
        if controller == "circadian" and m % 60 == 0:
            fcum = forecast_cumulative(memory, seen, day, hour, ahead)
        k = ds.SUN_K * p["eff_sun"] if sun[t] else ds.LED_K * p["eff_led"]
        h = lux[t] * k * p["area_m2"] * 60 * p["conv"]
        hour_harvest += h
        e += h - p["sleep_w"] * 60

        due = t >= next_up
        if due.any():
            can = due & (e - p["cost_j"] >= reserve)
            e = np.where(can, e - p["cost_j"], e)
            daily_updates[day] += can
            missed += due & ~can
            gap_now = np.where(can, t - last_ok, 0)
            longest_gap = np.maximum(longest_gap, gap_now)
            if day >= LEARN_DAYS:
                missed_late += due & ~can
                gap_late = np.maximum(gap_late, gap_now)
            last_ok = np.where(can, t, last_ok)

            iv = interval_for(controller, e, cap, p["cost_j"], p["sleep_w"], fcum)
            next_up = np.where(due, t + iv, next_up)

        if m % 60 == 59:  # end of the hour: learn it
            wd = day % 7
            if seen[wd, hour]:
                memory[wd, hour] = (1 - LEARN_RATE) * memory[wd, hour] + LEARN_RATE * hour_harvest
            else:
                memory[wd, hour] = hour_harvest
                seen[wd, hour] = True
            hour_harvest = np.zeros(N)

        dead += e <= 0
        if day >= LEARN_DAYS:
            dead_late += e <= 0
        e = np.clip(e, 0, cap)
        trace[t] = np.median(e / cap)

    longest_gap = np.maximum(longest_gap, minutes - last_ok)
    gap_late = np.maximum(gap_late, minutes - last_ok)
    late = daily_updates[LEARN_DAYS:]
    return {
        "missed": missed, "dead": dead, "trace": trace,
        "missed_late": missed_late, "dead_late": dead_late,
        "updates_day": late.mean(axis=0),
        "worst_day": late.min(axis=0),
        "longest_gap_h": gap_late / 60,
    }


def main():
    p = ds.sample_params(np.random.default_rng(42), N)
    rows, traces = [], {}
    for sc_name in SCENARIOS:
        lux, sun = build_scenario(sc_name)
        for ctrl in CONTROLLERS:
            r = run(ctrl, lux, sun, p)
            traces[(sc_name, ctrl)] = r["trace"]
            ok_all = (r["missed"] == 0) & (r["dead"] == 0)
            ok_late = (r["missed_late"] == 0) & (r["dead_late"] == 0)
            rows.append({
                "scenario": sc_name,
                "controller": ctrl,
                "no_fail_after_learning_%": round(100 * ok_late.mean(), 1),
                "no_fail_all_28d_%": round(100 * ok_all.mean(), 1),
                "updates_per_day": round(float(np.median(r["updates_day"])), 1),
                "worst_day_updates": round(float(np.median(r["worst_day"])), 1),
                "longest_gap_h": round(float(np.median(r["longest_gap_h"])), 1),
            })
            print(f"done: {sc_name} / {ctrl}")

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("\n=== 28 days, 150 devices each (scored from day 8, after the first full week of learning) ===")
    print(df.to_string(index=False))
    df.to_csv(DOCS / "circadian_results.csv", index=False)

    fig, axes = plt.subplots(len(SCENARIOS), 1, figsize=(11, 9), sharex=True)
    days = np.arange(DAYS * 1440) / 1440
    colors = {"fixed": "tab:red", "torpor": "tab:blue", "circadian": "tab:green"}
    for ax, sc_name in zip(axes, SCENARIOS):
        for ctrl in CONTROLLERS:
            ax.plot(days, 100 * traces[(sc_name, ctrl)], color=colors[ctrl], label=ctrl, lw=1.2)
        ax.set_title(sc_name)
        ax.set_ylabel("Stored %")
        ax.set_ylim(0, 105)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Day")
    fig.tight_layout()
    fig.savefig(DOCS / "circadian.png", dpi=110)
    print(f"\nSaved: {DOCS / 'circadian_results.csv'} and {DOCS / 'circadian.png'}")


if __name__ == "__main__":
    main()
