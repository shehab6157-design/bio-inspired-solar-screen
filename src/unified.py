"""
Step 3 (advanced): Unified bio-controller.

Until now each idea worked alone:
  chameleon mode chose the screen type, torpor chose how often to refresh.
Real animals don't run separate systems: one energy budget decides how
active to be, how much to show, how much to save for later.

Test device: a hybrid colour e-reader (reflective + glowing screen, 90 cm^2)
with a 6 Wh battery and a small indoor solar cell. Someone reads it every day:
mornings, a sunny lunch break, long evenings in dim light, weekends by a window.

Strategies compared over 60 days (150 simulated devices each):
  glow_only  : a normal glowing (OLED) screen, background sync every 10 min
  chameleon  : reflective in bright light, glowing in dim light, torpor sync
  unified    : chameleon + ONE shared energy plan (a glide path from a full
               battery to a 5% reserve on the chosen charge day) that decides
               - whether dim-light reading uses the glowing screen (best look)
                 or reflective + front light (cheaper), and
               - how often to sync in the background.

Scores: days until the battery is empty, hours a day on the cheaper
front-light look, and background syncs per day.
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

DAYS = 60
N = 150
SCREEN_CM2 = 90
BATTERY_J = 6 * 3600          # 6 Wh battery (typical e-reader size)
TARGET_DAYS = 30              # user's wish: charge at most once a month
SWITCH_LUX = 300
STRATEGIES = ["glow_only", "chameleon", "unified"]

PARAMS = {
    "emissive_base_w": (0.20, 0.35),       # per 100 cm^2, indoor brightness
    "emissive_sun_extra_w": (0.80, 1.40),  # per 100 cm^2, full sunlight
    "frontlight_w": (0.05, 0.15),          # per 100 cm^2
    "page_turn_mj": (20, 80),              # reflective page refresh
    "sync_mj": (50, 200),                  # background wireless sync
}

# (start_hour, end_hour, lux, source) while reading; weekdays and weekends
READING_WEEKDAY = [(7.5, 8.0, 150, "led"), (12.5, 13.0, 20000, "sun"), (21.0, 22.5, 60, "led")]
READING_WEEKEND = [(10.0, 11.0, 2000, "sun"), (21.0, 22.5, 60, "led")]


def build_days():
    """Per-minute ambient lux, sun flag and reading flag for DAYS days."""
    base_lux, base_sun = ds.light_per_minute(ds.PROFILES["home_dim"])
    lux = np.tile(base_lux, DAYS)
    sun = np.tile(base_sun, DAYS)
    reading = np.zeros(DAYS * 1440, dtype=bool)
    for d in range(DAYS):
        sessions = READING_WEEKEND if d % 7 in (5, 6) else READING_WEEKDAY
        for h0, h1, l, s in sessions:
            a, b = d * 1440 + int(h0 * 60), d * 1440 + int(h1 * 60)
            lux[a:b], sun[a:b], reading[a:b] = l, s == "sun", True
    return lux, sun, reading


def sample(rng):
    p = ds.sample_params(rng, N)
    for k, v in PARAMS.items():
        p[k] = rng.uniform(*v, N)
    scale = SCREEN_CM2 / 100
    p["em_w"] = lambda lux: scale * (p["emissive_base_w"] + p["emissive_sun_extra_w"] * min(1, lux / 20000))
    p["front_w"] = scale * p["frontlight_w"]
    p["reflect_w"] = p["page_turn_mj"] * 1e-3 * 2 / 60   # one page every 30 s
    return p


def run(strategy, lux, sun, reading, p):
    e = np.full(N, float(BATTERY_J))
    empty_day = np.full(N, np.nan)
    front_min = np.zeros(N)
    syncs = np.zeros(N)
    next_sync = np.zeros(N)

    for t in range(len(lux)):
        # unified: a glide path from full to a 5% reserve on the target day.
        # Above the path there is energy to spare (glow, sync often); below it, save.
        on_track = e >= BATTERY_J * (0.05 + 0.95 * (1 - t / (TARGET_DAYS * 1440)))

        alive = np.isnan(empty_day)            # an empty device does nothing until recharged
        k = ds.SUN_K * p["eff_sun"] if sun[t] else ds.LED_K * p["eff_led"]
        h = lux[t] * k * p["area_m2"] * 60 * p["conv"]
        use = p["sleep_w"] * 60

        if reading[t]:
            if strategy == "glow_only":
                use = use + p["em_w"](lux[t]) * 60
            elif lux[t] >= SWITCH_LUX:
                use = use + p["reflect_w"] * 60
            else:
                glow = p["em_w"](lux[t]) * 60
                cheap = (p["reflect_w"] + p["front_w"]) * 60
                if strategy == "chameleon":
                    use = use + glow
                else:  # unified: glow only while ahead of the glide path
                    ok = on_track
                    use = use + np.where(ok, glow, cheap)
                    front_min += ~ok & alive

        due = t >= next_sync
        if due.any():
            use = use + np.where(due, p["sync_mj"] * 1e-3, 0)
            syncs += due & alive
            soc = e / BATTERY_J
            if strategy == "glow_only":
                iv = np.full(N, 10.0)
            elif strategy == "chameleon":
                iv = ds.interval_minutes("torpor", soc)
            else:
                iv = np.where(on_track, 10.0, 60.0)
            next_sync = np.where(due, t + iv, next_sync)

        use = np.where(alive, use, 0.0)
        e = np.minimum(e + h - use, BATTERY_J)
        newly_empty = (e <= 0) & np.isnan(empty_day)
        empty_day[newly_empty] = t / 1440
        e = np.maximum(e, 0)

    alive_days = np.where(np.isnan(empty_day), DAYS, empty_day)
    return alive_days, front_min, syncs


def main():
    lux, sun, reading = build_days()
    p = sample(np.random.default_rng(42))
    rows = []
    for s in STRATEGIES:
        days, front, syncs = run(s, lux, sun, reading, p)
        lived = np.minimum(days, DAYS)
        rows.append({
            "strategy": s,
            "battery_days_median": round(float(np.median(lived)), 1),
            "battery_days_worst10%": round(float(np.percentile(lived, 10)), 1),
            "reached_target_%": round(100 * float(np.mean(lived >= TARGET_DAYS)), 1),
            "frontlight_h_per_day": round(float(np.median(front / 60 / np.maximum(lived, 1))), 2),
            "syncs_per_day": round(float(np.median(syncs / np.maximum(lived, 1))), 1),
        })
        print(f"done: {s}")
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(f"\n=== Hybrid e-reader, 6 Wh battery, target {TARGET_DAYS} days per charge, "
          f"{N} devices ===")
    print(df.to_string(index=False))
    df.to_csv(DOCS / "unified_results.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(df["strategy"], df["battery_days_median"], color=["tab:red", "tab:blue", "tab:green"])
    ax.axhline(TARGET_DAYS, color="black", ls="--", lw=1, label=f"target: {TARGET_DAYS} days")
    ax.set_ylabel("Days per battery charge (median)")
    ax.set_title("One shared energy budget vs separate features")
    ax.legend()
    fig.tight_layout()
    fig.savefig(DOCS / "unified.png", dpi=110)
    print(f"\nSaved: {DOCS / 'unified_results.csv'} and {DOCS / 'unified.png'}")


if __name__ == "__main__":
    main()
