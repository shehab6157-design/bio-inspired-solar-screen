"""
Milestone 2: 7-day, minute-by-minute energy simulation for a Tier 1
(e-paper / sensor-class) device.

Adds over simulate.py:
  - Real daily light profiles (dark nights, work hours, dim homes)
  - Duty-cycled load: tiny sleep draw + short bursts per screen refresh
  - Energy storage (supercapacitor-scale buffer)
  - Two update policies:
      fixed  : refresh every 10 minutes no matter what
      torpor : adapts refresh rate to stored energy, like a hummingbird
               entering torpor at night to survive on its reserves
  - Corrected indoor light conversion (LED light, not daylight)

All device numbers are ASSUMPTION RANGES, to be replaced by bench
measurements in the prototype milestone.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
DOCS.mkdir(exist_ok=True)

LED_K = 1 / 330   # W/m^2 per lux, white LED indoor light
SUN_K = 1 / 120   # W/m^2 per lux, daylight

PROFILES = {
    "office_desk": [(0, 8, 0, "led"), (8, 18, 400, "led"), (18, 24, 0, "led")],
    "home_dim":    [(0, 7, 0, "led"), (7, 9, 150, "led"), (9, 17, 50, "led"),
                    (17, 23, 200, "led"), (23, 24, 0, "led")],
    "near_window": [(0, 7, 0, "sun"), (7, 17, 1500, "sun"),
                    (17, 22, 200, "led"), (22, 24, 0, "led")],
}

DEVICE = {
    "area_cm2": 30,
    "eff_led": (0.15, 0.25),
    "eff_sun": (0.08, 0.15),
    "conv_eff": (0.60, 0.85),
    "sleep_uw": (5, 20),
    "refresh_mj": (20, 80),
    "radio_mj": (0.5, 2.0),
    "storage_j": 10.0,
    "start_soc": 0.5,
    "reserve_soc": 0.05,
}

DAYS = 7
TRIALS = 300
POLICIES = ["fixed", "torpor"]


def sample_params(rng, n):
    u = lambda key: rng.uniform(*DEVICE[key], n)
    return {
        "area_m2": DEVICE["area_cm2"] / 1e4,
        "eff_led": u("eff_led"),
        "eff_sun": u("eff_sun"),
        "conv": u("conv_eff"),
        "sleep_w": u("sleep_uw") * 1e-6,
        "cost_j": (u("refresh_mj") + u("radio_mj")) * 1e-3,
    }


def light_per_minute(profile):
    lux = np.zeros(1440)
    sun = np.zeros(1440, dtype=bool)
    for h0, h1, level, src in profile:
        lux[h0 * 60:h1 * 60] = level
        sun[h0 * 60:h1 * 60] = (src == "sun")
    return lux, sun


def interval_minutes(policy, soc):
    if policy == "fixed":
        return np.full(soc.shape, 10.0)
    return np.select([soc > 0.7, soc > 0.4, soc > 0.2], [5.0, 10.0, 30.0], default=120.0)


def simulate(profile, policy, p, n):
    cap = DEVICE["storage_j"]
    reserve = cap * DEVICE["reserve_soc"]
    lux, sun = light_per_minute(profile)
    minutes = DAYS * 1440

    e = np.full(n, cap * DEVICE["start_soc"])
    next_update = np.zeros(n)
    updates = np.zeros(n)
    missed = np.zeros(n)
    dead_min = np.zeros(n)
    wasted_j = np.zeros(n)
    harvested_j = np.zeros(n)
    soc_trace = np.zeros(minutes)

    for t in range(minutes):
        m = t % 1440
        k_eff = SUN_K * p["eff_sun"] if sun[m] else LED_K * p["eff_led"]
        harvest = lux[m] * k_eff * p["area_m2"] * 60 * p["conv"]
        harvested_j += harvest
        e += harvest - p["sleep_w"] * 60

        due = t >= next_update
        can = due & (e - p["cost_j"] >= reserve)
        e = np.where(can, e - p["cost_j"], e)
        updates += can
        missed += due & ~can
        next_update = np.where(due, t + interval_minutes(policy, e / cap), next_update)

        dead_min += e <= 0
        e = np.maximum(e, 0)
        wasted_j += np.maximum(e - cap, 0)
        e = np.minimum(e, cap)
        soc_trace[t] = np.median(e / cap)

    return {
        "updates": updates, "missed": missed, "dead_min": dead_min,
        "wasted_j": wasted_j, "harvested_j": harvested_j,
        "end_soc": e / cap, "soc_trace": soc_trace,
    }


def main():
    rng = np.random.default_rng(42)
    p = sample_params(rng, TRIALS)

    rows, traces = [], {}
    for name, profile in PROFILES.items():
        for policy in POLICIES:
            r = simulate(profile, policy, p, TRIALS)
            traces[(name, policy)] = r["soc_trace"]
            ok = (r["missed"] == 0) & (r["dead_min"] == 0)
            rows.append({
                "profile": name,
                "policy": policy,
                "pass_rate_%": round(100 * ok.mean(), 1),
                "harvest_J_per_day": round(np.median(r["harvested_j"]) / DAYS, 2),
                "updates_per_day": round(np.median(r["updates"]) / DAYS, 1),
                "missed_per_day": round(np.median(r["missed"]) / DAYS, 1),
                "dead_hours": round(np.median(r["dead_min"]) / 60, 1),
                "end_soc_%": round(100 * np.median(r["end_soc"]), 1),
            })

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    df.to_csv(DOCS / "day_sim_summary.csv", index=False)

    fig, axes = plt.subplots(len(PROFILES), 1, figsize=(10, 8), sharex=True)
    hours = np.arange(DAYS * 1440) / 60
    for ax, name in zip(axes, PROFILES):
        for policy in POLICIES:
            ax.plot(hours, 100 * traces[(name, policy)], label=policy)
        ax.set_title(name)
        ax.set_ylabel("Stored energy %")
        ax.set_ylim(0, 105)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("Hours")
    fig.tight_layout()
    fig.savefig(DOCS / "soc_traces.png", dpi=120)
    print(f"\nSaved: {DOCS / 'day_sim_summary.csv'} and {DOCS / 'soc_traces.png'}")


if __name__ == "__main__":
    main()
