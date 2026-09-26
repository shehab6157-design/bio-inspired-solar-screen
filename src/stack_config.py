"""
Milestone 3: Layer-stack configurator.

Question: which bio-inspired layers actually pay for themselves,
and on which device type?

Layers (all optional, every combination is tested):
  moth_eye   : anti-reflection texture on the cover glass (moth eye)
  dye        : transparent dye layer sending light to edge cells (photosynthetic antenna)
  firefly    : light-extraction texture for emissive (OLED) displays (firefly lantern)
  back_panel : back-side cell with a micro-lens array (fly compound eye)

For each device tier and stack it reports, per day:
  - energy harvested (front dye layer + back panel)
  - display energy saved or lost (extraction gain vs brightness compensation)
  - % of the device's daily energy covered
  - front transmission / readability (display quality cost)

All numbers are ASSUMPTION RANGES from the research pass, sampled with
Monte Carlo. Replace them with bench measurements later.
"""
from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
DOCS.mkdir(exist_ok=True)

LED_K = 1 / 330            # W/m^2 per lux, LED indoor light
SUN_K = 1 / 120            # W/m^2 per lux, daylight
GLASS_REFLECTION = 0.044   # plain cover glass front reflection
N = 2000                   # Monte Carlo samples
MIN_READABILITY = 85.0     # % - stacks below this are rejected (your call, change it)

LAYERS = ["moth_eye", "dye", "firefly", "back_panel"]

PARAMS = {
    "moth_reflection": (0.002, 0.01),
    "dye_eff": (0.01, 0.03),
    "dye_loss": (0.05, 0.15),
    "firefly_gain": (0.15, 0.61),
    "back_cell_eff": (0.15, 0.25),
    "lens_gain": (1.10, 1.30),
    "conv_eff": (0.60, 0.85),
}

TIERS = {
    "e-reader": {
        "display": "reflective",
        "front_cm2": 90, "back_cm2": 100,
        "load_j_day": (8, 25), "display_share": (0.4, 0.6),
        "front_light": [(300, "led", 4), (2000, "sun", 1)],
        "back_light": [(300, "led", 1)],
    },
    "phone": {
        "display": "emissive",
        "front_cm2": 100, "back_cm2": 110,
        "load_j_day": (40000, 60000), "display_share": (0.3, 0.5),
        "front_light": [(400, "led", 3), (10000, "sun", 0.5)],
        "back_light": [(400, "led", 2)],
    },
    "tv": {
        "display": "emissive",
        "front_cm2": 8300, "back_cm2": 8300,
        "load_j_day": (1.2e6, 2.4e6), "display_share": (0.8, 0.9),
        "front_light": [(200, "led", 10)],
        "back_light": [],
    },
}


def light_energy_j_per_m2(exposure):
    total = 0.0
    for lux, src, hours in exposure:
        k = SUN_K if src == "sun" else LED_K
        total += lux * k * hours * 3600
    return total


def evaluate(tier, stack, rng):
    u = lambda key: rng.uniform(*PARAMS[key], N)
    t = TIERS[tier]
    moth, dye, firefly, back = (layer in stack for layer in LAYERS)

    conv = u("conv_eff")
    load = rng.uniform(*t["load_j_day"], N)
    display_j = load * rng.uniform(*t["display_share"], N)

    t_moth = (1 - u("moth_reflection")) / (1 - GLASS_REFLECTION) if moth else np.ones(N)
    t_dye = (1 - u("dye_loss")) if dye else np.ones(N)
    t_front = t_moth * t_dye

    front_light = light_energy_j_per_m2(t["front_light"]) * t["front_cm2"] / 1e4
    back_light = light_energy_j_per_m2(t["back_light"]) * t["back_cm2"] / 1e4
    harvest = np.zeros(N)
    if dye:
        harvest += front_light * t_moth * u("dye_eff") * conv
    if back:
        harvest += back_light * u("back_cell_eff") * u("lens_gain") * conv

    if t["display"] == "emissive":
        gain = 1 + u("firefly_gain") if firefly else np.ones(N)
        new_display_j = display_j / gain / t_front
        saved = display_j - new_display_j
        readability = 100 * np.minimum(t_front, 1.05)
    else:
        saved = np.zeros(N)
        readability = 100 * t_front ** 2

    coverage = 100 * (harvest + saved) / load
    return {
        "tier": tier,
        "stack": "+".join(stack) if stack else "(none)",
        "layers": len(stack),
        "harvest_J_day": np.median(harvest),
        "display_saved_J_day": np.median(saved),
        "coverage_%_median": np.median(coverage),
        "coverage_%_p10": np.percentile(coverage, 10),
        "coverage_%_p90": np.percentile(coverage, 90),
        "readability_%": np.median(readability),
    }


def main():
    rows = []
    for tier, cfg in TIERS.items():
        for mask in product([0, 1], repeat=len(LAYERS)):
            stack = [l for l, on in zip(LAYERS, mask) if on]
            if cfg["display"] == "reflective" and "firefly" in stack:
                continue
            rng = np.random.default_rng(42)
            rows.append(evaluate(tier, stack, rng))

    df = pd.DataFrame(rows)
    df["passes_readability"] = df["readability_%"] >= MIN_READABILITY
    num_cols = df.select_dtypes("number").columns
    df[num_cols] = df[num_cols].round(3)
    df.to_csv(DOCS / "stack_results.csv", index=False)

    pd.set_option("display.width", 220)
    print("=== Recommended stack per device (best median coverage, readability >= "
          f"{MIN_READABILITY}%) ===")
    rec = (df[df["passes_readability"]]
           .sort_values("coverage_%_median", ascending=False)
           .groupby("tier", sort=False).head(1))
    print(rec[["tier", "stack", "coverage_%_median", "coverage_%_p10",
               "coverage_%_p90", "readability_%"]].to_string(index=False))

    print("\n=== Single-layer effect per device (what each layer adds alone) ===")
    single = df[df["layers"] == 1]
    print(single[["tier", "stack", "harvest_J_day", "display_saved_J_day",
                  "coverage_%_median", "readability_%"]].to_string(index=False))

    fig, axes = plt.subplots(len(TIERS), 1, figsize=(10, 12))
    for ax, tier in zip(axes, TIERS):
        d = df[df["tier"] == tier].sort_values("coverage_%_median")
        colors = ["tab:green" if ok else "tab:red" for ok in d["passes_readability"]]
        err = [d["coverage_%_median"] - d["coverage_%_p10"],
               d["coverage_%_p90"] - d["coverage_%_median"]]
        ax.barh(d["stack"], d["coverage_%_median"], xerr=err, color=colors, capsize=3)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(f"{tier}: % of daily energy covered (red = fails readability)")
        ax.tick_params(axis="y", labelsize=8)
    axes[-1].set_xlabel("% of daily energy (median, bars = p10-p90)")
    fig.tight_layout()
    fig.savefig(DOCS / "stack_comparison.png", dpi=120)
    print(f"\nSaved: {DOCS / 'stack_results.csv'} and {DOCS / 'stack_comparison.png'}")


if __name__ == "__main__":
    main()
