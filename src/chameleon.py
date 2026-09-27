"""
Milestone 6a: Chameleon mode (hybrid reflective + emissive display).

Nature idea: chameleons and octopuses change how their skin handles
light depending on surroundings. Here the screen does the same:
  - in bright light it runs as a REFLECTIVE display (the room/sun lights it,
    near-zero power, like e-paper or butterfly structural colour)
  - in dim light it switches to EMISSIVE (OLED), which looks best in the dark

Compared modes for a phone-sized screen over a day of real use:
  emissive   : normal OLED phone, brightness rises with ambient light
  reflective : reflective only, needs a front light in dim rooms (lower quality)
  chameleon  : reflective when ambient >= threshold, emissive otherwise

All numbers are ASSUMPTION RANGES; reflective video-rate power comes from
the 2025 Nature e-paper result (0.5-1.7 mW/cm^2).
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

N = 2000
SCREEN_CM2 = 100
SWITCH_LUX = 300

PARAMS = {
    "emissive_base_w": (0.20, 0.35),
    "emissive_sun_extra_w": (0.80, 1.40),
    "reflective_mw_cm2": (0.5, 1.7),
    "frontlight_w": (0.05, 0.15),
}

USERS = {
    "office_worker": [(0.5, 20000), (3.0, 400), (1.5, 150), (0.5, 5)],
    "outdoor_worker": [(2.5, 30000), (1.0, 300), (1.0, 100)],
    "night_owl": [(0.5, 400), (2.0, 80), (2.0, 5)],
}
MODES = ["emissive", "reflective", "chameleon"]


def sample(rng):
    return {k: rng.uniform(*v, N) for k, v in PARAMS.items()}


def mode_power(mode, lux, p):
    """Returns (watts, reduced_quality) for one mode at one light level."""
    emissive = p["emissive_base_w"] + p["emissive_sun_extra_w"] * min(1.0, lux / 20000)
    reflective = p["reflective_mw_cm2"] * SCREEN_CM2 / 1000
    dim = lux < SWITCH_LUX
    if mode == "emissive":
        return emissive, False
    if mode == "reflective":
        return (reflective + p["frontlight_w"] if dim else reflective), dim
    return (emissive, False) if dim else (reflective, False)


def evaluate_user(usage, p):
    rows = []
    for mode in MODES:
        energy = np.zeros(N)
        low_quality_h = 0.0
        for hours, lux in usage:
            w, lowq = mode_power(mode, lux, p)
            energy += w * hours * 3600
            low_quality_h += hours if lowq else 0
        rows.append((mode, energy, low_quality_h))
    return rows


def main():
    p = sample(np.random.default_rng(42))
    out = []
    for user, usage in USERS.items():
        results = evaluate_user(usage, p)
        base = results[0][1]
        for mode, energy, lowq in results:
            out.append({
                "user": user,
                "mode": mode,
                "display_J_day": round(float(np.median(energy)), 0),
                "saved_vs_emissive_%": round(float(np.median(100 * (1 - energy / base))), 1),
                "reduced_quality_h": lowq,
            })
    df = pd.DataFrame(out)
    print(f"Chameleon switch point: {SWITCH_LUX} lux\n")
    print(df.to_string(index=False))
    df.to_csv(DOCS / "chameleon_results.csv", index=False)

    users = list(USERS)
    x = np.arange(len(users))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, mode in enumerate(MODES):
        vals = [df[(df.user == u) & (df["mode"] == mode)]["display_J_day"].iloc[0] for u in users]
        ax.bar(x + (i - 1) * 0.27, vals, 0.27, label=mode)
    ax.set_xticks(x, users)
    ax.set_ylabel("Display energy per day (J)")
    ax.set_title(f"Chameleon mode vs emissive vs reflective (switch at {SWITCH_LUX} lux)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(DOCS / "chameleon.png", dpi=120)
    print(f"\nSaved: {DOCS / 'chameleon_results.csv'} and {DOCS / 'chameleon.png'}")


if __name__ == "__main__":
    main()
