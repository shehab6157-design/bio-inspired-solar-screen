"""
The Light-Life Score: rating screens the way people actually use them.

Official energy tests measure a screen's POWER at a few fixed room-light
levels in a lab. They never ask whether the picture can actually be READ
against glare, and they never go outdoors.

The Light-Life Score asks one question:
  how much energy does this screen need for every hour it is truly readable?

It runs each screen through a "light diet": the hours a day it spends in
each kind of light, from a dark bedroom to full sun. At every light level:
  - power: what the screen draws to reach the brightness that light needs
  - readability: does it beat the glare (screen >= 3x the reflected room light)?
Score = energy per READABLE hour (Wh). Lower is better. A screen that saves
power by being unreadable in the sun cannot hide: its readable share drops.

Grades use doubling bands (like an energy label):
  A <= 0.05, B <= 0.1, C <= 0.2, D <= 0.4, E <= 0.8, F <= 1.6, G above (Wh per readable hour,
  for a phone-size 100 cm^2 screen; TVs are graded after dividing by screen area).
A screen readable less than 95% of the time gets a "!" next to its grade.

The light diet is an ASSUMPTION built from typical days; phase 2 would
replace it with real light logged by volunteers' phones.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import polarizer as pz
import chameleon as ch

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
DOCS.mkdir(exist_ok=True)

# hours of screen-on use per day in each kind of light: (hours, lux, place)
DIETS = {
    "handheld": [(1.0, 5, "dark room"), (1.5, 80, "dim home"), (2.0, 400, "office / lit room"),
                 (0.5, 2000, "by a window"), (0.5, 10000, "outdoors, shade"), (0.5, 30000, "outdoors, sun")],
    "tv": [(1.5, 5, "dark room"), (2.0, 50, "evening lamp"), (1.0, 200, "lit room"), (0.5, 1000, "sunny room")],
}
REFERENCE_CM2 = 100
BANDS = [("A", 0.05), ("B", 0.1), ("C", 0.2), ("D", 0.4), ("E", 0.8), ("F", 1.6)]
READABLE_BAR = 95.0

SCREENS = {
    # name: (kind, diet, description)
    "phone today (polarizer OLED)":         ("emissive", "handheld", {"stack": "polarizer (today)"}),
    "phone, polarizer-free":                ("emissive", "handheld", {"stack": "polarizer-free"}),
    "phone, polarizer-free + moth eye":     ("emissive", "handheld", {"stack": "polarizer-free + moth eye"}),
    "phone, full bio-stack (+firefly, moon mode)": ("bio", "handheld", {"stack": "polarizer-free + moth + firefly"}),
    "reflective colour screen + front light": ("reflective", "handheld", {}),
    "TV today (polarizer OLED)":            ("emissive_tv", "tv", {"stack": "polarizer (today)"}),
    "TV, polarizer-free + moth eye":        ("emissive_tv", "tv", {"stack": "polarizer-free + moth eye"}),
    "TV, + firefly":                        ("emissive_tv", "tv", {"stack": "polarizer-free + moth + firefly"}),
}


def grade(wh_per_readable_h, cm2=REFERENCE_CM2):
    x = wh_per_readable_h * REFERENCE_CM2 / cm2
    for letter, limit in BANDS:
        if x <= limit:
            return letter
    return "G"


def score(name, seed=42):
    kind, diet, spec = SCREENS[name]
    rng = np.random.default_rng(seed)
    p = pz.sample(rng)
    device = "tv" if kind == "emissive_tv" else "phone"
    w200 = rng.uniform(*pz.DEVICES[device]["w_at_200"], pz.N)
    pc = ch.sample(np.random.default_rng(3))
    cm2 = 8300 if device == "tv" else REFERENCE_CM2
    energy = np.zeros(pz.N)
    readable_h = np.zeros(pz.N)
    total_h = 0.0
    rows = []
    for hours, lux, place in DIETS[diet]:
        total_h += hours
        reflective = pc["reflective_mw_cm2"] * cm2 / 1000
        if kind == "reflective" or (kind == "bio" and lux >= ch.SWITCH_LUX):
            watts = reflective + (pc["frontlight_w"] * cm2 / 100 if lux < ch.SWITCH_LUX else 0)
            e = watts * hours * 3600
            ok = np.full(pz.N, hours)                  # reflective: lit by the room (or front light)
        else:
            e, bad_min = pz.day(device, [(hours, lux)], spec["stack"], p, w200)
            ok = hours - bad_min / 60
        energy += e
        readable_h += ok
        rows.append((place, float(np.median(e)) / 3600, float(np.median(ok)) / hours * 100))
    wh = energy / 3600
    per_readable = np.median(wh / np.maximum(readable_h, 1e-9))
    readable_pct = float(np.median(100 * readable_h / total_h))
    letter = grade(per_readable, cm2)
    return {
        "screen": name,
        "Wh_per_day": round(float(np.median(wh)), 3),
        "readable_%": round(readable_pct, 1),
        "Wh_per_readable_hour": round(float(per_readable), 4),
        "grade": letter + ("!" if readable_pct < READABLE_BAR else ""),
        "by_place": rows,
    }


def main():
    results = [score(n) for n in SCREENS]
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "by_place"} for r in results])
    pd.set_option("display.width", 200)
    print("=== Light-Life Score: energy per READABLE hour (lower is better) ===")
    print("Handheld light diet: " + ", ".join(f"{h} h {pl}" for h, _, pl in DIETS["handheld"]))
    print("TV light diet:       " + ", ".join(f"{h} h {pl}" for h, _, pl in DIETS["tv"]))
    print()
    print(df.to_string(index=False))
    df.to_csv(DOCS / "light_score.csv", index=False)

    print("\n=== Where each phone loses readability (readable % by place) ===")
    places = [pl for _, _, pl in DIETS["handheld"]]
    detail = pd.DataFrame({r["screen"]: [round(x[2]) for x in r["by_place"]]
                           for r in results if SCREENS[r["screen"]][1] == "handheld"}, index=places).T
    print(detail.to_string())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, diet in zip(axes, ("handheld", "tv")):
        d = df[[SCREENS[s][1] == diet for s in df["screen"]]].iloc[::-1]
        colors = ["tab:red" if g.endswith("!") else "tab:green" for g in d["grade"]]
        bars = ax.barh(d["screen"], d["Wh_per_readable_hour"], color=colors)
        for b, g, rp in zip(bars, d["grade"], d["readable_%"]):
            ax.text(b.get_width(), b.get_y() + b.get_height() / 2, f"  {g}  ({rp:.0f}% readable)", va="center", fontsize=9)
        ax.set_xlabel("Wh per readable hour (lower is better)")
        ax.set_title(f"{'Phones and handhelds' if diet == 'handheld' else 'TVs'}")
        ax.tick_params(axis="y", labelsize=8)
        ax.set_xlim(0, d["Wh_per_readable_hour"].max() * 1.6)
    fig.suptitle("Light-Life Score: energy per hour a screen is actually readable (red = readable under 95%)")
    fig.tight_layout()
    fig.savefig(DOCS / "light_score.png", dpi=110)
    print(f"\nSaved: {DOCS / 'light_score.csv'} and {DOCS / 'light_score.png'}")


if __name__ == "__main__":
    main()
