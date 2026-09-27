"""
The light eye: the screen's own harvesting layers used as its eyes.

The bio-stack already has two light-catching layers:
  - a visible-light cell (bezel / back panel)
  - a near-infrared dye layer (Step 1)
Read together, at no extra cost, they say how MUCH light there is and what
KIND: LED room light has almost no infrared, daylight has a lot (even after
window glass, which blocks some infrared - modern low-e glass blocks most).

What the eye is used for:
  dark / pocket  -> pause refreshes and harvesting circuits
  lamp light     -> no infrared harvesting (there is nothing to harvest)
  daylight       -> infrared harvesting on, moon mode, watch for sun heat

Checks (Monte Carlo, realistic sensor noise and calibration spread):
  1. Can it tell daylight from lamp light, at every light level?
     Compared with a normal one-band light sensor, which sees only brightness.
  2. Can light alone tell a POCKET from a DARK BEDROOM? Phones use a
     proximity sensor for this; the test shows whether light could replace it.

Using solar cells as light sensors is known in research; the two-band reading
of the screen's own layers is this project's version of it.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import spectral as sp

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
DOCS.mkdir(exist_ok=True)

N = 3000
VIS = (sp.WL >= 400) & (sp.WL < 700)
NIR = (sp.WL >= 700) & (sp.WL <= 1200)


def band_watts_per_lux(source):
    """Visible and near-infrared W/m^2 carried by 1 lux of this light."""
    spec = sp.normalise_to_lux(sp.SPECTRA_SHAPE[source], 1.0)
    return spec[VIS].sum() * sp.DL, spec[NIR].sum() * sp.DL


VIS_LED, NIR_LED = band_watts_per_lux("led")
VIS_SUN, NIR_SUN = band_watts_per_lux("sun")

# (name, light type, lux range, share of infrared that gets through window glass)
ENVIRONMENTS = [
    ("in a pocket",             "dark",  (0.0, 1.0),       None),
    ("dark bedroom (reading)",  "lamp",  (1.0, 10.0),      None),
    ("dim lamp",                "lamp",  (20, 100),        None),
    ("office LED",              "lamp",  (300, 600),       None),
    ("window, clear glass",     "day",   (500, 3000),      (0.7, 0.9)),
    ("window, low-e glass",     "day",   (500, 3000),      (0.15, 0.4)),
    ("outdoors, shade",         "day",   (5000, 15000),    (1.0, 1.0)),
    ("outdoors, sun",           "day",   (30000, 90000),   (1.0, 1.0)),
]
CAL_SPREAD = 0.20          # +-20% gain error per reading (angle, dust, calibration)
DARK_NOISE_LUX = 1.5       # noise floor of a cheap cell, in lux-equivalent
DARK_LUX = 2.0             # below this the eye says "dark"
TYPE_MIN_LUX = 20.0        # too dim to judge the colour of light reliably: assume lamp light
DAY_RATIO = 0.10           # infrared/visible above this = daylight (LED ~0.004, sun ~0.87;
                           # low enough to still see daylight through low-e glass)


def readings(env, rng):
    name, kind, (lo, hi), glass = env
    lux = rng.uniform(lo, hi, N)
    source = "led" if kind in ("lamp", "dark") else "sun"
    vis_w = lux * (VIS_LED if source == "led" else VIS_SUN)
    nir_w = lux * (NIR_LED if source == "led" else NIR_SUN)
    if glass:
        nir_w = nir_w * rng.uniform(*glass, N)
    gain_v = 1 + rng.uniform(-CAL_SPREAD, CAL_SPREAD, N)
    gain_n = 1 + rng.uniform(-CAL_SPREAD, CAL_SPREAD, N)
    noise = DARK_NOISE_LUX * VIS_LED
    v = np.maximum(vis_w * gain_v + rng.normal(0, noise, N), 0)
    n = np.maximum(nir_w * gain_n + rng.normal(0, noise, N), 0)
    return lux, v, n


def classify(v, n):
    """Two-band light eye: 'dark', 'lamp' or 'day'."""
    lux_est = v / VIS_LED
    ratio = n / np.maximum(v, 1e-12)
    is_day = (ratio > DAY_RATIO) & (lux_est >= TYPE_MIN_LUX)
    return np.where(lux_est < DARK_LUX, "dark", np.where(is_day, "day", "lamp"))


def classify_one_band(v):
    """A normal one-band sensor sees only brightness: best it can do is guess
    daylight when it is very bright."""
    lux_est = v / VIS_LED
    return np.where(lux_est < DARK_LUX, "dark", np.where(lux_est > 3000, "day", "lamp"))


def main():
    rng = np.random.default_rng(42)
    print(f"Infrared per unit of visible light: LED {NIR_LED / VIS_LED:.3f}, sunlight {NIR_SUN / VIS_SUN:.2f}\n")
    rows, pocket, bedroom = [], None, None
    for env in ENVIRONMENTS:
        lux, v, n = readings(env, rng)
        truth = env[1]
        two = classify(v, n)
        one = classify_one_band(v)
        rows.append({"environment": env[0], "truth": truth,
                     "light_eye_correct_%": round(100 * float(np.mean(two == truth)), 1),
                     "one_band_sensor_correct_%": round(100 * float(np.mean(one == truth)), 1)})
        if env[0] == "in a pocket":
            pocket = v / VIS_LED
        if env[0].startswith("dark bedroom"):
            bedroom = v / VIS_LED
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("=== 1. What kind of light is this? (share of readings classified correctly) ===")
    print(df.to_string(index=False))
    df.to_csv(DOCS / "light_eye_results.csv", index=False)

    print("\n=== 2. Pocket or dark bedroom? (light alone) ===")
    best = None
    for thr in np.linspace(0.2, 3.0, 29):
        caught = np.mean(pocket < thr)            # pockets correctly detected
        wrong = np.mean(bedroom < thr)            # bedrooms mistaken for a pocket
        if best is None or caught - wrong > best[1] - best[2]:
            best = (thr, caught, wrong)
    thr, caught, wrong = best
    print(f"Best light-only rule (darker than {thr:.1f} lux = pocket): catches {100 * caught:.0f}% of pockets, "
          f"but mistakes {100 * wrong:.0f}% of dark-bedroom readings for a pocket.")
    print("Mistaking a bedroom for a pocket would switch the screen off while someone reads in the dark,")
    print("so the eye should team up with the phone's proximity sensor for this one decision.")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    x = np.arange(len(df))
    axes[0].bar(x - 0.2, df["one_band_sensor_correct_%"], 0.4, label="normal one-band light sensor", color="tab:gray")
    axes[0].bar(x + 0.2, df["light_eye_correct_%"], 0.4, label="two-band light eye", color="tab:green")
    axes[0].set_xticks(x, df["environment"], rotation=30, ha="right", fontsize=8)
    axes[0].set_ylabel("Correctly identified (%)")
    axes[0].set_title("What kind of light is this?")
    axes[0].legend(fontsize=8)
    bins = np.linspace(0, 12, 49)
    axes[1].hist(pocket, bins, alpha=0.6, label="in a pocket", color="tab:purple")
    axes[1].hist(bedroom, bins, alpha=0.6, label="dark bedroom", color="tab:orange")
    axes[1].axvline(thr, color="black", ls="--", lw=1, label="best light-only threshold")
    axes[1].set_xlabel("Light the eye measures (lux)")
    axes[1].set_title("Pocket vs dark bedroom: the readings overlap")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(DOCS / "light_eye.png", dpi=110)
    print(f"\nSaved: {DOCS / 'light_eye_results.csv'} and {DOCS / 'light_eye.png'}")


if __name__ == "__main__":
    main()
