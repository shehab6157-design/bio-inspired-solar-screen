"""
Step 1 (advanced): Spectral light model.

Before: the dye (leaf antenna) layer took a flat 5-15% of ALL light, so on
glowing screens it always dimmed the picture and lost energy.

Now: light is handled wavelength by wavelength (300-1200 nm).
  - Sun, LED room light and the OLED screen each have their own spectrum
  - The eye's sensitivity curve (CIE photopic V(lambda)) decides what we see
  - The dye layer absorbs only a chosen colour band (centre + width)

Question: is there a band that harvests light the eye barely uses (UV or
near-infrared), so the screen does not dim? The script searches every band
for each device and compares it with the old flat dye.

Nature idea: plants and bacteria tune their antenna pigments to the light
they live in (purple bacteria harvest near-infrared under green algae).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import stack_config as sc

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
DOCS.mkdir(exist_ok=True)

WL = np.arange(300, 1201, 5.0)          # wavelength grid, nm
DL = 5.0                                  # nm step
LM_PER_W = 683.0                          # lumens per watt at 555 nm

# chain efficiency: absorbed light -> re-emitted -> guided to edge -> electricity
DYE_CHAIN = (0.15, 0.35)
PEAK_ABSORPTION = 0.8                     # max fraction absorbed at the band centre
RECORD_EFF = 3.0                          # % - best transparent dye-layer result so far


def gauss(center, fwhm):
    s = fwhm / 2.3548
    return np.exp(-0.5 * ((WL - center) / s) ** 2)


def eye_sensitivity():
    """CIE photopic luminosity curve (Gaussian approximation)."""
    um = WL / 1000
    return 1.019 * np.exp(-285.4 * (um - 0.559) ** 2)


def blackbody(temp_k):
    h, c, k = 6.626e-34, 2.998e8, 1.381e-23
    lam = WL * 1e-9
    return (2 * h * c**2 / lam**5) / (np.exp(h * c / (lam * k * temp_k)) - 1)


V = eye_sensitivity()
SPECTRA_SHAPE = {
    "sun": blackbody(5778),
    "led": gauss(450, 22) * 0.9 + gauss(565, 120),
    "oled": gauss(460, 30) + gauss(530, 40) + gauss(620, 35),
}


def normalise_to_lux(shape, lux=1.0):
    lux_of_shape = LM_PER_W * np.sum(shape * V) * DL
    return shape * lux / lux_of_shape


def watts_per_lux(source):
    return np.sum(normalise_to_lux(SPECTRA_SHAPE[source])) * DL


def dye_absorption(center, width):
    return PEAK_ABSORPTION * gauss(center, width)


def flat_dye(loss):
    return np.full_like(WL, loss)


def perceived_transmission(absorb, light):
    return np.sum(light * V * (1 - absorb)) / np.sum(light * V)


def colour_tint(absorb):
    t = [1 - np.interp(w, WL, absorb) for w in (460, 530, 620)]
    return max(t) - min(t)


def daily_harvest_j(tier, absorb, chain):
    t = sc.TIERS[tier]
    area = t["front_cm2"] / 1e4
    total = 0.0
    for lux, src, hours in t["front_light"]:
        spec = normalise_to_lux(SPECTRA_SHAPE[src], lux)
        total += np.sum(spec * absorb) * DL * area * hours * 3600
    return total * chain


def sunlight_efficiency(absorb, chain):
    sun = SPECTRA_SHAPE["sun"]
    return np.sum(sun * absorb) / np.sum(sun) * chain


def evaluate(tier, absorb, chain):
    t = sc.TIERS[tier]
    load = np.mean(t["load_j_day"])
    display_j = load * np.mean(t["display_share"])
    harvest = daily_harvest_j(tier, absorb, chain)
    if t["display"] == "emissive":
        tr = perceived_transmission(absorb, SPECTRA_SHAPE["oled"])
        penalty = display_j * (1 / tr - 1)
        readability = 100 * tr
    else:
        tr = perceived_transmission(absorb, SPECTRA_SHAPE["led"])
        penalty = 0.0
        readability = 100 * tr ** 2
    return {
        "harvest_J_day": harvest,
        "penalty_J_day": penalty,
        "net_%": 100 * (harvest - penalty) / load,
        "readability_%": readability,
        "tint": colour_tint(absorb),
    }


def search_best_band(tier, chain, min_read=sc.MIN_READABILITY, max_tint=0.05):
    centers = np.arange(320, 1150, 10)
    widths = np.arange(20, 301, 20)
    grid = np.full((len(widths), len(centers)), np.nan)
    best = None
    for i, w in enumerate(widths):
        for j, c in enumerate(centers):
            r = evaluate(tier, dye_absorption(c, w), chain)
            grid[i, j] = r["net_%"]
            ok = r["readability_%"] >= min_read and r["tint"] <= max_tint
            if ok and (best is None or r["net_%"] > best[2]["net_%"]):
                best = (c, w, r)
    return centers, widths, grid, best


def main():
    mid_chain = np.mean(DYE_CHAIN)

    print("=== Check 1: light physics vs the constants the older models assumed ===")
    for src, assumed in (("led", sc.LED_K), ("sun", sc.SUN_K)):
        derived = watts_per_lux(src)
        print(f"{src:4s}: derived {1/derived:6.0f} lux per W/m^2   assumed {1/assumed:6.0f}")

    rows = []
    for tier in ("e-reader", "phone", "tv"):
        centers, widths, grid, best = search_best_band(tier, mid_chain)
        flat = evaluate(tier, flat_dye(0.10), mid_chain)
        rows.append({"tier": tier, "dye": "flat (old model)", "band": "all colours",
                     **{k: round(v, 3) for k, v in flat.items()}})
        if best:
            c, w, r = best
            lo = evaluate(tier, dye_absorption(c, w), DYE_CHAIN[0])["net_%"]
            hi = evaluate(tier, dye_absorption(c, w), DYE_CHAIN[1])["net_%"]
            eff = 100 * sunlight_efficiency(dye_absorption(c, w), mid_chain)
            rows.append({"tier": tier, "dye": "best spectral band",
                         "band": f"{c:.0f} nm, width {w:.0f} nm",
                         **{k: round(v, 3) for k, v in r.items()},
                         "net_%_range": f"{lo:.3f} to {hi:.3f}",
                         "sun_eff_%": round(eff, 2)})

        fig, ax = plt.subplots(figsize=(9, 4))
        lim = np.nanmax(np.abs(grid))
        im = ax.imshow(grid, aspect="auto", origin="lower", cmap="RdYlGn", vmin=-lim, vmax=lim,
                       extent=[centers[0], centers[-1], widths[0], widths[-1]])
        if best:
            ax.plot(best[0], best[1], "k*", markersize=14, label="best allowed band")
            ax.legend(loc="upper right")
        ax.axvspan(400, 700, color="white", alpha=0.12)
        ax.set_xlabel("Dye absorption band centre (nm)   [400-700 nm = visible]")
        ax.set_ylabel("Band width (nm)")
        ax.set_title(f"{tier}: net % of daily energy from the dye layer alone")
        fig.colorbar(im, ax=ax, label="net %")
        fig.tight_layout()
        fig.savefig(DOCS / f"spectral_heatmap_{tier}.png", dpi=110)
        plt.close(fig)

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print("\n=== Check 2: flat dye vs best invisible band (readability >= "
          f"{sc.MIN_READABILITY}%, colour tint <= 5%) ===")
    print(df.to_string(index=False))
    df.to_csv(DOCS / "spectral_results.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 4))
    for name, shape in SPECTRA_SHAPE.items():
        ax.plot(WL, shape / shape.max(), label=name)
    ax.plot(WL, V, "k--", label="human eye")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Relative intensity")
    ax.set_title("Light sources vs what the eye sees")
    ax.legend()
    fig.tight_layout()
    fig.savefig(DOCS / "spectra.png", dpi=110)
    print(f"\nReality check: best transparent dye layers measured so far reach about "
          f"{RECORD_EFF}% of sunlight. Rows with sun_eff_% above that are optimistic.")
    print(f"\nSaved: spectral_results.csv, spectra.png and spectral_heatmap_*.png in {DOCS}")


if __name__ == "__main__":
    main()
