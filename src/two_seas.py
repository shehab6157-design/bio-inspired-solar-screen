"""
Step 8: The "Two Seas" screen - keeping a phone cool in the sun.

A phone held in summer sun receives several watts of sunlight on its face,
more than its own electronics use. Phones react by dimming the screen or
shutting down, exactly when you need brightness most.

Inspiration: two seas that meet with a barrier they do not cross (Quran
55:19-20) - one surface where different parts of the light each keep to
their own job:
  visible light      -> passes through to your eyes (picture untouched)
  near-infrared      -> reflected away (heat mirror) or harvested (dye layer)
  mid-infrared       -> radiated to the cold sky (sky-cooling emitter)

Front stacks compared, 60 minutes of outdoor use in summer sun:
  today            : glass + polarizer OLED, as phones are now
  heat mirror      : + a clear layer that reflects near-infrared
  NIR harvest      : + a dye layer that absorbs near-infrared for power
  sky emitter      : + a selective emitter tuned to the 8-13 um sky window
  two seas         : heat mirror + the efficient display from Step 7
                     (polarizer-free + moth eye: less glare, less power)
  two seas + chameleon : as above, but outdoors the screen switches to its
                     reflective mode (Step 6a): like the moon, it shows the
                     picture with the sun's own light, so it reflects visible
                     sunlight instead of absorbing it, and needs almost no power

Brightness: the screen must beat the glare (same rule as Step 7). Phones
protect themselves when hot: above THROTTLE_C brightness is capped, above
SHUTDOWN_C the screen turns off. Minutes where the screen can't reach the
brightness needed are counted as unreadable.

All values are ASSUMPTION RANGES; location enters only as sunlight strength
and air temperature.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import polarizer as pz

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
DOCS.mkdir(exist_ok=True)

N = 2000
MINUTES = 60
DT = 10.0                          # seconds per step
SIGMA = 5.67e-8
UV, VIS, NIR = 0.05, 0.43, 0.52    # share of sunlight energy (approximate, sea level)
THROTTLE_C, SHUTDOWN_C = 42.0, 47.0
MAX_NITS, THROTTLED_NITS = 1200.0, 600.0
GLARE_LUX = 30000                  # light reflecting toward the eye (screen angled away from the sun)

PARAMS = {
    "sun_w_m2": (600, 900),        # sunlight falling on the phone's face
    "air_c": (30, 38),             # summer air temperature
    "sky_drop_k": (10, 20),        # clear sky is this much colder than the air (broadband)
    "win_sky_drop_k": (25, 45),    # ...and colder still inside the 8-13 um window
    "h_w_m2k": (6, 14),            # convection + hand contact
    "heat_cap_j_k": (150, 250),    # thermal mass of the phone
    "front_m2": (0.009, 0.011),
    "back_m2": (0.010, 0.012),
    "chip_w": (1.0, 2.0),          # processor, radio etc. while in use
    "a_vis": (0.85, 0.95),         # front absorbs this much visible light (dark screen)
    "a_nir": (0.75, 0.90),         # ...and this much near-infrared
    "glass_eps": (0.84, 0.90),     # cover glass already radiates heat well
    "mirror": (0.6, 0.9),          # share of near-infrared a heat mirror reflects
    "harvest": (0.6, 0.9),         # share of near-infrared a dye layer absorbs
    "harvest_eff": (0.03, 0.08),   # of that, share turned into electricity (keeps the whole
                                   # layer near the ~3% sunlight record for clear dye layers)
    "heat_kept": (0.4, 0.8),       # of the rest, share that stays in the phone as heat
    "refl_a_vis": (0.3, 0.6),      # a reflective (paper-like) screen absorbs far less visible light
    "refl_w": (0.05, 0.17),        # reflective colour screen power, 100 cm^2 (0.5-1.7 mW/cm^2)
}
STACKS = ["today", "heat mirror", "NIR harvest", "sky emitter", "two seas", "two seas + chameleon"]


def sample(rng):
    return {k: rng.uniform(*v, N) for k, v in PARAMS.items()}


def display_watts(stack, nits, opt):
    """Screen power at a brightness, using Step 7's optics for 'two seas'."""
    t_ref, _ = pz.stack_optics("polarizer (today)", opt)
    name = "polarizer-free + moth eye" if stack == "two seas" else "polarizer (today)"
    t_out, _ = pz.stack_optics(name, opt)
    return 0.275 * (nits / 200) * (t_ref / t_out)


def needed_nits(stack, opt):
    name = "polarizer-free + moth eye" if stack == "two seas" else "polarizer (today)"
    _, refl = pz.stack_optics(name, opt)
    return np.maximum(pz.comfort_nits(GLARE_LUX), (pz.CONTRAST_TARGET - 1) * refl * GLARE_LUX / np.pi)


def run(stack, p, opt):
    k = lambda c: c + 273.15
    t_air, t_sky = k(p["air_c"]), k(p["air_c"] - p["sky_drop_k"])
    t_win = k(p["air_c"] - p["win_sky_drop_k"])
    a_nir = p["a_nir"].copy()
    harvested_w = np.zeros(N)
    if stack in ("heat mirror", "two seas"):
        a_nir = a_nir * (1 - p["mirror"])
    elif stack == "NIR harvest":
        absorbed = p["harvest"]
        harvested_w = p["sun_w_m2"] * p["front_m2"] * NIR * absorbed * p["harvest_eff"]
        a_nir = a_nir * (1 - absorbed) + absorbed * (1 - p["harvest_eff"]) * p["heat_kept"]
    moon = stack == "two seas + chameleon"
    if moon:
        a_nir = p["a_nir"] * (1 - p["mirror"])
    alpha = UV * 0.95 + VIS * (p["refl_a_vis"] if moon else p["a_vis"]) + NIR * a_nir
    need = np.zeros(N) if moon else needed_nits(stack, opt)   # reflective: sunlight lights the picture

    temp = np.full(N, k(28.0))                      # phone comes out of a cool building
    unreadable = np.zeros(N); first_throttle = np.full(N, np.nan); off = np.zeros(N)
    peak = temp.copy()
    for step in range(int(MINUTES * 60 / DT)):
        c = temp - 273.15
        cap = np.where(c >= SHUTDOWN_C, 0.0, np.where(c >= THROTTLE_C, THROTTLED_NITS, MAX_NITS))
        shown = np.minimum(need, cap)
        unreadable += ((shown < need) | (cap == 0)) * DT / 60
        off += (cap == 0) * DT / 60
        newly = np.isnan(first_throttle) & (c >= THROTTLE_C)
        first_throttle[newly] = step * DT / 60

        screen_w = np.where(cap > 0, p["refl_w"], 0.0) if moon else display_watts(stack, shown, opt)
        heat_in = p["sun_w_m2"] * p["front_m2"] * alpha + screen_w + p["chip_w"] * (cap > 0)
        conv = p["h_w_m2k"] * (p["front_m2"] + p["back_m2"]) * (temp - t_air)
        back = 0.8 * SIGMA * p["back_m2"] * (temp**4 - t_air**4)
        if stack == "sky emitter":
            win = 0.3                                   # share of a warm body's heat radiation in 8-13 um
            front = SIGMA * p["front_m2"] * (0.95 * win * (temp**4 - t_win**4)
                                              + 0.10 * (1 - win) * (temp**4 - t_air**4))
        else:
            front = p["glass_eps"] * SIGMA * p["front_m2"] * (temp**4 - t_sky**4)
        temp = temp + (heat_in - conv - back - front) * DT / p["heat_cap_j_k"]
        peak = np.maximum(peak, temp)

    return {
        "peak_c": peak - 273.15,
        "min_to_throttle": np.where(np.isnan(first_throttle), MINUTES, first_throttle),
        "unreadable_min": unreadable,
        "screen_off_min": off,
        "harvested_J": harvested_w * MINUTES * 60,
        "sun_heat_W": p["sun_w_m2"] * p["front_m2"] * alpha,
    }


def main():
    rng = np.random.default_rng(42)
    p = sample(rng)
    opt = pz.sample(np.random.default_rng(7))
    rows = []
    for stack in STACKS:
        r = run(stack, p, opt)
        rows.append({"stack": stack, **{k: round(float(np.median(v)), 1) for k, v in r.items()},
                     "never_throttled_%": round(100 * float(np.mean(r["min_to_throttle"] >= MINUTES)), 1)})
    df = pd.DataFrame(rows)
    df.to_csv(DOCS / "two_seas_results.csv", index=False)
    pd.set_option("display.width", 220)
    print(f"=== {MINUTES} minutes of phone use in summer sun (median phone) ===")
    print(df.to_string(index=False))
    today = df.set_index("stack").loc["today"]
    print(f"\nFor comparison: sunlight heating the phone's face today is about {today['sun_heat_W']:.1f} W, "
          f"while the chip uses {np.median(p['chip_w']):.1f} W.")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    colors = ["tab:gray", "tab:orange", "tab:purple", "tab:blue", "tab:green", "darkgreen"]
    for ax, col, title in zip(axes, ["peak_c", "min_to_throttle", "unreadable_min"],
                              ["Peak temperature (C)", "Minutes until the phone dims itself",
                               "Unreadable minutes in the hour"]):
        ax.bar(df["stack"], df[col], color=colors)
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", labelsize=8, rotation=20)
    axes[0].axhline(THROTTLE_C, color="red", ls="--", lw=1)
    fig.suptitle("Two Seas screen: keeping a phone readable in summer sun")
    fig.tight_layout()
    fig.savefig(DOCS / "two_seas.png", dpi=110)
    print(f"\nSaved: {DOCS / 'two_seas_results.csv'} and {DOCS / 'two_seas.png'}")


if __name__ == "__main__":
    main()
