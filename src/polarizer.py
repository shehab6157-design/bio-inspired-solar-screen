"""
Step 7: Moth eye + polarizer-free OLED, for phones and TVs.

Every OLED phone and TV has a circular polarizer: it stops room light from
bouncing off the metal inside the panel, but it also throws away more than
half of the screen's own light. Samsung removed it in 2021 (Eco2 OLED):
about 33% more light out and up to 25% less power - but with weaker sunlight
readability, because the replacement lets more room light reflect back.

The idea tested here: that weakness is glare, and glare is exactly what the
moth-eye texture removes. So: remove the polarizer (more light out) AND add
a moth eye (less reflection), optionally with firefly extraction on top.

How the model decides brightness, minute by minute:
  - a comfortable brightness for the room light (like auto-brightness)
  - but never below what is needed to beat the glare: the screen must be at
    least CONTRAST_TARGET times brighter than the room light it reflects
  - if the needed brightness is above the panel's maximum, that minute is
    counted as UNREADABLE
Power follows the brightness and how much of the panel's light gets out.

All stack values are ASSUMPTION RANGES. The polarizer-free panel's internal
reflection is not published, so it is swept at the end to find the break-even.
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
GLASS_R = 0.044            # reflection of plain cover glass
CONTRAST_TARGET = 3.0      # screen light vs reflected room light, a common sunlight-readable bar

# stack layers: emission transmission, internal reflection (ranges)
STACKS = {
    "polarizer (today)":            {"pol": True,  "moth": False, "firefly": False},
    "polarizer-free":               {"pol": False, "moth": False, "firefly": False},
    "polarizer + moth eye":         {"pol": True,  "moth": True,  "firefly": False},
    "polarizer-free + moth eye":    {"pol": False, "moth": True,  "firefly": False},
    "polarizer-free + moth + firefly": {"pol": False, "moth": True, "firefly": True},
}
PARAMS = {
    "t_pol": (0.42, 0.46),          # light through a circular polarizer (~44%)
    "t_free_gain": (1.25, 1.40),    # polarizer-free: ~33% more light out
    "r_int_pol": (0.003, 0.008),    # room light leaking back from inside, with polarizer
    "r_int_free": (0.015, 0.04),    # same without polarizer (colour filter): NOT published
    "moth_r": (0.002, 0.01),        # surface reflection with moth-eye texture
    "firefly_gain": (0.15, 0.61),   # extra light out of the OLED
}

DEVICES = {
    # watts at 200 nits with the polarizer stack, and the panel's maximum brightness
    "phone": {"w_at_200": (0.20, 0.35), "max_nits": 1200,
              "users": {
                  "office worker": [(0.5, 20000), (3.0, 400), (1.5, 150), (0.5, 5)],
                  "outdoor worker": [(2.5, 30000), (1.0, 300), (1.0, 100)],
                  "night owl": [(0.5, 400), (2.0, 80), (2.0, 5)],
              }},
    "tv": {"w_at_200": (60.0, 110.0), "max_nits": 800,
           "users": {
               "evening viewer": [(4.0, 50)],
               "bright living room": [(2.0, 1000), (3.0, 100)],
               "sunny afternoon": [(3.0, 3000)],
           }},
}


def comfort_nits(lux):
    """Auto-brightness style target: dim rooms need little, bright rooms more."""
    return np.clip(40 * (max(lux, 1) / 10) ** 0.45, 10, 500)


def sample(rng):
    return {k: rng.uniform(*v, N) for k, v in PARAMS.items()}


def stack_optics(stack, p):
    """Returns (share of panel light that gets out, total reflection of room light)."""
    s = STACKS[stack]
    surface = p["moth_r"] if s["moth"] else np.full(N, GLASS_R)
    t_glass = (1 - surface) / (1 - GLASS_R)            # moth eye also lets a little more out
    if s["pol"]:
        t, r_int = p["t_pol"], p["r_int_pol"]
    else:
        t, r_int = p["t_pol"] * p["t_free_gain"], p["r_int_free"]
    extract = 1 + p["firefly_gain"] if s["firefly"] else 1.0
    return t * t_glass * extract, surface + r_int


def day(device, usage, stack, p, w_at_200):
    cfg = DEVICES[device]
    t_out, refl = stack_optics(stack, p)
    t_ref, _ = stack_optics("polarizer (today)", p)
    energy = np.zeros(N)
    unreadable = np.zeros(N)
    for hours, lux in usage:
        reflected = refl * lux / np.pi                  # nits of glare off the screen
        needed = np.maximum(comfort_nits(lux), (CONTRAST_TARGET - 1) * reflected)
        shown = np.minimum(needed, cfg["max_nits"])
        unreadable += np.where(needed > cfg["max_nits"], hours * 60, 0)
        watts = w_at_200 * (shown / 200) * (t_ref / t_out)   # same picture needs less drive if more gets out
        energy += watts * hours * 3600
    return energy, unreadable


def main():
    rng = np.random.default_rng(42)
    p = sample(rng)
    rows = []
    for device, cfg in DEVICES.items():
        w200 = rng.uniform(*cfg["w_at_200"], N)
        for user, usage in cfg["users"].items():
            base, _ = day(device, usage, "polarizer (today)", p, w200)
            for stack in STACKS:
                e, bad = day(device, usage, stack, p, w200)
                rows.append({
                    "device": device, "user": user, "stack": stack,
                    "screen_J_day": round(float(np.median(e)), 0),
                    "saved_%": round(float(np.median(100 * (1 - e / base))), 1),
                    "unreadable_min_day": round(float(np.median(bad)), 1),
                })
    df = pd.DataFrame(rows)
    df.to_csv(DOCS / "polarizer_results.csv", index=False)
    pd.set_option("display.width", 220)
    print(f"=== Screen energy per day by front stack (glare must stay {CONTRAST_TARGET:g}x below the picture) ===")
    print(df.to_string(index=False))

    # break-even: how much internal reflection can a polarizer-free panel have
    # before it loses to today's polarizer for the outdoor worker?
    print("\n=== Break-even check: polarizer-free internal reflection (not published) ===")
    usage = DEVICES["phone"]["users"]["outdoor worker"]
    w200 = np.full(N, 0.275)
    base, _ = day("phone", usage, "polarizer (today)", p, w200)
    sweep = []
    for r in (0.005, 0.01, 0.02, 0.03, 0.04, 0.06, 0.08):
        q = dict(p, r_int_free=np.full(N, r))
        for stack in ("polarizer-free", "polarizer-free + moth eye"):
            e, bad = day("phone", usage, stack, q, w200)
            sweep.append({"internal_reflection_%": 100 * r, "stack": stack,
                          "saved_%": round(float(np.median(100 * (1 - e / base))), 1),
                          "unreadable_min_day": round(float(np.median(bad)), 1)})
    sw = pd.DataFrame(sweep)
    print("Energy saved vs today (%):")
    print(sw.pivot(index="internal_reflection_%", columns="stack", values="saved_%").to_string())
    print("\nMinutes a day the screen cannot beat the glare (unreadable):")
    print(sw.pivot(index="internal_reflection_%", columns="stack", values="unreadable_min_day").to_string())
    print("(Savings in rows with unreadable minutes are not real: the screen is simply maxed out.)")
    sw.to_csv(DOCS / "polarizer_breakeven.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, device in zip(axes, DEVICES):
        d = df[df["device"] == device]
        users = list(DEVICES[device]["users"])
        x = np.arange(len(users))
        w = 0.8 / len(STACKS)
        for i, stack in enumerate(STACKS):
            vals = [d[(d["user"] == u) & (d["stack"] == stack)]["saved_%"].iloc[0] for u in users]
            ax.bar(x + (i - len(STACKS) / 2 + 0.5) * w, vals, w, label=stack)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x, users)
        ax.set_ylabel("Screen energy saved vs today (%)")
        ax.set_title(device)
    axes[0].legend(fontsize=7)
    fig.suptitle("Removing the polarizer, with and without a moth eye")
    fig.tight_layout()
    fig.savefig(DOCS / "polarizer.png", dpi=110)
    print(f"\nSaved: polarizer_results.csv, polarizer_breakeven.csv and polarizer.png in {DOCS}")


if __name__ == "__main__":
    main()
