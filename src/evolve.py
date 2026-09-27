"""
Step 4 (advanced): Evolution-based optimizer.

So far WE chose the controller settings by hand (switch at 300 lux, plan
for 30 days, sync every 10 or 60 minutes). Nature doesn't design by hand:
it breeds, varies and keeps what survives.

Here a genetic algorithm (NSGA-II style: keeps a whole front of best
trade-offs, not just one winner) evolves the unified controller of the
hybrid e-reader from unified.py. Each "creature" is one set of 4 genes:

  switch_lux : light level above which the screen goes reflective (200-1500)
  plan_days  : how far the glide path plans ahead (20-50 days)
  fast_sync  : sync interval when energy is plentiful (5-30 min)
  slow_sync  : sync interval when saving (30-240 min)

Two goals pull against each other:
  1. battery life      - days per charge (more is better)
  2. reading comfort   - share of dim-light reading on the glowing screen
                         instead of the cheaper front-light look (more is better)
Rule: at least 24 background syncs a day (roughly hourly), or the design is rejected.

The result is a Pareto front: the set of designs where you cannot improve
one goal without losing the other. A maker picks a point on it.

Speed trick: every creature x every simulated device runs side by side in
one vectorised simulation, so a whole generation costs one run.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import day_sim as ds
import unified as u

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
DOCS.mkdir(exist_ok=True)

DAYS = 40
DEVICES = 25            # simulated devices per creature
POP = 24
GENERATIONS = 15
MIN_SYNCS = 24
GENES = {"switch_lux": (200, 1500), "plan_days": (20, 50),
         "fast_sync": (5, 30), "slow_sync": (30, 240)}
HAND_DESIGN = {"switch_lux": 300, "plan_days": 30, "fast_sync": 10, "slow_sync": 60}


def evaluate(pop, lux, sun, reading, p):
    """pop: array (P, 4). Returns battery days, comfort %, syncs/day per creature."""
    P = len(pop)
    n = P * DEVICES
    g = {k: np.repeat(pop[:, i], DEVICES) for i, k in enumerate(GENES)}
    tile = lambda a: np.tile(a, P)
    base_w = tile(p["emissive_base_w"]); sun_w = tile(p["emissive_sun_extra_w"])
    front = tile(p["front_w"]); refl = tile(p["reflect_w"]); sync_j = tile(p["sync_mj"]) * 1e-3
    sleep = tile(p["sleep_w"]); eff_s = tile(p["eff_sun"]); eff_l = tile(p["eff_led"]); conv = tile(p["conv"])
    scale = u.SCREEN_CM2 / 100
    cap = u.BATTERY_J

    e = np.full(n, float(cap))
    empty = np.full(n, np.nan)
    glow_min = np.zeros(n); dim_min = np.zeros(n)
    syncs = np.zeros(n); next_sync = np.zeros(n)

    for t in range(len(lux)):
        alive = np.isnan(empty)
        on_track = e >= cap * (0.05 + 0.95 * (1 - t / (g["plan_days"] * 1440)))
        k = ds.SUN_K * eff_s if sun[t] else ds.LED_K * eff_l
        h = lux[t] * k * p["area_m2"] * 60 * conv
        use = sleep * 60
        if reading[t]:
            bright = lux[t] >= g["switch_lux"]
            glow = scale * (base_w + sun_w * min(1, lux[t] / 20000)) * 60
            cheap = (refl + front) * 60
            use = use + np.where(bright, refl * 60, np.where(on_track, glow, cheap))
            dim = ~bright & alive
            dim_min += dim
            glow_min += dim & on_track
        due = t >= next_sync
        if due.any():
            use = use + np.where(due, sync_j, 0)
            syncs += due & alive
            next_sync = np.where(due, t + np.where(on_track, g["fast_sync"], g["slow_sync"]), next_sync)
        use = np.where(alive, use, 0.0)
        e = np.minimum(e + h - use, cap)
        empty[(e <= 0) & np.isnan(empty)] = t / 1440
        e = np.maximum(e, 0)

    days = np.where(np.isnan(empty), DAYS, empty).reshape(P, DEVICES)
    comfort = (100 * glow_min / np.maximum(dim_min, 1)).reshape(P, DEVICES)
    spd = (syncs / np.maximum(days.ravel(), 1)).reshape(P, DEVICES)
    return np.median(days, 1), np.median(comfort, 1), np.median(spd, 1)


def dominates(a, b):
    return np.all(a >= b) and np.any(a > b)


def fronts(objs):
    """Non-dominated sorting: returns rank per creature (0 = best front)."""
    n = len(objs)
    rank = np.full(n, -1)
    remaining = set(range(n))
    r = 0
    while remaining:
        front = [i for i in remaining if not any(dominates(objs[j], objs[i]) for j in remaining if j != i)]
        for i in front:
            rank[i] = r
        remaining -= set(front)
        r += 1
    return rank


def crowding(objs, idx):
    d = np.zeros(len(idx))
    for m in range(objs.shape[1]):
        order = np.argsort(objs[idx, m])
        d[order[0]] = d[order[-1]] = np.inf
        span = objs[idx, m].max() - objs[idx, m].min() or 1
        for k in range(1, len(idx) - 1):
            d[order[k]] += (objs[idx[order[k + 1]], m] - objs[idx[order[k - 1]], m]) / span
    return d


def select(objs, size):
    rank = fronts(objs)
    chosen = []
    for r in range(rank.max() + 1):
        idx = np.where(rank == r)[0]
        if len(chosen) + len(idx) <= size:
            chosen += list(idx)
        else:
            cd = crowding(objs, idx)
            chosen += list(idx[np.argsort(-cd)][:size - len(chosen)])
            break
    return np.array(chosen)


def breed(parents, rng):
    lo = np.array([v[0] for v in GENES.values()], float)
    hi = np.array([v[1] for v in GENES.values()], float)
    kids = []
    for _ in range(len(parents)):
        a, b = parents[rng.integers(len(parents), size=2)]
        w = rng.random(len(lo))
        child = w * a + (1 - w) * b                                   # crossover
        mutate = rng.random(len(lo)) < 0.3
        child += mutate * rng.normal(0, 0.1, len(lo)) * (hi - lo)    # mutation
        kids.append(np.clip(child, lo, hi))
    return np.array(kids)


def objectives(days, comfort, spd):
    """Designs that sync less than MIN_SYNCS a day are pushed to the bottom."""
    penalty = np.where(spd < MIN_SYNCS, -1000, 0)
    return np.column_stack([days + penalty, comfort + penalty])


def main():
    old = (u.DAYS, u.N)
    u.DAYS, u.N = DAYS, DEVICES
    lux, sun, reading = u.build_days()
    p = u.sample(np.random.default_rng(42))
    u.DAYS, u.N = old
    rng = np.random.default_rng(7)

    lo = np.array([v[0] for v in GENES.values()], float)
    hi = np.array([v[1] for v in GENES.values()], float)
    pop = lo + rng.random((POP, len(GENES))) * (hi - lo)
    pop[0] = list(HAND_DESIGN.values())                     # include our hand design
    days, comfort, spd = evaluate(pop, lux, sun, reading, p)
    hand = (days[0], comfort[0], spd[0])
    archive = [np.column_stack([pop, days, comfort, spd])]

    for gen in range(GENERATIONS):
        kids = breed(pop, rng)
        kd, kc, ks = evaluate(kids, lux, sun, reading, p)
        allpop = np.vstack([pop, kids])
        alld, allc, alls = np.concatenate([days, kd]), np.concatenate([comfort, kc]), np.concatenate([spd, ks])
        keep = select(objectives(alld, allc, alls), POP)
        pop, days, comfort, spd = allpop[keep], alld[keep], allc[keep], alls[keep]
        archive.append(np.column_stack([kids, kd, kc, ks]))
        best = fronts(objectives(days, comfort, spd)) == 0
        print(f"generation {gen + 1:2d}: {best.sum():2d} designs on the best front, "
              f"longest battery {days.max():.1f} days, best comfort {comfort.max():.0f}%")

    cols = list(GENES) + ["battery_days", "comfort_%", "syncs_per_day"]
    df = pd.DataFrame(np.vstack(archive), columns=cols)
    df = df[df["syncs_per_day"] >= MIN_SYNCS]
    df = df.round(1)
    rank = fronts(df[["battery_days", "comfort_%"]].to_numpy())
    pareto = (df[rank == 0].sort_values(["battery_days", "comfort_%"], ascending=[True, False])
              .drop_duplicates(subset=["battery_days", "comfort_%"]))
    pareto.to_csv(DOCS / "pareto_front.csv", index=False)
    show = pareto.iloc[np.unique(np.linspace(0, len(pareto) - 1, min(8, len(pareto))).astype(int))]

    pd.set_option("display.width", 200)
    print(f"\nHand design (300 lux, 30-day plan, 10/60 min sync): "
          f"{hand[0]:.1f} days, {hand[1]:.0f}% comfort, {hand[2]:.0f} syncs/day")
    print(f"\n=== Pareto front: {len(pareto)} best trade-offs found (8 shown, all in the CSV) ===")
    print(show.to_string(index=False))
    better = pareto[(pareto["battery_days"] >= hand[0]) & (pareto["comfort_%"] > hand[1])]
    if len(better):
        b = better.sort_values("comfort_%").iloc[-1]
        print(f"\nEvolution beat the hand design: {b['battery_days']} days with "
              f"{b['comfort_%']}% comfort (hand: {hand[0]:.1f} days, {hand[1]:.0f}%)")
    else:
        print("\nNo evolved design beats the hand design on both goals at once: "
              "the hand design already sits on the best trade-off curve.")

    print("\n=== Which genes matter? (correlation with each goal, all designs tried) ===")
    for gene in GENES:
        r_days = np.corrcoef(df[gene], df["battery_days"])[0, 1]
        r_comf = np.corrcoef(df[gene], df["comfort_%"])[0, 1]
        print(f"{gene:11s} battery {r_days:+.2f}   comfort {r_comf:+.2f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df["battery_days"], df["comfort_%"], s=10, alpha=0.3, label="all designs tried")
    ax.plot(pareto["battery_days"], pareto["comfort_%"], "o-", color="tab:green", label="Pareto front")
    ax.plot(hand[0], hand[1], "r*", markersize=16, label="our hand design")
    ax.axvline(u.TARGET_DAYS, color="black", ls="--", lw=1, label="30-day target")
    ax.set_xlabel("Battery days per charge")
    ax.set_ylabel("Reading comfort: % of dim reading on the glowing screen")
    ax.set_title("Evolved controller designs: battery life vs comfort")
    ax.legend()
    fig.tight_layout()
    fig.savefig(DOCS / "pareto_front.png", dpi=110)
    print(f"\nSaved: {DOCS / 'pareto_front.csv'} and {DOCS / 'pareto_front.png'}")


if __name__ == "__main__":
    main()
