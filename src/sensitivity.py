"""
Milestone 4: Sensitivity analysis (tornado charts).

Question: for each device's recommended stack, which assumption
changes the answer the most? Those are the numbers to measure first
on the bench prototype.

Method: one-at-a-time. Every assumption is fixed at the middle of its
range, then each one is swung to its low and high end while the rest
stay fixed. The wider the swing, the more that assumption matters.
"""
import copy
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import stack_config as sc

DOCS = sc.DOCS
LIGHT_LEVEL = (0.5, 1.5)   # real rooms can be half or 1.5x as bright as assumed

BASE_PARAMS = copy.deepcopy(sc.PARAMS)
BASE_TIERS = copy.deepcopy(sc.TIERS)


def mid(rng_pair):
    return (rng_pair[0] + rng_pair[1]) / 2


def run_fixed(tier, stack, param_override=None, tier_override=None, light=1.0):
    params = {k: (mid(v), mid(v)) for k, v in BASE_PARAMS.items()}
    tiers = copy.deepcopy(BASE_TIERS)
    t = tiers[tier]
    for key in ("load_j_day", "display_share"):
        t[key] = (mid(t[key]), mid(t[key]))
    t["front_light"] = [(lux * light, s, h) for lux, s, h in t["front_light"]]
    t["back_light"] = [(lux * light, s, h) for lux, s, h in t["back_light"]]
    if param_override:
        k, v = param_override
        params[k] = (v, v)
    if tier_override:
        k, v = tier_override
        t[k] = (v, v)
    sc.PARAMS, sc.TIERS = params, tiers
    try:
        result = sc.evaluate(tier, stack, np.random.default_rng(0))
    finally:
        sc.PARAMS, sc.TIERS = BASE_PARAMS, BASE_TIERS
    return result["coverage_%_median"]


def recommended_stacks():
    df = pd.read_csv(DOCS / "stack_results.csv")
    rec = (df[df["passes_readability"]]
           .sort_values("coverage_%_median", ascending=False)
           .groupby("tier", sort=False).head(1))
    return {r.tier: ([] if r.stack == "(none)" else r.stack.split("+"))
            for r in rec.itertuples()}


def main():
    stacks = recommended_stacks()
    rows = []
    for tier, stack in stacks.items():
        base = run_fixed(tier, stack)
        tests = [(k, "param", v) for k, v in BASE_PARAMS.items()]
        tests += [("load_j_day", "tier", BASE_TIERS[tier]["load_j_day"]),
                  ("display_share", "tier", BASE_TIERS[tier]["display_share"]),
                  ("light_level", "light", LIGHT_LEVEL)]
        for name, kind, (lo, hi) in tests:
            vals = []
            for v in (lo, hi):
                if kind == "param":
                    vals.append(run_fixed(tier, stack, param_override=(name, v)))
                elif kind == "tier":
                    vals.append(run_fixed(tier, stack, tier_override=(name, v)))
                else:
                    vals.append(run_fixed(tier, stack, light=v))
            rows.append({"tier": tier, "stack": "+".join(stack), "assumption": name,
                         "base_%": base, "at_low_%": vals[0], "at_high_%": vals[1],
                         "swing_pts": abs(vals[1] - vals[0])})

    df = pd.DataFrame(rows)
    df = df[df["swing_pts"] > 1e-9]
    df = df.sort_values(["tier", "swing_pts"], ascending=[True, False])
    num_cols = df.select_dtypes("number").columns
    df[num_cols] = df[num_cols].round(2)
    df.to_csv(DOCS / "sensitivity.csv", index=False)

    pd.set_option("display.width", 200)
    print("=== Top 3 assumptions driving each device's result ===")
    for tier in stacks:
        d = df[df["tier"] == tier].head(3)
        print(f"\n{tier}  (stack: {d['stack'].iloc[0]}, base coverage {d['base_%'].iloc[0]}%)")
        print(d[["assumption", "at_low_%", "at_high_%", "swing_pts"]].to_string(index=False))

    fig, axes = plt.subplots(len(stacks), 1, figsize=(10, 11))
    for ax, tier in zip(axes, stacks):
        d = df[df["tier"] == tier].sort_values("swing_pts")
        base = d["base_%"].iloc[0]
        lo = d["at_low_%"] - base
        hi = d["at_high_%"] - base
        ax.barh(d["assumption"], lo, left=base, color="tab:blue", label="assumption at low end")
        ax.barh(d["assumption"], hi, left=base, color="tab:orange", label="assumption at high end")
        ax.axvline(base, color="black", linewidth=0.8)
        ax.set_title(f"{tier} ({d['stack'].iloc[0]}): % of daily energy covered")
        ax.legend(loc="lower right", fontsize=8)
    axes[-1].set_xlabel("% of daily energy covered")
    fig.tight_layout()
    fig.savefig(DOCS / "sensitivity_tornado.png", dpi=120)
    print(f"\nSaved: {DOCS / 'sensitivity.csv'} and {DOCS / 'sensitivity_tornado.png'}")


if __name__ == "__main__":
    main()
