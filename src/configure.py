"""
Milestone 6b: Any-device configurator.

Describe any screen device in a small JSON file and get its best
bio-inspired layer stack, using the same model as stack_config.py.

Usage (from the project folder):
    python3 src/configure.py devices/smartwatch.json
"""
import json
import sys
from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd

import stack_config as sc

REQUIRED = ["name", "display", "front_cm2", "back_cm2", "load_j_day",
            "display_share", "front_light", "back_light"]


def load_device(path):
    with open(path) as f:
        dev = json.load(f)
    missing = [k for k in REQUIRED if k not in dev]
    if missing:
        raise ValueError(f"{path}: missing fields {missing}")
    if dev["display"] not in ("reflective", "emissive"):
        raise ValueError(f"{path}: display must be 'reflective' or 'emissive'")
    return dev


def configure(dev):
    name = dev["name"]
    sc.TIERS[name] = {k: dev[k] for k in REQUIRED if k != "name"}
    rows = []
    for mask in product([0, 1], repeat=len(sc.LAYERS)):
        stack = [l for l, on in zip(sc.LAYERS, mask) if on]
        if dev["display"] == "reflective" and "firefly" in stack:
            continue
        rows.append(sc.evaluate(name, stack, np.random.default_rng(42)))
    df = pd.DataFrame(rows)
    df["passes_readability"] = df["readability_%"] >= sc.MIN_READABILITY
    return df.sort_values("coverage_%_median", ascending=False)


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 src/configure.py devices/<device>.json")
        sys.exit(1)
    for path in sys.argv[1:]:
        dev = load_device(path)
        df = configure(dev)
        best = df[df["passes_readability"]].iloc[0]
        print(f"\n=== {dev['name']} ({dev['display']}) ===")
        print(f"Recommended stack : {best['stack']}")
        print(f"Daily energy covered: {best['coverage_%_median']:.1f}% "
              f"(range {best['coverage_%_p10']:.1f}% to {best['coverage_%_p90']:.1f}%)")
        print(f"Readability       : {best['readability_%']:.1f}%")
        print("\nTop 5 stacks:")
        cols = ["stack", "coverage_%_median", "readability_%", "passes_readability"]
        print(df[cols].head(5).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
