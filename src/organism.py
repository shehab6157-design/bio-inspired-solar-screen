"""
The organism: every layer working together, device by device.

A body doesn't run its eyes, skin and metabolism as separate machines; one
energy budget serves them all. This script stacks the project's layers one
at a time on a real day of use and shows what each one adds:

PHONE (office worker's day: indoors, a little sun, a dark evening)
  today                     : polarizer OLED, every message wakes the phone
  + moth eye, no polarizer  : Step 7
  + firefly extraction      : Step 3/7
  + moon mode               : reflective whenever the room is bright (Step 6a)
  + hive                    : messages go to the right screen (Step 6)
  and, separately, an hour in summer sun (Step 8 - Two Seas)

TV (evening viewer)       : today -> moth eye, no polarizer -> + firefly
E-READER (60 days)        : glowing only -> chameleon -> unified glide path
HOME SCREENS (real light) : read from the saved placement, hive and climate runs
                            (they take minutes; regenerate them with run_all.sh)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import polarizer as pz
import chameleon as ch
import two_seas as ts
import unified as un
import hive as hv

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
DOCS.mkdir(exist_ok=True)

PHONE_DAY = pz.DEVICES["phone"]["users"]["office worker"]
TV_DAY = pz.DEVICES["tv"]["users"]["evening viewer"]
MESSAGES_PER_DAY = sum(per_day for per_day, _ in hv.KINDS.values())
URGENT_PER_DAY = hv.KINDS["urgent"][0]


def med(x):
    return float(np.median(x))


def phone_ledger():
    rng = np.random.default_rng(42)
    p = pz.sample(rng)
    w200 = rng.uniform(*pz.DEVICES["phone"]["w_at_200"], pz.N)
    pc = ch.sample(np.random.default_rng(3))
    wake_j, watch_j = np.mean((4, 10)), np.mean((0.5, 1.2))       # as in hive.py

    def screen(stack):
        return pz.day("phone", PHONE_DAY, stack, p, w200)[0]

    def moon():
        total = np.zeros(pz.N)
        for hours, lux in PHONE_DAY:
            if lux >= ch.SWITCH_LUX:        # bright enough: the room lights the picture
                total += pc["reflective_mw_cm2"] * ch.SCREEN_CM2 / 1000 * hours * 3600
            else:
                total += pz.day("phone", [(hours, lux)], "polarizer-free + moth + firefly", p, w200)[0]
        return total

    today_msgs = MESSAGES_PER_DAY * (wake_j + watch_j)
    # hive, as measured in hive.py on real light: urgent alerts still wake phone + watch,
    # about 60% of the rest are shown free on e-paper, the other 40% on the watch
    hive_msgs = URGENT_PER_DAY * (wake_j + watch_j) + (MESSAGES_PER_DAY - URGENT_PER_DAY) * 0.4 * watch_j
    steps = [
        ("today", screen("polarizer (today)"), today_msgs, MESSAGES_PER_DAY),
        ("+ moth eye, no polarizer", screen("polarizer-free + moth eye"), today_msgs, MESSAGES_PER_DAY),
        ("+ firefly extraction", screen("polarizer-free + moth + firefly"), today_msgs, MESSAGES_PER_DAY),
        ("+ moon mode", moon(), today_msgs, MESSAGES_PER_DAY),
        ("+ hive", moon(), hive_msgs, URGENT_PER_DAY),
    ]
    base = med(steps[0][1]) + steps[0][2]
    rows = []
    for name, scr, msgs, wakes in steps:
        total = med(scr) + msgs
        rows.append({"layer": name, "screen_J": round(med(scr)), "messages_J": round(msgs),
                     "total_J_day": round(total), "saved_%": round(100 * (1 - total / base), 1),
                     "phone_wakes_day": wakes})
    return pd.DataFrame(rows)


def sun_hour():
    p = ts.sample(np.random.default_rng(1))
    opt = pz.sample(np.random.default_rng(2))
    out = []
    for stack in ("today", "two seas + chameleon"):
        r = ts.run(stack, p, opt)
        out.append({"phone": stack, "peak_C": round(med(r["peak_c"]), 1),
                    "unreadable_min_of_60": round(med(r["unreadable_min"]), 1)})
    return pd.DataFrame(out)


def tv_ledger():
    rng = np.random.default_rng(42)
    p = pz.sample(rng)
    w200 = rng.uniform(*pz.DEVICES["tv"]["w_at_200"], pz.N)
    rows, base = [], None
    for name, stack in (("today", "polarizer (today)"),
                        ("+ moth eye, no polarizer", "polarizer-free + moth eye"),
                        ("+ firefly extraction", "polarizer-free + moth + firefly")):
        j = med(pz.day("tv", TV_DAY, stack, p, w200)[0])
        base = base or j
        rows.append({"layer": name, "kWh_per_year": round(j * 365 / 3.6e6, 1),
                     "saved_%": round(100 * (1 - j / base), 1)})
    return pd.DataFrame(rows)


def ereader_ledger():
    old = (un.N, un.DAYS)
    un.N, un.DAYS = 60, 60
    try:
        lux, sun, reading = un.build_days()
        p = un.sample(np.random.default_rng(42))
        rows = []
        for s in un.STRATEGIES:
            days, _, _ = un.run(s, lux, sun, reading, p)
            rows.append({"layer": {"glow_only": "glowing screen only", "chameleon": "+ chameleon",
                                   "unified": "+ unified glide path"}[s],
                         "battery_days": round(med(np.minimum(days, un.DAYS)), 1)})
    finally:
        un.N, un.DAYS = old
    return pd.DataFrame(rows)


def home_screens():
    lines = []
    f = DOCS / "placement_results.csv"
    if f.exists():
        d = pd.read_csv(f)
        d = d[(d["lamp"] == "daylight only") & (d["survived_winter_%"] >= 95)]
        for ctrl in ("fixed", "torpor", "circadian"):
            x = d[d["controller"] == ctrl]
            if len(x):
                lines.append(f"placement  {ctrl:9s}: survives real winter down to "
                             f"{x['daylight_factor_%'].min():g}% daylight ({x.sort_values('daylight_factor_%').iloc[0]['place']})")
    f = DOCS / "climates_results.csv"
    if f.exists():
        d = pd.read_csv(f)
        d = d[d["home"] == "daylight only"]
        for city in d["city"].unique():
            n = d[(d.city == city) & (d.policy == "nearest")]["epaper_dark_time_%"].iloc[0]
            h = d[(d.city == city) & (d.policy == "hive")]["epaper_dark_time_%"].iloc[0]
            lines.append(f"hive       {city:16s}: e-paper dark time {n:.1f}% -> {h:.1f}% with bee thresholds")
    return lines or ["(no saved real-light results yet - run ./run_all.sh once)"]


def main():
    pd.set_option("display.width", 200)
    phone = phone_ledger()
    print("=== PHONE: one office worker's day, layer by layer ===")
    print(phone.to_string(index=False))
    sun = sun_hour()
    print("\n=== PHONE: one hour in summer sun ===")
    print(sun.to_string(index=False))
    tv = tv_ledger()
    print("\n=== TV: evening viewer, 4 hours a day ===")
    print(tv.to_string(index=False))
    er = ereader_ledger()
    print("\n=== E-READER: days per charge (6 Wh battery) ===")
    print(er.to_string(index=False))
    print("\n=== HOME SCREENS: real sunlight results ===")
    for line in home_screens():
        print(line)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].bar(phone["layer"], phone["total_J_day"], color="tab:green")
    axes[0].set_title("Phone: screen + message energy (J/day)")
    axes[1].bar(tv["layer"], tv["kWh_per_year"], color="tab:blue")
    axes[1].set_title("TV: kWh per year")
    axes[2].bar(er["layer"], er["battery_days"], color="tab:orange")
    axes[2].set_title("E-reader: days per charge")
    for ax in axes:
        ax.tick_params(axis="x", labelsize=8, rotation=25)
    fig.suptitle("The organism: every layer working together")
    fig.tight_layout()
    fig.savefig(DOCS / "organism.png", dpi=110)
    phone.to_csv(DOCS / "organism_phone.csv", index=False)
    tv.to_csv(DOCS / "organism_tv.csv", index=False)
    er.to_csv(DOCS / "organism_ereader.csv", index=False)
    print(f"\nSaved: organism.png and organism_*.csv in {DOCS}")


if __name__ == "__main__":
    main()
