"""
Step 6: Hive screens - a household of screens that works like a bee colony.

Today every notification wakes the phone (and buzzes the watch), even when
a screen that runs on free room light is right in front of you.

In a bee colony there is no boss. Each worker has a RESPONSE THRESHOLD for
each task: a well-fed forager accepts work easily, a tired one only when the
need is urgent (the response-threshold model of division of labour). Work
flows to whoever is best placed right now, and nobody does the same job twice.

The household colony:
  phone and smartwatch (battery, charged from the grid, always with you)
  TV in the living room (only used as an overlay while it is already on)
  5 small light-powered e-paper screens: kitchen, entrance, desk, living
  room, bedroom - each gets REAL winter light for northern Israel (PVGIS,
  November) at its own distance from a window, plus an evening lamp.
  Each e-paper also has its own job: refresh its dashboard every 30 minutes.

Messages of three kinds arrive between 07:00 and 23:00:
  glance  (weather, calendar, parcel) : 30 a day, may wait up to 2 hours
  message (chat, email)               : 40 a day, should be seen within 5 minutes
  urgent  (call, alarm)               :  3 a day, phone + watch at once, always

Policies compared (30 simulated households, 28 winter days):
  today    : everything wakes the phone and buzzes the watch
  nearest  : show it on the closest screen you can see, cheapest first
  hive     : response thresholds + waiting: glance items wait until you walk
             past an e-paper that has energy to spare; tired e-papers (low
             store) decline work unless it is at its deadline; and each
             e-paper paces its own dashboard with torpor (slower when low),
             so the layers from earlier steps work together inside the colony

Tested in two homes: with an evening lamp, and without (daylight only).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import day_sim as ds
import placement as pl

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
DOCS.mkdir(exist_ok=True)

DAYS = 28
START_DAY = 61                           # winter series is Nov, Dec, Jan, Feb: day 61 = 1 January
H = 30                                   # households
ROOMS = ["kitchen", "entrance", "desk", "living", "bedroom"]
DAYLIGHT = [0.02, 0.005, 0.01, 0.01, 0.003]    # share of outdoor light per room
DASHBOARD_EVERY = 30                     # minutes between an e-paper's own refreshes
THRESHOLD = 0.3                          # hive: e-paper accepts work above 30% stored energy
RESERVE = 0.05
POLICIES = ["today", "nearest", "hive"]
KINDS = {"glance": (30, 120), "message": (40, 5), "urgent": (3, 0)}   # per day, deadline min

# where the person is: (start_hour, end_hour, room); "out" = away from home
WEEKDAY = [(0, 7, "bedroom"), (7, 8, "kitchen"), (8, 8.1, "entrance"), (8.1, 17, "out"),
           (17, 17.1, "entrance"), (17, 19, "kitchen"), (19, 23, "living"), (23, 24, "bedroom")]
WEEKEND = [(0, 9, "bedroom"), (9, 10, "kitchen"), (10, 13, "living"), (13, 14, "kitchen"),
           (14, 18, "desk"), (18, 20, "kitchen"), (20, 23, "living"), (23, 24, "bedroom")]
TV_HOURS = (20, 23)


def schedule():
    """Per-minute room index (-1 = out) and TV-on flag for DAYS days."""
    room = np.full(DAYS * 1440, -1)
    tv = np.zeros(DAYS * 1440, dtype=bool)
    for d in range(DAYS):
        for h0, h1, r in (WEEKEND if d % 7 in (5, 6) else WEEKDAY):
            a, b = d * 1440 + int(h0 * 60), d * 1440 + int(h1 * 60)
            room[a:b] = -1 if r == "out" else ROOMS.index(r)
        a = d * 1440
        tv[a + TV_HOURS[0] * 60:a + TV_HOURS[1] * 60] = True
    return room, tv


def room_light(lamp=True):
    """Real winter light for each room, shape (rooms, minutes)."""
    lux, sun = [], []
    for df_value in DAYLIGHT:
        l, s, _ = pl.winter_light(df_value, lamp=lamp)
        a = START_DAY * 1440
        lux.append(l[a:a + DAYS * 1440]); sun.append(s[a:a + DAYS * 1440])
    return np.array(lux), np.array(sun)


def arrivals(rng):
    """List per minute of (household, kind) arrivals, between 07:00 and 23:00."""
    events = {}
    for h in range(H):
        for d in range(DAYS):
            for kind, (per_day, _) in KINDS.items():
                for _ in range(rng.poisson(per_day)):
                    t = d * 1440 + int(rng.uniform(7 * 60, 23 * 60))
                    events.setdefault(t, []).append((h, kind))
    return events


def sample(rng):
    p = ds.sample_params(rng, H * len(ROOMS))
    ep = {k: (v.reshape(H, len(ROOMS)) if isinstance(v, np.ndarray) else v) for k, v in p.items()}
    grid = {"phone": rng.uniform(4, 10, H),      # J per phone wake (screen + chip)
            "watch": rng.uniform(0.5, 1.2, H),   # J per watch glance
            "tv": rng.uniform(0.2, 1.0, H)}      # J for an overlay on a TV that is already on
    return ep, grid


def run(policy, room, tv, lux, sun, events, ep, grid):
    cap = ds.DEVICE["storage_j"]
    e = np.full((H, len(ROOMS)), cap * ds.DEVICE["start_soc"])
    grid_j = np.zeros(H); phone_wakes = np.zeros(H)
    shown = np.zeros(H); on_time = np.zeros(H); free = np.zeros(H)
    dark = np.zeros((H, len(ROOMS)))              # minutes an e-paper cannot afford an update
    refreshes = np.zeros((H, len(ROOMS)))
    next_dash = np.zeros((H, len(ROOMS)))
    queue = [[] for _ in range(H)]                  # waiting glance items: (created, deadline)

    def epaper_can(h, r, soc_min):
        return e[h, r] - ep["cost_j"][h, r] >= cap * soc_min

    def show(h, where, created, t, deadline):
        pay(h, where)
        shown[h] += 1
        on_time[h] += (t - created) <= deadline

    def pay(h, where):
        if where == "phone+watch":
            grid_j[h] += grid["phone"][h] + grid["watch"][h]; phone_wakes[h] += 1
        elif where == "phone":
            grid_j[h] += grid["phone"][h]; phone_wakes[h] += 1
        elif where == "watch":
            grid_j[h] += grid["watch"][h]
        elif where == "tv":
            grid_j[h] += grid["tv"][h]
        else:                                        # an e-paper room index
            e[h, where] -= ep["cost_j"][h, where]; free[h] += 1

    for t in range(DAYS * 1440):
        m = t % 1440
        awake = 7 * 60 <= m < 23 * 60
        # e-papers harvest light and keep their own dashboards fresh
        k = np.where(sun[:, t], ds.SUN_K * ep["eff_sun"], ds.LED_K * ep["eff_led"])
        e += lux[:, t] * k * ep["area_m2"] * 60 * ep["conv"] - ep["sleep_w"] * 60
        due = t >= next_dash
        if due.any():
            ok = due & (e - ep["cost_j"] >= cap * RESERVE)
            e = np.where(ok, e - ep["cost_j"], e)
            refreshes += ok
            if policy == "hive":                     # torpor pacing inside the colony
                iv = ds.interval_minutes("torpor", e / cap)
            else:
                iv = np.full(e.shape, float(DASHBOARD_EVERY))
            next_dash = np.where(due, t + iv, next_dash)
        e = np.clip(e, 0, cap)
        dark += e - ep["cost_j"] < cap * RESERVE

        r = room[t]
        # hive: waiting glance items go to an e-paper with energy to spare, or to the watch at the deadline
        if policy == "hive":
            for h in range(H):
                keep = []
                for created, deadline in queue[h]:
                    if r >= 0 and epaper_can(h, r, THRESHOLD):
                        show(h, r, created, t, deadline)
                    elif t - created >= deadline:
                        show(h, "watch" if awake else "phone", created, t, deadline)
                    else:
                        keep.append((created, deadline))
                queue[h] = keep

        for h, kind in events.get(t, []):
            deadline = KINDS[kind][1]
            if policy == "today" or kind == "urgent":
                show(h, "phone+watch", t, t, deadline)
            elif policy == "nearest":
                if r >= 0 and epaper_can(h, r, RESERVE):
                    show(h, r, t, t, deadline)
                elif r == ROOMS.index("living") and tv[t]:
                    show(h, "tv", t, t, deadline)
                else:
                    show(h, "watch", t, t, deadline)
            else:  # hive
                if r >= 0 and epaper_can(h, r, THRESHOLD):
                    show(h, r, t, t, deadline)
                elif kind == "glance":
                    queue[h].append((t, deadline))    # wait for a well-fed e-paper
                elif r == ROOMS.index("living") and tv[t]:
                    show(h, "tv", t, t, deadline)
                else:
                    show(h, "watch", t, t, deadline)

    return {
        "grid_J_day": grid_j / DAYS,
        "phone_wakes_day": phone_wakes / DAYS,
        "on_time_%": 100 * on_time / np.maximum(shown, 1),
        "on_free_light_%": 100 * free / np.maximum(shown, 1),
        "epaper_dark_time_%": 100 * dark.mean(axis=1) / (DAYS * 1440),
        "dashboard_refresh_day": refreshes.mean(axis=1) / DAYS,
        "delivered": shown,
    }


def main():
    rng = np.random.default_rng(11)
    room, tv = schedule()
    events = arrivals(rng)
    ep, grid = sample(rng)
    rows = []
    for lamp in (True, False):
        lux, sun = room_light(lamp)
        for policy in POLICIES:
            r = run(policy, room, tv, lux, sun, events, ep, grid)
            rows.append({"home": "evening lamp" if lamp else "daylight only", "policy": policy,
                         **{k: round(float(np.median(v)), 1) for k, v in r.items() if k != "delivered"}})
            print(f"done: {policy} ({'lamp' if lamp else 'no lamp'})")
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(f"\n=== Hive screens: {H} households, {DAYS} real winter days, median household ===")
    print(df.to_string(index=False))
    df.to_csv(DOCS / "hive_results.csv", index=False)

    d = df[df["home"] == "daylight only"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    colors = ["tab:red", "tab:blue", "tab:green"]
    for ax, col, title in zip(axes, ["phone_wakes_day", "grid_J_day", "on_free_light_%", "epaper_dark_time_%"],
                              ["Phone wake-ups per day", "Grid energy for messages (J/day)",
                               "Messages shown on free light (%)", "Time an e-paper can't update (%)"]):
        ax.bar(d["policy"], d[col], color=colors)
        ax.set_title(title, fontsize=10)
    fig.suptitle("Hive screens (daylight-only home, real winter light)")
    fig.tight_layout()
    fig.savefig(DOCS / "hive.png", dpi=110)
    print(f"\nSaved: {DOCS / 'hive_results.csv'} and {DOCS / 'hive.png'}")


if __name__ == "__main__":
    main()
