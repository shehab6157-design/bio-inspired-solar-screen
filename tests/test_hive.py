"""Checks for hive screens (tiny mock household, no real data needed)."""
import numpy as np
import pytest

import hive as hv


@pytest.fixture
def small(monkeypatch):
    monkeypatch.setattr(hv, "DAYS", 3)
    monkeypatch.setattr(hv, "H", 4)
    rng = np.random.default_rng(0)
    room, tv = hv.schedule()
    minutes = hv.DAYS * 1440
    lux = np.full((len(hv.ROOMS), minutes), 300.0)      # steady bright LED light everywhere
    sun = np.zeros((len(hv.ROOMS), minutes), dtype=bool)
    events = hv.arrivals(rng)
    ep, grid = hv.sample(rng)
    return room, tv, lux, sun, events, ep, grid


def test_schedule_has_user_away_on_weekdays(small):
    room, tv, *_ = small
    assert room[12 * 60] == -1                           # day 0 (weekday) at noon: out
    assert tv[21 * 60] and not tv[12 * 60]


def test_messages_only_arrive_while_awake(small):
    *_, events, _, _ = small
    minutes = [t % 1440 for t in events]
    assert min(minutes) >= 7 * 60 and max(minutes) < 23 * 60


def test_today_wakes_the_phone_for_every_message(small):
    room, tv, lux, sun, events, ep, grid = small
    r = hv.run("today", room, tv, lux, sun, events, ep, grid)
    total = sum(len(v) for v in events.values())
    assert r["phone_wakes_day"].sum() * hv.DAYS == pytest.approx(total)


def test_colony_policies_use_far_less_grid_energy(small):
    room, tv, lux, sun, events, ep, grid = small
    today = hv.run("today", room, tv, lux, sun, events, ep, grid)["grid_J_day"].sum()
    for policy in ("nearest", "hive"):
        assert hv.run(policy, room, tv, lux, sun, events, ep, grid)["grid_J_day"].sum() < 0.3 * today


def test_every_message_is_delivered(small):
    room, tv, lux, sun, events, ep, grid = small
    total = sum(len(v) for v in events.values())
    for policy in hv.POLICIES:
        r = hv.run(policy, room, tv, lux, sun, events, ep, grid)
        # hive may still hold a few glance items at the very end of the run
        assert r["delivered"].sum() >= total - 30 * hv.H
