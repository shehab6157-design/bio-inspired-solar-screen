"""Checks for the waggle dance model (small synthetic homes, no real data needed)."""
import numpy as np
import pytest

import hive as hv
import waggle as wg


@pytest.fixture
def home(monkeypatch):
    monkeypatch.setattr(hv, "H", 4)
    ep, _ = hv.sample(np.random.default_rng(0))
    days = 10
    minute = np.tile(np.arange(1440), days)
    daylight = np.where((minute > 8 * 60) & (minute < 16 * 60), 20000.0, 0.0)
    # every other day is cloudy (a quarter of the light)
    day_index = np.repeat(np.arange(days), 1440)
    daylight = daylight * np.where(day_index % 2, 0.25, 1.0)
    lux = np.array([daylight * f for f in hv.DAYLIGHT])
    sun = lux > 0
    return lux, sun, ep


def test_every_policy_runs_and_reports_sensible_numbers(home):
    lux, sun, ep = home
    for policy in wg.POLICIES:
        r = wg.run(policy, lux, sun, ep)
        assert np.all((r["dark_%"] >= 0) & (r["dark_%"] <= 100))
        assert np.all(r["refresh_day"] >= 0)


def test_planning_ahead_is_not_worse_than_reacting(home):
    lux, sun, ep = home
    torpor = np.median(wg.run("torpor", lux, sun, ep)["worst_room_dark_%"])
    plan = np.median(wg.run("self-forecast", lux, sun, ep)["worst_room_dark_%"])
    assert plan <= torpor + 1.0


def test_darkness_means_no_updates(home, monkeypatch):
    lux, sun, ep = home
    r = wg.run("waggle dance", lux * 0, sun & False, ep)
    assert np.all(r["refresh_day"] < 30)
