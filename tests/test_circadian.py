"""Checks for the circadian (predictive) controller."""
import numpy as np
import pytest

import circadian as c
import day_sim as ds


def test_empty_memory_forecasts_no_energy():
    memory = np.zeros((7, 24, 3))
    seen = np.zeros((7, 24), dtype=bool)
    f = c.forecast_cumulative(memory, seen, 0, 0, np.arange(1, 169))
    assert np.all(f == 0)


def test_forecast_grows_with_learned_light():
    memory = np.ones((7, 24, 2))
    seen = np.ones((7, 24), dtype=bool)
    f = c.forecast_cumulative(memory, seen, 0, 0, np.arange(1, 169))
    assert np.all(np.diff(f, axis=0) >= 0)
    assert f[-1, 0] == pytest.approx(168)


def test_dark_forecast_means_slower_refresh():
    e = np.array([5.0]); cost = np.array([0.05]); sleep = np.array([10e-6])
    dark = np.zeros((len(c.HORIZONS_H), 1))
    bright = c.HORIZONS_H[:, None] * 2.0
    slow = c.interval_for("circadian", e, 10.0, cost, sleep, dark)
    fast = c.interval_for("circadian", e, 10.0, cost, sleep, bright)
    assert slow[0] > fast[0]


def test_circadian_interval_stays_within_limits():
    e = np.linspace(0, 10, 20); cost = np.full(20, 0.05); sleep = np.full(20, 10e-6)
    iv = c.interval_for("circadian", e, 10.0, cost, sleep, np.zeros((len(c.HORIZONS_H), 20)))
    assert np.all(iv >= c.MIN_I) and np.all(iv <= c.MAX_I)


def test_circadian_survives_learned_weekends_better_than_torpor(monkeypatch):
    monkeypatch.setattr(c, "DAYS", 15)      # two weekends: learn the first, face the second
    monkeypatch.setattr(c, "N", 20)
    p = ds.sample_params(np.random.default_rng(42), 20)
    lux, sun = c.build_scenario("office_weekends")
    ok = {}
    for ctrl in ("torpor", "circadian"):
        r = c.run(ctrl, lux, sun, p)
        ok[ctrl] = ((r["missed_late"] == 0) & (r["dead_late"] == 0)).mean()
    assert ok["circadian"] > ok["torpor"]
