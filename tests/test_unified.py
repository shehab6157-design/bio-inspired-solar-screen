"""Checks for the unified bio-controller."""
import numpy as np
import pytest

import unified as u


@pytest.fixture(scope="module")
def setup():
    old = (u.N, u.DAYS)
    u.N, u.DAYS = 20, 40                  # smaller run to keep tests quick
    lux, sun, reading = u.build_days()
    p = u.sample(np.random.default_rng(42))
    res = {s: u.run(s, lux, sun, reading, p) for s in u.STRATEGIES}
    yield lux, reading, res
    u.N, u.DAYS = old


def test_reading_schedule_is_two_and_a_half_hours_a_day(setup):
    _, reading, _ = setup
    assert reading.sum() / u.DAYS == pytest.approx(150, abs=1)


def test_each_step_lasts_longer(setup):
    _, _, res = setup
    med = {s: np.median(r[0]) for s, r in res.items()}
    assert med["glow_only"] < med["chameleon"] < med["unified"]


def test_only_unified_uses_the_cheaper_front_light(setup):
    _, _, res = setup
    assert res["glow_only"][1].sum() == 0
    assert res["chameleon"][1].sum() == 0
    assert res["unified"][1].sum() > 0


def test_empty_devices_stop_syncing(setup):
    _, _, res = setup
    days, _, syncs = res["glow_only"]
    assert np.all(syncs / np.maximum(days, 1) <= 145)
