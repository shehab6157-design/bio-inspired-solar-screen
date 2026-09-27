"""Checks for the Two Seas thermal model."""
import numpy as np
import pytest

import polarizer as pz
import two_seas as ts


@pytest.fixture(scope="module")
def results():
    p = ts.sample(np.random.default_rng(1))
    opt = pz.sample(np.random.default_rng(2))
    return {s: ts.run(s, p, opt) for s in ts.STACKS}


def test_sunlight_fractions_add_up():
    assert ts.UV + ts.VIS + ts.NIR == pytest.approx(1.0)


def test_heat_mirror_lowers_the_sun_heating(results):
    assert np.all(results["heat mirror"]["sun_heat_W"] < results["today"]["sun_heat_W"])


def test_heat_mirror_keeps_the_phone_cooler(results):
    assert np.median(results["heat mirror"]["peak_c"]) < np.median(results["today"]["peak_c"])


def test_only_the_harvest_layer_makes_power(results):
    for s in ts.STACKS:
        made = results[s]["harvested_J"].sum()
        assert (made > 0) == (s == "NIR harvest")


def test_moon_mode_keeps_the_screen_readable(results):
    moon = results["two seas + chameleon"]
    assert np.median(moon["unreadable_min"]) < np.median(results["today"]["unreadable_min"])
    assert np.median(moon["peak_c"]) < np.median(results["two seas"]["peak_c"])


def test_temperatures_stay_physical(results):
    for r in results.values():
        assert np.all(r["peak_c"] > 25) and np.all(r["peak_c"] < 90)
