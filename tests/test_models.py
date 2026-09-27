"""
Physics and behaviour checks for the simulators.
Each test states a rule the model must never break.
"""
import numpy as np
import pytest

import stack_config as sc
import day_sim as ds


def rng():
    return np.random.default_rng(42)


# ---------- light and harvesting ----------

def test_no_light_means_no_light_energy():
    assert sc.light_energy_j_per_m2([]) == 0
    assert sc.light_energy_j_per_m2([(0, "led", 8)]) == 0


def test_light_energy_scales_linearly_with_lux():
    one = sc.light_energy_j_per_m2([(300, "led", 4)])
    two = sc.light_energy_j_per_m2([(600, "led", 4)])
    assert two == pytest.approx(2 * one)


def test_daylight_carries_more_power_per_lux_than_led():
    # daylight includes infrared the eye barely sees, so each lux carries more watts
    assert sc.SUN_K > sc.LED_K


# ---------- layer behaviour ----------

def test_no_layers_changes_nothing():
    r = sc.evaluate("phone", [], rng())
    assert r["coverage_%_median"] == pytest.approx(0)
    assert r["readability_%"] == pytest.approx(100)


def test_firefly_does_nothing_on_reflective_display():
    r = sc.evaluate("e-reader", ["firefly"], rng())
    assert r["display_saved_J_day"] == 0


def test_firefly_saves_energy_on_emissive_displays():
    for tier in ("phone", "tv"):
        assert sc.evaluate(tier, ["firefly"], rng())["display_saved_J_day"] > 0


def test_dye_layer_is_net_loss_on_emissive_displays():
    for tier in ("phone", "tv"):
        assert sc.evaluate(tier, ["dye"], rng())["coverage_%_median"] < 0


def test_dye_layer_is_net_gain_on_e_reader():
    assert sc.evaluate("e-reader", ["dye"], rng())["coverage_%_median"] > 0


def test_moth_eye_improves_transmission():
    assert sc.evaluate("phone", ["moth_eye"], rng())["readability_%"] > 100


def test_moth_eye_restores_e_reader_readability_with_dye():
    dye_only = sc.evaluate("e-reader", ["dye"], rng())["readability_%"]
    both = sc.evaluate("e-reader", ["moth_eye", "dye"], rng())["readability_%"]
    assert both > dye_only


def test_results_are_reproducible():
    a = sc.evaluate("tv", ["moth_eye", "firefly"], rng())
    b = sc.evaluate("tv", ["moth_eye", "firefly"], rng())
    assert a == b


# ---------- 7-day simulation ----------

@pytest.fixture(scope="module")
def params():
    return ds.sample_params(np.random.default_rng(1), 50)


def test_darkness_drains_the_store(params):
    dark = [(0, 24, 0, "led")]
    r = ds.simulate(dark, "fixed", params, 50)
    assert np.all(r["harvested_j"] == 0)
    assert np.all(r["end_soc"] < ds.DEVICE["start_soc"])


def test_stored_energy_stays_within_limits(params):
    r = ds.simulate(ds.PROFILES["near_window"], "torpor", params, 50)
    assert r["soc_trace"].min() >= 0
    assert r["soc_trace"].max() <= 1


def test_torpor_is_never_worse_than_fixed_in_dim_rooms(params):
    fixed = ds.simulate(ds.PROFILES["home_dim"], "fixed", params, 50)
    torpor = ds.simulate(ds.PROFILES["home_dim"], "torpor", params, 50)
    ok = lambda r: ((r["missed"] == 0) & (r["dead_min"] == 0)).mean()
    assert ok(torpor) >= ok(fixed)


def test_torpor_slows_down_when_energy_is_low():
    soc = np.array([0.9, 0.5, 0.3, 0.1])
    intervals = ds.interval_minutes("torpor", soc)
    assert list(intervals) == sorted(intervals)
