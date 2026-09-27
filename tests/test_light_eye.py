"""Checks for the light eye (two-band sensing)."""
import numpy as np

import light_eye as le


def test_led_light_has_almost_no_infrared_but_sunlight_has_lots():
    assert le.NIR_LED / le.VIS_LED < 0.02
    assert le.NIR_SUN / le.VIS_SUN > 0.5


def test_darkness_is_seen_as_dark():
    assert all(le.classify(np.zeros(5), np.zeros(5)) == "dark")


def test_bright_led_is_lamp_and_sunlight_is_day():
    lux = np.full(3, 500.0)
    assert all(le.classify(lux * le.VIS_LED, lux * le.NIR_LED) == "lamp")
    assert all(le.classify(lux * le.VIS_SUN, lux * le.NIR_SUN) == "day")


def test_eye_sees_daylight_through_window_glass_better_than_one_band_sensor():
    rng = np.random.default_rng(0)
    env = next(e for e in le.ENVIRONMENTS if e[0] == "window, low-e glass")
    _, v, n = le.readings(env, rng)
    assert np.mean(le.classify(v, n) == "day") > 0.95
    assert np.mean(le.classify_one_band(v) == "day") < 0.6


def test_dim_light_is_not_mistaken_for_daylight():
    rng = np.random.default_rng(1)
    env = next(e for e in le.ENVIRONMENTS if e[0] == "dim lamp")
    _, v, n = le.readings(env, rng)
    assert np.mean(le.classify(v, n) == "day") < 0.02
