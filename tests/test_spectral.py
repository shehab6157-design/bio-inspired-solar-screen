"""Checks for the spectral light model."""
import numpy as np
import pytest

import spectral as sp


def test_eye_is_most_sensitive_near_555_nm():
    assert abs(sp.WL[np.argmax(sp.V)] - 555) <= 5


def test_lux_normalisation_is_exact():
    for src in ("sun", "led"):
        spec = sp.normalise_to_lux(sp.SPECTRA_SHAPE[src], 500)
        lux = sp.LM_PER_W * np.sum(spec * sp.V) * sp.DL
        assert lux == pytest.approx(500)


def test_derived_sun_constant_matches_older_model():
    assert 1 / sp.watts_per_lux("sun") == pytest.approx(1 / sp.sc.SUN_K, rel=0.1)


def test_led_room_light_has_almost_no_infrared():
    led = sp.normalise_to_lux(sp.SPECTRA_SHAPE["led"])
    ir_share = led[sp.WL > 750].sum() / led.sum()
    assert ir_share < 0.01


def test_infrared_band_barely_dims_the_screen():
    absorb = sp.dye_absorption(950, 150)
    assert sp.perceived_transmission(absorb, sp.SPECTRA_SHAPE["oled"]) > 0.99


def test_green_band_dims_the_screen_a_lot():
    absorb = sp.dye_absorption(550, 150)
    assert sp.perceived_transmission(absorb, sp.SPECTRA_SHAPE["oled"]) < 0.6


def test_no_absorption_means_no_harvest_and_no_dimming():
    none = np.zeros_like(sp.WL)
    r = sp.evaluate("phone", none, 0.25)
    assert r["harvest_J_day"] == 0
    assert r["readability_%"] == pytest.approx(100)


def test_invisible_band_beats_flat_dye_on_phone():
    flat = sp.evaluate("phone", sp.flat_dye(0.10), 0.25)["net_%"]
    ir = sp.evaluate("phone", sp.dye_absorption(1000, 240), 0.25)["net_%"]
    assert ir > flat
