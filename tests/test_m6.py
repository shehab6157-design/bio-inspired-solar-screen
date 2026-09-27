"""Checks for chameleon mode and the any-device configurator."""
import json
from pathlib import Path
import numpy as np
import pytest

import chameleon as ch
import configure as cf


@pytest.fixture(scope="module")
def p():
    return ch.sample(np.random.default_rng(42))


def test_chameleon_never_uses_more_energy_than_emissive(p):
    for usage in ch.USERS.values():
        results = {mode: energy for mode, energy, _ in ch.evaluate_user(usage, p)}
        assert np.all(results["chameleon"] <= results["emissive"] + 1e-9)


def test_chameleon_keeps_full_quality(p):
    for usage in ch.USERS.values():
        lowq = {mode: h for mode, _, h in ch.evaluate_user(usage, p)}
        assert lowq["chameleon"] == 0


def test_reflective_only_loses_quality_in_the_dark(p):
    lowq = {mode: h for mode, _, h in ch.evaluate_user(ch.USERS["night_owl"], p)}
    assert lowq["reflective"] > 0


def test_chameleon_uses_reflective_in_bright_light(p):
    w_bright, _ = ch.mode_power("chameleon", 20000, p)
    w_refl, _ = ch.mode_power("reflective", 20000, p)
    assert np.allclose(w_bright, w_refl)


def test_configurator_rejects_incomplete_device(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"name": "x", "display": "emissive"}))
    with pytest.raises(ValueError):
        cf.load_device(bad)


def test_configurator_recommends_no_firefly_for_reflective():
    df = cf.configure(cf.load_device("devices/shelf_label.json"))
    assert not df["stack"].str.contains("firefly").any()


def test_configurator_works_for_every_example_device():
    for path in Path("devices").glob("*.json"):
        assert len(cf.configure(cf.load_device(path))) > 0
