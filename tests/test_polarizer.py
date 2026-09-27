"""Checks for the polarizer-free + moth eye model."""
import numpy as np
import pytest

import polarizer as pz


@pytest.fixture(scope="module")
def p():
    return pz.sample(np.random.default_rng(0))


def test_removing_the_polarizer_lets_more_light_out(p):
    t_pol, _ = pz.stack_optics("polarizer (today)", p)
    t_free, _ = pz.stack_optics("polarizer-free", p)
    assert np.all(t_free > t_pol)


def test_removing_the_polarizer_reflects_more_room_light(p):
    _, r_pol = pz.stack_optics("polarizer (today)", p)
    _, r_free = pz.stack_optics("polarizer-free", p)
    assert np.all(r_free > r_pol)


def test_moth_eye_cuts_reflection(p):
    _, r_free = pz.stack_optics("polarizer-free", p)
    _, r_moth = pz.stack_optics("polarizer-free + moth eye", p)
    assert np.all(r_moth < r_free)


def test_saving_in_dim_rooms_matches_samsungs_claim(p):
    w = np.full(pz.N, 0.3)
    base, _ = pz.day("phone", [(2.0, 50)], "polarizer (today)", p, w)
    free, _ = pz.day("phone", [(2.0, 50)], "polarizer-free", p, w)
    saved = np.median(100 * (1 - free / base))
    assert 20 <= saved <= 30          # Samsung reports up to 25% less power


def test_polarizer_free_alone_struggles_in_strong_sun(p):
    w = np.full(pz.N, 0.3)
    _, bad = pz.day("phone", [(1.0, 30000)], "polarizer-free", p, w)
    _, bad_moth = pz.day("phone", [(1.0, 30000)], "polarizer-free + moth eye", p, w)
    assert np.median(bad) > 0 and np.median(bad_moth) == 0


def test_brighter_rooms_need_brighter_screens():
    assert pz.comfort_nits(5) < pz.comfort_nits(400) < pz.comfort_nits(20000)
