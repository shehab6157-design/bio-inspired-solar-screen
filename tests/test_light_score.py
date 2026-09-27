"""Checks for the Light-Life Score."""
import pytest

import light_score as ls


@pytest.fixture(scope="module")
def results():
    return {n: ls.score(n) for n in ls.SCREENS}


def test_grades_follow_the_bands():
    assert ls.grade(0.04) == "A" and ls.grade(0.3) == "D" and ls.grade(5.0) == "G"


def test_big_screens_are_graded_per_area():
    assert ls.grade(10.0, cm2=8300) == ls.grade(10.0 * 100 / 8300)


def test_light_diets_are_realistic_days():
    assert sum(h for h, _, _ in ls.DIETS["handheld"]) == pytest.approx(6.0)
    assert sum(h for h, _, _ in ls.DIETS["tv"]) == pytest.approx(5.0)


def test_unreadable_screens_are_flagged(results):
    r = results["phone, polarizer-free"]
    assert r["readable_%"] < ls.READABLE_BAR and r["grade"].endswith("!")


def test_bio_stack_beats_todays_phone(results):
    assert results["phone, full bio-stack (+firefly, moon mode)"]["Wh_per_readable_hour"] < \
           results["phone today (polarizer OLED)"]["Wh_per_readable_hour"]
    assert results["phone, full bio-stack (+firefly, moon mode)"]["readable_%"] == 100


def test_each_tv_layer_improves_the_score(results):
    tv = [results[n]["Wh_per_readable_hour"] for n in ls.SCREENS if n.startswith("TV")]
    assert tv == sorted(tv, reverse=True)
