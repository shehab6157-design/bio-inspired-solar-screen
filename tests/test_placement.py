"""Checks for the placement sweep's winter light builder (uses a tiny mock year)."""
import numpy as np
import pandas as pd
import pytest

import placement as pl
import real_data as rd


@pytest.fixture
def mock_year(monkeypatch):
    """A fake PVGIS year: 500 W/m^2 at local noon every day, dark otherwise."""
    times = pd.date_range("2020-01-01", "2020-12-31 23:00", freq="h")
    watts = np.where(times.hour == 10, 500.0, 0.0)     # 10:00 UTC = 12:00 local
    monkeypatch.setattr(rd, "load_pvgis", lambda path=None: (times, watts))


def test_winter_is_november_to_february(mock_year):
    lux, sun, days = pl.winter_light(0.03)
    assert days == 30 + 31 + 31 + 29                    # Nov + Dec + Jan + Feb 2020
    assert len(lux) == days * 1440


def test_deeper_in_the_room_means_less_daylight(mock_year):
    near, _, _ = pl.winter_light(0.03, lamp=False)
    far, _, _ = pl.winter_light(0.005, lamp=False)
    assert far.sum() == pytest.approx(near.sum() / 6)


def test_no_lamp_means_dark_evenings(mock_year):
    lux, _, _ = pl.winter_light(0.03, lamp=False)
    evening = lux[18 * 60:21 * 60]                       # first day, 18:00-21:00
    assert np.all(evening == 0)


def test_lamp_setting_is_restored_afterwards(mock_year):
    before = rd.EVENING_LED
    pl.winter_light(0.01, lamp=False)
    assert rd.EVENING_LED == before
