"""Checks for the real-data loader (uses a tiny file in the real PVGIS format)."""
import numpy as np
import pytest

import real_data as rd

SAMPLE = """Latitude (decimal degrees):\t32.690
Longitude (decimal degrees):\t35.420
Elevation (m):\t127
Radiation database:\tPVGIS-SARAH2


Slope: 0 deg.
Azimuth: 0 deg.
time,G(i),H_sun,T2m,WS10m,Int
""" + "\n".join(
    f"20200101:{h:02d}09,{max(0, 500 - 60 * abs(h - 10)):.1f},0.0,15.0,1.0,0.0" for h in range(24)
) + """

G(i): Global irradiance on the inclined plane (plane of the array) (W/m2)
PVGIS (c) European Union, 2001-2024
"""


@pytest.fixture
def pvgis_file(tmp_path):
    f = tmp_path / "pvgis.csv"
    f.write_text(SAMPLE)
    return f


def test_loader_reads_only_the_hourly_rows(pvgis_file):
    times, watts = rd.load_pvgis(pvgis_file)
    assert len(watts) == 24
    assert watts.max() == pytest.approx(500)


def test_loader_rejects_a_file_with_no_data(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("<html>error page</html>")
    with pytest.raises(ValueError):
        rd.load_pvgis(bad)


def test_indoor_light_is_minute_by_minute_and_shifted_to_local_time(pvgis_file):
    times, watts = rd.load_pvgis(pvgis_file)
    lux, sun, days, start = rd.indoor_minutes(times, watts)
    assert len(lux) == days * 1440
    peak_minute = int(np.argmax(np.where(sun, lux, 0)))
    assert peak_minute // 60 == 10 + rd.UTC_OFFSET_H


def test_evening_lamp_turns_on_after_dark(pvgis_file):
    times, watts = rd.load_pvgis(pvgis_file)
    lux, sun, _, _ = rd.indoor_minutes(times, watts)
    h0, h1, led = rd.EVENING_LED
    assert np.all(lux[h0 * 60 + 60:h1 * 60] >= led - 1e-9)


def test_darkness_gives_no_energy():
    lux = np.zeros(1440); sun = np.zeros(1440, dtype=bool)
    assert rd.daily_energy_per_m2(lux, sun, 1)[0] == 0
