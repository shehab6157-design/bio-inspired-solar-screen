"""Checks that the climate test really switches cities."""
import real_data as rd
import climates as cl

ROW = "20200101:{h:02d}09,{g:.1f},0.0,15.0,1.0,0.0"


def write(tmp_path, name, peak):
    rows = [ROW.format(h=h, g=max(0.0, peak - 60 * abs(h - 10))) for h in range(24)]
    f = tmp_path / name
    f.write_text("time,G(i),H_sun,T2m,WS10m,Int\n" + "\n".join(rows) + "\n")
    return f


def test_switching_city_changes_the_data(tmp_path, monkeypatch):
    monkeypatch.setattr(cl, "ROOT", tmp_path.parent)
    data_dir = tmp_path.parent / "data"
    data_dir.mkdir(exist_ok=True)
    write(data_dir, "bright.csv", 500)
    write(data_dir, "dark.csv", 100)
    try:
        cl.use_location("bright.csv", 2)
        _, bright = rd.load_pvgis()
        cl.use_location("dark.csv", 1)
        _, dark = rd.load_pvgis()
        assert bright.max() == 500 and dark.max() == 100
        assert rd.UTC_OFFSET_H == 1
    finally:
        rd.load_pvgis = cl.ORIGINAL_LOADER
        rd.UTC_OFFSET_H = 2
