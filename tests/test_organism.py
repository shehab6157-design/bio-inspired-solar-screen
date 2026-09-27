"""Checks for the organism (all layers together)."""
import organism as og


def test_phone_layers_only_ever_save_energy():
    d = og.phone_ledger()
    totals = list(d["total_J_day"])
    assert totals == sorted(totals, reverse=True)
    assert d["saved_%"].iloc[0] == 0


def test_hive_is_what_cuts_phone_wakeups():
    d = og.phone_ledger().set_index("layer")
    assert d.loc["+ hive", "phone_wakes_day"] < d.loc["+ moon mode", "phone_wakes_day"]


def test_tv_layers_save_energy():
    d = og.tv_ledger()
    assert d["kWh_per_year"].is_monotonic_decreasing


def test_home_screens_report_never_crashes():
    assert len(og.home_screens()) >= 1
