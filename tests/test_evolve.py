"""Checks for the evolution-based optimizer."""
import numpy as np
import pytest

import evolve as ev
import unified as u


def test_dominance_rule():
    assert ev.dominates(np.array([2, 2]), np.array([1, 2]))
    assert not ev.dominates(np.array([2, 1]), np.array([1, 2]))
    assert not ev.dominates(np.array([1, 1]), np.array([1, 1]))


def test_front_ranking():
    objs = np.array([[3, 1], [1, 3], [2, 2], [1, 1]])
    rank = ev.fronts(objs)
    assert list(rank[:3]) == [0, 0, 0] and rank[3] == 1


def test_children_stay_inside_gene_limits():
    rng = np.random.default_rng(0)
    lo = np.array([v[0] for v in ev.GENES.values()])
    hi = np.array([v[1] for v in ev.GENES.values()])
    kids = ev.breed(lo + rng.random((20, 4)) * (hi - lo), rng)
    assert np.all(kids >= lo) and np.all(kids <= hi)


def test_designs_with_too_few_syncs_are_pushed_down():
    objs = ev.objectives(np.array([30.0, 30.0]), np.array([10.0, 10.0]), np.array([30.0, 5.0]))
    assert objs[1, 0] < objs[0, 0]


def test_longer_plan_means_longer_battery(monkeypatch):
    monkeypatch.setattr(ev, "DAYS", 12)
    monkeypatch.setattr(ev, "DEVICES", 5)
    monkeypatch.setattr(u, "DAYS", 12)
    monkeypatch.setattr(u, "N", 5)
    lux, sun, reading = u.build_days()
    p = u.sample(np.random.default_rng(1))
    pop = np.array([[300, 5, 10, 60], [300, 50, 10, 60]], float)
    days, comfort, _ = ev.evaluate(pop, lux, sun, reading, p)
    assert days[1] >= days[0] and comfort[0] >= comfort[1]
