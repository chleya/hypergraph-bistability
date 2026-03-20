"""
Tests for minimal_model module
"""
import numpy as np
import sys
sys.path.insert(0, 'F:/hypergraph_bistability/src')

from minimal_model import feasibility_function, find_fixed_points, stability, model_simulation


def test_feasibility_function():
    x = 0.5
    a, b = 1.0, 0.1
    result = feasibility_function(x, a, b)
    assert isinstance(result, (int, float, np.floating))


def test_find_fixed_points():
    a, b = 1.0, 0.15
    roots = find_fixed_points(a, b)
    assert isinstance(roots, list)
    assert len(roots) <= 2
    for root in roots:
        assert 0 <= root <= 1


def test_find_fixed_points_no_solution():
    a, b = 1.0, 0.3
    roots = find_fixed_points(a, b)
    assert roots == []


def test_stability():
    x, a, b = 0.5, 1.0, 0.1
    dFdx = stability(x, a, b)
    assert isinstance(dFdx, (int, float, np.floating))


def test_model_simulation():
    a, b, x0 = 1.0, 0.15, 0.3
    trajectory = model_simulation(a, b, x0, steps=10)
    assert len(trajectory) == 11
    assert all(0 <= x <= 1 for x in trajectory)


def test_model_simulation_convergence():
    a, b = 1.0, 0.15
    trajectory = model_simulation(a, b, x0=0.5, steps=100)
    final = trajectory[-1]
    assert 0 <= final <= 1


def test_bistability_region():
    a = 1.0
    for b in [0.10, 0.15, 0.20]:
        roots = find_fixed_points(a, b)
        if len(roots) == 2:
            stabilities = [stability(r, a, b) for r in roots]
            stable_count = sum(1 for s in stabilities if s < 0)
            assert stable_count >= 1
