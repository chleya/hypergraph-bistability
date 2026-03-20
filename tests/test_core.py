"""
Tests for the new core module
"""
import numpy as np
import sys
sys.path.insert(0, 'F:/hypergraph_bistability/src')

from core import (
    MultiGroupHypergraph,
    MultiLayerHypergraph,
    HypergraphDynamics,
    double_well_potential,
    multi_well_potential,
    compute_barrier_height,
    find_fixed_points,
    classify_stability,
    simulate_dynamics,
    add_gaussian_noise,
    compute_noise_effect,
)


def test_multi_group_hypergraph_core():
    hg = MultiGroupHypergraph(N=30, n_groups=3, seed=42)
    assert hg.N == 30
    assert hg.n_groups == 3


def test_multi_layer_hypergraph_core():
    ml = MultiLayerHypergraph(N=30, n_groups=3, n_layers=2, seed=42)
    assert ml.n_layers == 2
    assert len(ml.layers) == 2


def test_hypergraph_dynamics_init():
    dyn = HypergraphDynamics(L=2, k=3, Kc_list=[0.32, 0.40, 0.48])
    assert dyn.L == 2
    assert dyn.k == 3
    assert len(dyn.a) == 3


def test_hypergraph_dynamics_compute_dM():
    dyn = HypergraphDynamics(L=2, k=3)
    M = np.random.uniform(0.1, 0.9, (2, 3))
    dM = dyn.compute_dM(M, lam=0.1, mu=0.0)
    assert dM.shape == (6,)


def test_double_well_potential():
    V = double_well_potential(0.5, a=1.0, b=0.15)
    assert isinstance(V, (float, np.floating))


def test_multi_well_potential():
    wells = [(0.2, -1.0), (0.5, -0.5), (0.8, -1.0)]
    V = multi_well_potential(0.5, wells)
    assert isinstance(V, (float, np.floating))


def test_compute_barrier_height():
    def V(x):
        return double_well_potential(x, a=1.0, b=0.15)
    barrier = compute_barrier_height(0.2, 0.8, V)
    assert barrier >= 0


def test_find_fixed_points_core():
    roots = find_fixed_points(a=1.0, b=0.15)
    assert isinstance(roots, list)


def test_classify_stability():
    roots = find_fixed_points(a=1.0, b=0.15)
    if len(roots) >= 2:
        classified = classify_stability(roots, a=1.0, b=0.15)
        assert all(s in ['stable', 'unstable'] for _, s in classified)


def test_simulate_dynamics():
    traj = simulate_dynamics(x0=0.3, a=1.0, b=0.15, steps=50)
    assert len(traj) == 51
    assert all(0 <= x <= 1 for x in traj)


def test_add_gaussian_noise():
    m = np.random.uniform(0.1, 0.9, 10)
    m_noisy = add_gaussian_noise(m, sigma=0.01)
    assert len(m_noisy) == len(m)


def test_compute_noise_effect():
    states = np.random.uniform(0.1, 0.9, (5, 10))
    n = compute_noise_effect(states, threshold=0.1)
    assert n >= 1
