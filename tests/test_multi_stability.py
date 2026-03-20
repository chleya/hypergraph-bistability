"""
Tests for multi_stability module
"""
import numpy as np
import sys
sys.path.insert(0, 'F:/hypergraph_bistability/src')

from multi_stability.multi_stability_core import (
    MultiGroupHypergraph,
    MultiLayerMultiGroupHypergraph,
    count_stable_states
)


def test_multi_group_hypergraph_init():
    N, n_groups = 30, 3
    hg = MultiGroupHypergraph(N=N, n_groups=n_groups, seed=42)
    assert hg.N == N
    assert hg.n_groups == n_groups
    assert len(hg.V) == N


def test_multi_group_node_assignment():
    N, n_groups = 30, 3
    hg = MultiGroupHypergraph(N=N, n_groups=n_groups, seed=42)
    for v in hg.V:
        g = hg.get_node_group(v)
        assert 0 <= g < n_groups


def test_get_group_nodes():
    N, n_groups = 30, 3
    hg = MultiGroupHypergraph(N=N, n_groups=n_groups, seed=42)
    for g in range(n_groups):
        nodes = hg.get_group_nodes(g)
        assert len(nodes) > 0
        for v in nodes:
            assert hg.get_node_group(v) == g


def test_order_parameter():
    N, n_groups = 30, 3
    hg = MultiGroupHypergraph(N=N, n_groups=n_groups, seed=42)
    for g in range(n_groups):
        M_g = hg.get_group_order_parameter(g)
        assert M_g >= 0


def test_run_dynamics():
    N, n_groups = 30, 3
    hg = MultiGroupHypergraph(N=N, n_groups=n_groups, seed=42)
    history = hg.run_dynamics(steps=10)
    assert len(history) == n_groups
    for g in range(n_groups):
        assert len(history[g]) == 10


def test_get_stable_states():
    N, n_groups = 30, 3
    hg = MultiGroupHypergraph(N=N, n_groups=n_groups, seed=42)
    hg.run_dynamics(steps=20)
    states = hg.get_stable_states()
    assert len(states) == n_groups


def test_count_stable_states():
    M_dict = {0: 0.3, 1: 0.5, 2: 0.3}
    count, unique = count_stable_states(M_dict, tolerance=0.1)
    assert count >= 1
    assert len(unique) == count


def test_multi_layer_init():
    ml_hg = MultiLayerMultiGroupHypergraph(N=30, n_groups=3, n_layers=2, seed=42)
    assert ml_hg.n_layers == 2
    assert len(ml_hg.layers) == 2


def test_multi_layer_dynamics():
    ml_hg = MultiLayerMultiGroupHypergraph(N=30, n_groups=3, n_layers=2, seed=42)
    history = ml_hg.run_dynamics(steps=10)
    assert len(history) == 2
