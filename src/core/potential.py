"""Compatibility wrapper for the canonical core potential module."""

from hypergraph_bistability.core.potential import (
    classify_stability,
    compute_barrier_height,
    double_well_potential,
    find_fixed_points,
    multi_well_potential,
    simulate_dynamics,
)

__all__ = [
    "double_well_potential",
    "multi_well_potential",
    "compute_barrier_height",
    "find_fixed_points",
    "classify_stability",
    "simulate_dynamics",
]
