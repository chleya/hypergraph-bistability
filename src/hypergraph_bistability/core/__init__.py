"""Stable core scientific APIs."""

from .dynamics import (
    HypergraphDynamics,
    apply_deletion_rule,
    apply_fusion_rule,
    apply_growth_rule,
    apply_split_rule,
    compute_order_parameter,
)
from .model import MultiGroupHypergraph, MultiLayerHypergraph
from .noise import add_gaussian_noise, compute_noise_effect
from .potential import (
    classify_stability,
    compute_barrier_height,
    double_well_potential,
    find_fixed_points,
    multi_well_potential,
    simulate_dynamics,
)

__all__ = [
    "MultiGroupHypergraph",
    "MultiLayerHypergraph",
    "HypergraphDynamics",
    "compute_order_parameter",
    "apply_growth_rule",
    "apply_fusion_rule",
    "apply_split_rule",
    "apply_deletion_rule",
    "add_gaussian_noise",
    "compute_noise_effect",
    "double_well_potential",
    "multi_well_potential",
    "compute_barrier_height",
    "find_fixed_points",
    "classify_stability",
    "simulate_dynamics",
]
