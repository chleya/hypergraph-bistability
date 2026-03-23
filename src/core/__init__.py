"""Deprecated compatibility layer.

This module is DEPRECATED. Please use hypergraph_bistability.core instead.

This compatibility layer will be removed in a future version.
Please update your imports:

OLD (deprecated):
    from core import MultiGroupHypergraph
    from core.dynamics import HypergraphDynamics

NEW:
    from hypergraph_bistability.core import MultiGroupHypergraph
    from hypergraph_bistability.core.dynamics import HypergraphDynamics
"""

import warnings

warnings.warn(
    "The 'core' module is deprecated. "
    "Please use 'hypergraph_bistability.core' instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Thin compatibility layer - redirect to new location
from hypergraph_bistability.core import (
    MultiGroupHypergraph,
    MultiLayerHypergraph,
    HypergraphDynamics,
    compute_order_parameter,
    apply_growth_rule,
    apply_fusion_rule,
    apply_split_rule,
    apply_deletion_rule,
    add_gaussian_noise,
    compute_noise_effect,
    double_well_potential,
    multi_well_potential,
    compute_barrier_height,
    find_fixed_points,
    classify_stability,
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
