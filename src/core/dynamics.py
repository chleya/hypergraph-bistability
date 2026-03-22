"""Compatibility wrapper for the canonical core dynamics module."""

from hypergraph_bistability.core.dynamics import (
    HypergraphDynamics,
    apply_deletion_rule,
    apply_fusion_rule,
    apply_growth_rule,
    apply_split_rule,
    compute_order_parameter,
)

__all__ = [
    "HypergraphDynamics",
    "compute_order_parameter",
    "apply_growth_rule",
    "apply_fusion_rule",
    "apply_split_rule",
    "apply_deletion_rule",
]
