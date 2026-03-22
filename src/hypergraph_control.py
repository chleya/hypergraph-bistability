"""Compatibility wrapper for the canonical critical coupling helpers."""

from hypergraph_bistability.control.critical import (
    compute_lambda_c,
    get_all_lambda_c,
    power_law_approximation,
)

__all__ = ["compute_lambda_c", "get_all_lambda_c", "power_law_approximation"]
