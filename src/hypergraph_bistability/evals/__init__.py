"""Evaluation helpers for practical agent progress."""

from hypergraph_bistability.evals.config import (
    DEFAULT_LLM_EVAL_BASE_URL,
    DEFAULT_LLM_EVAL_MODEL,
    DEFAULT_LLM_EVAL_TEMPERATURE,
)
from hypergraph_bistability.evals.runner import run_eval_suite
from hypergraph_bistability.evals.runner import (
    CONFLICT_PRACTICAL_SCENARIO_NAMES,
    CONFLICT_SIDECAR_SCENARIO_NAMES,
    CONTINUITY_SCENARIO_NAMES,
    LONG_TASK_SCENARIO_NAMES,
    PRACTICAL_ROBUSTNESS_SCENARIO_NAMES,
    PRACTICAL_SIDECAR_SCENARIO_NAMES,
    PRODUCT_SCENARIO_NAMES,
    run_conflict_regression,
    run_conflict_sidecar_regression,
    run_continuity_regression,
    run_long_task_regression,
    run_practical_robustness_regression,
    run_practical_sidecar_regression,
    run_product_regression,
)
from hypergraph_bistability.evals.scenarios import DEFAULT_EVAL_SCENARIOS

__all__ = [
    "CONTINUITY_SCENARIO_NAMES",
    "CONFLICT_PRACTICAL_SCENARIO_NAMES",
    "CONFLICT_SIDECAR_SCENARIO_NAMES",
    "LONG_TASK_SCENARIO_NAMES",
    "PRACTICAL_ROBUSTNESS_SCENARIO_NAMES",
    "PRACTICAL_SIDECAR_SCENARIO_NAMES",
    "PRODUCT_SCENARIO_NAMES",
    "DEFAULT_EVAL_SCENARIOS",
    "DEFAULT_LLM_EVAL_BASE_URL",
    "DEFAULT_LLM_EVAL_MODEL",
    "DEFAULT_LLM_EVAL_TEMPERATURE",
    "run_conflict_regression",
    "run_conflict_sidecar_regression",
    "run_continuity_regression",
    "run_long_task_regression",
    "run_practical_robustness_regression",
    "run_practical_sidecar_regression",
    "run_product_regression",
    "run_eval_suite",
]
