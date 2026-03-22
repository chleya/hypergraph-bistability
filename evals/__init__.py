"""Evaluation framework for hypergraph agent memory."""

from .metrics import (
    Metric,
    TaskContinuation,
    BlockerPreservation,
    DecisionContinuity,
    ProcedureContinuity,
    ConflictContinuity,
    RepeatedWorkAvoidance,
    MemoryRecallPrecision,
    MemoryRecallUsefulness,
    IrrelevantRecallRate,
    TokenUsage,
    Latency,
)
from .scenarios import Scenario, PreferenceRecallScenario, TaskContinuityScenario, ContextSwitchScenario
from .evaluator import Evaluator, EvalResult

__all__ = [
    "Metric",
    "TaskContinuation",
    "BlockerPreservation", 
    "DecisionContinuity",
    "ProcedureContinuity",
    "ConflictContinuity",
    "RepeatedWorkAvoidance",
    "MemoryRecallPrecision",
    "MemoryRecallUsefulness",
    "IrrelevantRecallRate",
    "TokenUsage",
    "Latency",
    "Scenario",
    "PreferenceRecallScenario",
    "TaskContinuityScenario", 
    "ContextSwitchScenario",
    "Evaluator",
    "EvalResult",
]
