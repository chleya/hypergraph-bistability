"""Deprecated compatibility layer.

This module is DEPRECATED. Please use hypergraph_bistability.agent instead.

This compatibility layer will be removed in a future version.
Please update your imports:

OLD (deprecated):
    from agent import HypergraphAgent
    from agent.hypergraph_agent import HypergraphAgent

NEW:
    from hypergraph_bistability import HypergraphAgent
    from hypergraph_bistability.agent import HypergraphAgent
"""

import warnings

warnings.warn(
    "The 'agent' module is deprecated. "
    "Please use 'hypergraph_bistability.agent' instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Thin compatibility layer - redirect to new location
from hypergraph_bistability.agent import (
    AdaptiveController,
    CognitiveMode,
    HypergraphAgent,
    HypergraphMemoryAgent,
    QueryLayer,
    WorkingSet,
    TaskState,
    ConflictInfo,
    DecisionResidue,
    ProcedureInfo,
    HandoffBundle,
    SessionState,
    STABLE_RUNTIME_PROFILE_V1,
    get_runtime_profile,
    get_query_layer,
)

__all__ = [
    "AdaptiveController",
    "CognitiveMode",
    "HypergraphAgent",
    "HypergraphMemoryAgent",
    "QueryLayer",
    "WorkingSet",
    "TaskState",
    "ConflictInfo",
    "DecisionResidue",
    "ProcedureInfo",
    "HandoffBundle",
    "SessionState",
    "STABLE_RUNTIME_PROFILE_V1",
    "get_runtime_profile",
    "get_query_layer",
]
