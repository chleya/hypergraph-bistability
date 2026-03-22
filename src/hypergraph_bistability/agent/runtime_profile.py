"""Named runtime profiles for the practical agent stack."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict


STABLE_RUNTIME_PROFILE_V1: Dict[str, Any] = {
    "name": "stable_v1",
    "retrieval_strategy": "hyperedge_expansion",
    "controller_mode": "adaptive",
    "write_policy": "WritePolicy",
    "retrieval_policy": "RetrievalPolicy",
    "enabled_structures": [
        "competition",
        "hyperedge_expansion",
        "confidence_tags",
        "contradiction_linkage",
        "conflict_hyperedges",
        "decision_residues",
        "task_phase",
        "procedural_memory_v1",
    ],
    "response_contracts": [
        "root_cause_literal_prefix",
        "diff_style_literal_prefix",
        "concise_preference_prefix",
        "decision_handoff_completion",
        "verification_handoff_completion",
        "conflict_backing_evidence_completion",
    ],
    "evaluation_gates": {
        "continuity_regression": "_continuity_regression_v3.json",
        "product_regression": "_product_regression_v10.json",
    },
}


def get_runtime_profile(profile_name: str = "stable_v1") -> Dict[str, Any]:
    """Return a copy of the named runtime profile."""
    if profile_name != "stable_v1":
        raise ValueError(f"Unknown runtime profile: {profile_name}")
    return deepcopy(STABLE_RUNTIME_PROFILE_V1)
