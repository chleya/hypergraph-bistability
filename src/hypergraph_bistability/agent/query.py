"""
Query Layer for Hypergraph Agent.

Provides standardized interfaces for querying agent runtime state.
Separates state capture from query presentation for cleaner external APIs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TaskState:
    """Current task state in the working set."""
    linked_task: str
    phase_summary: str  # e.g., "planning", "execution", "review"
    status: str  # e.g., "active", "blocked", "completed"
    blocker_count: int
    next_step_count: int
    active_decisions_count: int
    active_procedures_count: int


@dataclass
class ConflictInfo:
    """Conflict information from the working set."""
    has_conflict: bool
    dominant_hypothesis: Optional[str] = None
    dominant_evidence: Optional[str] = None
    contradicted_hypotheses: List[str] = field(default_factory=list)
    contradiction_edges: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class DecisionResidue:
    """Decision residue in the working set."""
    decision_id: str
    decision_content: str
    linked_task: str
    verification_status: str  # "verified", "pending", "ruled_out"


@dataclass
class ProcedureInfo:
    """Procedure information in the working set."""
    procedure_id: str
    procedure_content: str
    linked_task: str
    phase: str  # e.g., "verify", "implement", "test"


@dataclass
class HandoffBundle:
    """Complete handoff bundle for task transfer."""
    linked_task: str
    dominant_conflict: Optional[ConflictInfo] = None
    active_decisions: List[DecisionResidue] = field(default_factory=list)
    applicable_procedures: List[ProcedureInfo] = field(default_factory=list)
    blockers: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    next_steps: List[str] = field(default_factory=list)
    ready_signals: List[str] = field(default_factory=list)


@dataclass
class WorkingSet:
    """Complete working set state."""
    linked_task: Optional[str]
    active_nodes: List[Dict[str, Any]] = field(default_factory=list)
    active_blockers: List[Dict[str, Any]] = field(default_factory=list)
    next_step_candidates: List[Dict[str, Any]] = field(default_factory=list)
    active_decisions: List[Dict[str, Any]] = field(default_factory=list)
    active_procedures: List[Dict[str, Any]] = field(default_factory=list)
    recent_verifications: List[Dict[str, Any]] = field(default_factory=list)


class QueryLayer:
    """
    Standardized query layer for HypergraphAgent.
    
    Provides clean interfaces for querying runtime state without
    needing to access the underlying agent implementation.
    """
    
    def __init__(self, agent: Any):
        """
        Initialize query layer with an agent instance.
        
        Args:
            agent: HypergraphAgent instance to query
        """
        self._agent = agent
    
    def get_working_set(self, linked_task: Optional[str] = None) -> WorkingSet:
        """
        Get the complete working set for a task.
        
        Args:
            linked_task: Optional task ID to filter by. 
                        If None, returns the dominant task's working set.
        
        Returns:
            WorkingSet object with all active elements
        """
        raw = self._agent.get_working_set(linked_task)
        
        return WorkingSet(
            linked_task=raw.get("linked_task"),
            active_nodes=raw.get("active_nodes", []),
            active_blockers=raw.get("active_blockers", []),
            next_step_candidates=raw.get("next_step_candidates", []),
            active_decisions=raw.get("active_decisions", []),
            active_procedures=raw.get("active_procedures", []),
            recent_verifications=raw.get("recent_verifications", []),
        )
    
    def query_current_task_state(self, linked_task: Optional[str] = None) -> TaskState:
        """
        Query the current task state.
        
        Args:
            linked_task: Optional task ID to query
        
        Returns:
            TaskState with current phase and status
        """
        raw = self._agent.query_current_task_state(linked_task)
        
        return TaskState(
            linked_task=raw.get("linked_task", ""),
            phase_summary=raw.get("phase_summary", "unknown"),
            status=raw.get("status", "unknown"),
            blocker_count=raw.get("blocker_count", 0),
            next_step_count=raw.get("next_step_count", 0),
            active_decisions_count=raw.get("decision_count", 0),
            active_procedures_count=raw.get("procedure_count", 0),
        )
    
    def query_dominant_conflict(self, linked_task: Optional[str] = None) -> Optional[ConflictInfo]:
        """
        Query the dominant conflict for a task.
        
        Args:
            linked_task: Optional task ID to query
        
        Returns:
            ConflictInfo if conflict exists, None otherwise
        """
        raw = self._agent.query_dominant_conflict(linked_task)
        
        if not raw:
            return None
        
        return ConflictInfo(
            has_conflict=raw.get("status") == "active" or bool(raw.get("dominant_content")),
            dominant_hypothesis=raw.get("dominant_content"),
            dominant_evidence=raw.get("backing_hyperedge_id"),
            contradicted_hypotheses=raw.get("contradicted_contents", []),
            contradiction_edges=raw.get("contradiction_edges", []),
        )
    
    def query_decision_residue(self, linked_task: Optional[str] = None) -> List[DecisionResidue]:
        """
        Query decision residues for a task.
        
        Args:
            linked_task: Optional task ID to query
        
        Returns:
            List of DecisionResidue objects
        """
        raw_list = self._agent.query_decision_residue(linked_task)
        
        return [
            DecisionResidue(
                decision_id=raw.get("dominant_decision_node_id", ""),
                decision_content=raw.get("dominant_decision_content", ""),
                linked_task=raw.get("linked_task", ""),
                verification_status=raw.get("status", "pending"),
            )
            for raw in raw_list
        ]
    
    def query_applicable_procedures(self, linked_task: Optional[str] = None) -> List[ProcedureInfo]:
        """
        Query applicable procedures for a task.
        
        Args:
            linked_task: Optional task ID to query
        
        Returns:
            List of ProcedureInfo objects
        """
        raw_list = self._agent.query_applicable_procedures(linked_task)
        
        return [
            ProcedureInfo(
                procedure_id=raw.get("dominant_procedure_node_id", ""),
                procedure_content=raw.get("dominant_procedure_content", ""),
                linked_task=raw.get("linked_task", ""),
                phase=raw.get("procedure_types", ["unknown"])[0] if raw.get("procedure_types") else "unknown",
            )
            for raw in raw_list
        ]
    
    def query_handoff_bundle(self, linked_task: Optional[str] = None) -> HandoffBundle:
        """
        Query complete handoff bundle for task transfer.
        
        Args:
            linked_task: Optional task ID to query
        
        Returns:
            HandoffBundle with all transfer information
        """
        raw = self._agent.query_handoff_bundle(linked_task)
        
        return HandoffBundle(
            linked_task=raw.get("linked_task", ""),
            dominant_conflict=self.query_dominant_conflict(linked_task),
            active_decisions=self.query_decision_residue(linked_task),
            applicable_procedures=self.query_applicable_procedures(linked_task),
            blockers=raw.get("blockers", []),
            evidence=raw.get("evidence", []),
            next_steps=raw.get("next_steps", []),
            ready_signals=raw.get("ready_signals", []),
        )
    
    def query_memory_stats(self) -> Dict[str, Any]:
        """
        Query memory statistics.
        
        Returns:
            Dict with memory stats
        """
        state = self._agent.get_session_state()
        
        return {
            "conversation_turns": len(state.conversation_history) // 2,
            "turn_log_entries": len(state.turn_log),
            "controller_state": state.controller_state,
        }
    
    def query_hypergraph_summary(self) -> Dict[str, Any]:
        """
        Query hypergraph summary.
        
        Returns:
            Dict with hypergraph stats
        """
        view = self._agent.get_hypergraph_view()
        
        return {
            "node_count": len(view.get("nodes", {})),
            "hyperedge_count": len(view.get("hyperedges", {})),
            "edge_count": len(view.get("edges", [])),
        }


def get_query_layer(agent: Any) -> QueryLayer:
    """
    Get a QueryLayer instance for an agent.
    
    Args:
        agent: HypergraphAgent instance
    
    Returns:
        QueryLayer instance
    """
    return QueryLayer(agent)
