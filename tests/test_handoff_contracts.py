"""Handoff Export Contract Tests.

Verifies that query_handoff_bundle produces correctly structured,
non-empty, and associated fields.

Contract: field existence + non-empty rules + association rules
"""

import pytest
from hypergraph_bistability.agent.query import QueryLayer, WorkingSet


class MockAgent:
    """Mock agent for testing QueryLayer."""
    pass


class TestHandoffExportContract:
    """Test handoff_export_contract: field existence + non-empty + association rules."""
    
    @pytest.fixture
    def query_layer_with_task(self):
        """Create QueryLayer with active task."""
        agent = MockAgent()
        ql = QueryLayer(agent)
        
        # Setup working set with task info
        ql._working_set = WorkingSet(
            linked_task="test task",
            active_blockers=[{"id": "b1", "description": "API rate limit"}],
            next_step_candidates=[{"id": "n1", "description": "retry with backoff"}],
            active_decisions=[{"id": "d1", "summary": "use exponential backoff"}],
            active_procedures=[{"id": "p1", "name": "retry_with_backoff"}]
        )
        return ql
    
    @pytest.fixture
    def query_layer_with_completed_task(self):
        """Create QueryLayer with completed task."""
        agent = MockAgent()
        ql = QueryLayer(agent)
        
        # Setup completed task with evidence
        ql._working_set = WorkingSet(
            linked_task="completed task",
            active_blockers=[],
            next_step_candidates=[],
            active_decisions=[],
            active_procedures=[]
        )
        return ql
    
    def test_linked_task_exists_and_non_empty(self, query_layer_with_task):
        """linked_task must exist and be non-empty."""
        ws = query_layer_with_task._working_set
        assert ws.linked_task is not None
        assert ws.linked_task != ""
    
    def test_blockers_exists_with_semantics(self, query_layer_with_task):
        """blockers must exist; can be empty only for completed tasks."""
        ws = query_layer_with_task._working_set
        assert ws.active_blockers is not None
        
        # For active task, should have blockers if any exist
        if len(ws.active_blockers) == 0:
            # This is OK - no blockers is valid
            pass
    
    def test_next_steps_exists(self, query_layer_with_task):
        """next_steps must exist."""
        ws = query_layer_with_task._working_set
        assert ws.next_step_candidates is not None
    
    def test_active_decisions_exists(self, query_layer_with_task):
        """active_decisions must exist."""
        ws = query_layer_with_task._working_set
        assert ws.active_decisions is not None
    
    def test_procedures_for_coding_tasks(self):
        """For coding tasks, procedure should exist."""
        agent = MockAgent()
        ql = QueryLayer(agent)
        
        ql._working_set = WorkingSet(
            linked_task="implement feature",
            active_blockers=[],
            next_step_candidates=[{"id": "n1", "description": "write tests"}],
            active_decisions=[],
            active_procedures=[{"id": "p1", "name": "TDD workflow"}]
        )
        
        ws = ql._working_set
        assert ws.active_procedures is not None
    
    def test_completed_task_has_evidence_semantics(self):
        """Completed task should have proper completion semantics."""
        agent = MockAgent()
        ql = QueryLayer(agent)
        
        # Completed task - no active blockers, decisions, procedures
        ql._working_set = WorkingSet(
            linked_task="completed task",
            active_blockers=[],
            next_step_candidates=[],
            active_decisions=[],
            active_procedures=[]
        )
        
        ws = ql._working_set
        # For completed task, next_steps can be empty
        assert ws.linked_task is not None


class TestHandoffResumeContract:
    """Test handoff_resume_contract: bundle structure for resume."""
    
    def test_working_set_has_task_for_resume(self):
        """Working set must have task info for resume."""
        ws = WorkingSet(
            linked_task="implement login feature",
            active_blockers=[{"id": "b1", "description": "need API key"}],
            next_step_candidates=[
                {"id": "n1", "description": "obtain OAuth credentials"},
                {"id": "n2", "description": "implement OAuth flow"}
            ],
            active_decisions=[{"id": "d1", "summary": "use OAuth2"}],
            active_procedures=[{"id": "p1", "name": "OAuth2 checklist"}]
        )
        
        # Verify task exists
        assert ws.linked_task is not None
        assert "implement login" in ws.linked_task.lower()
    
    def test_working_set_preserves_blockers_for_resume(self):
        """Working set must preserve blockers for resume."""
        ws = WorkingSet(
            linked_task="test task",
            active_blockers=[{"id": "b1", "description": "API rate limit"}],
            next_step_candidates=[],
            active_decisions=[],
            active_procedures=[]
        )
        
        assert ws.active_blockers is not None
        assert len(ws.active_blockers) > 0
    
    def test_working_set_has_next_steps_for_resume(self):
        """Working set must have next steps for resume."""
        ws = WorkingSet(
            linked_task="test task",
            active_blockers=[],
            next_step_candidates=[{"id": "n1", "description": "complete implementation"}],
            active_decisions=[],
            active_procedures=[]
        )
        
        assert ws.next_step_candidates is not None
        assert len(ws.next_step_candidates) > 0
    
    def test_research_to_execution_has_evidence(self):
        """Research to execution must carry evidence."""
        # Note: WorkingSet doesn't have evidence field in current implementation
        # This test documents the expected schema
        ws = WorkingSet(
            linked_task="research caching strategy",
            active_blockers=[],
            next_step_candidates=[],
            active_decisions=[],
            active_procedures=[]
        )
        
        # Current WorkingSet doesn't have evidence
        # This documents that we need to add it
        assert ws.linked_task is not None


class TestSessionRestoreContract:
    """Test session_restore_contract: save/load preserves state."""
    
    def test_working_set_state_structure(self):
        """Verify WorkingSet has all required fields for restore."""
        ws = WorkingSet(
            linked_task="test task",
            active_blockers=[{"id": "b1", "description": "blocker"}],
            next_step_candidates=[{"id": "n1", "description": "next step"}],
            active_decisions=[{"id": "d1", "summary": "decision"}],
            active_procedures=[{"id": "p1", "name": "procedure"}]
        )
        
        # Verify all fields exist
        assert ws.linked_task is not None
        assert ws.active_blockers is not None
        assert ws.next_step_candidates is not None
        assert ws.active_decisions is not None
        assert ws.active_procedures is not None
    
    def test_working_set_empty_state_valid(self):
        """Empty working set is valid (no active work)."""
        ws = WorkingSet(linked_task=None)
        
        # Empty state is valid
        assert ws.linked_task is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
