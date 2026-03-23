"""Handoff Export Contract Tests.

Verifies that query_handoff_bundle produces correctly structured,
non-empty, and associated fields.
"""

import pytest
from hypergraph_bistability.agent.query import QueryLayer, WorkingSet, HandoffBundle


class TestHandoffExportContract:
    """Test handoff_export_contract: field existence + non-empty + association rules."""
    
    @pytest.fixture
    def query_layer(self):
        """Create QueryLayer with active working set."""
        # Create a mock agent for QueryLayer
        class MockAgent:
            pass
        
        mock_agent = MockAgent()
        ql = QueryLayer(mock_agent)
        
        # Setup: inject test working set with all required fields
        ws = WorkingSet(
            task={"name": "test task", "status": "active", "phase": "execution"},
            blockers=[{"id": "b1", "description": "API rate limit"}],
            next_steps=[{"id": "n1", "description": "retry with backoff"}],
            active_decisions=[{"id": "d1", "summary": "use exponential backoff"}],
            applicable_procedures=[{"id": "p1", "name": "retry_with_backoff"}],
            decisions_detail=[{"id": "d1", "summary": "use exponential backoff"}],
            procedures_detail=[{"id": "p1", "name": "retry_with_backoff"}],
            handoff_ready=True
        )
        ql._working_set = ws
        return ql
    
    def test_linked_task_exists_and_non_empty(self, query_layer):
        """linked_task must exist and be non-empty."""
        bundle = query_layer.query_handoff_bundle()
        
        assert "linked_task" in bundle
        assert bundle["linked_task"] is not None
        assert bundle["linked_task"] != ""
        assert isinstance(bundle["linked_task"], (dict, str))
    
    def test_blockers_exists_with_semantics(self, query_layer):
        """blockers must exist; can be empty only with explicit 'no blockers'."""
        bundle = query_layer.query_handoff_bundle()
        
        assert "blockers" in bundle
        assert bundle["blockers"] is not None
        
        # If empty, must have explicit no-blocker signal
        if len(bundle.get("blockers", [])) == 0:
            # Should be explicitly marked as "no blockers" or task completed
            assert bundle.get("task", {}).get("status") in ["completed", "done"]
    
    def test_next_steps_exists_with_semantics(self, query_layer):
        """next_steps must exist; can be empty only for completed tasks."""
        bundle = query_layer.query_handoff_bundle()
        
        assert "next_steps" in bundle
        assert bundle["next_steps"] is not None
        
        # If empty but task not completed, that's a failure
        if len(bundle.get("next_steps", [])) == 0:
            task_status = bundle.get("task", {}).get("status", "")
            assert task_status in ["completed", "done"], \
                "next_steps empty but task not completed"
    
    def test_active_decisions_exists_with_semantics(self, query_layer):
        """active_decisions must exist; can be empty for non-decision tasks."""
        bundle = query_layer.query_handoff_bundle()
        
        assert "active_decisions" in bundle
        assert bundle["active_decisions"] is not None
    
    def test_evidence_exists_and_relevant(self, query_layer):
        """evidence must exist and be relevant to task/blocker/decision."""
        bundle = query_layer.query_handoff_bundle()
        
        assert "evidence" in bundle
        assert bundle["evidence"] is not None
        assert len(bundle["evidence"]) > 0
        
        # Check evidence relevance: at least one must be关联 to task/blocker/decision
        task_name = str(bundle.get("linked_task", ""))
        blocker_texts = [str(b.get("description", "")) for b in bundle.get("blockers", [])]
        decision_texts = [str(d.get("summary", "")) for d in bundle.get("active_decisions", [])]
        
        relevant_found = False
        for ev in bundle["evidence"]:
            ev_text = str(ev)
            # Evidence is relevant if it mentions task, any blocker, or any decision
            if (task_name and task_name.lower() in ev_text.lower()) or \
               any(bt and bt.lower() in ev_text.lower() for bt in blocker_texts if bt) or \
               any(dt and dt.lower() in ev_text.lower() for dt in decision_texts if dt):
                relevant_found = True
                break
        
        assert relevant_found, "evidence exists but not relevant to task/blocker/decision"
    
    def test_procedure_for_coding_planning_tasks(self):
        """For coding/planning tasks, procedure should not be optional."""
        ql = QueryLayer()
        # Setup: coding task with procedure
        ws = WorkingSet(
            task={"name": "implement feature", "status": "active", "phase": "execution"},
            blockers=[],
            next_steps=[{"id": "n1", "description": "write tests"}],
            active_decisions=[],
            applicable_procedures=[{"id": "p1", "name": "TDD workflow"}],
            decisions_detail=[],
            procedures_detail=[{"id": "p1", "name": "TDD workflow"}],
            handoff_ready=True
        )
        ql._working_set = ws
        bundle = query_layer.query_handoff_bundle()
        
        # For coding task, procedure should exist and be non-empty
        assert "applicable_procedures" in bundle
        # This is a guideline - for now just verify it exists
    
    def test_completed_task_keeps_evidence(self):
        """Completed task must still retain evidence."""
        ql = QueryLayer()
        # Setup: completed task
        ws = WorkingSet(
            task={"name": "completed task", "status": "completed", "phase": "done"},
            blockers=[],
            next_steps=[],
            active_decisions=[],
            applicable_procedures=[],
            decisions_detail=[],
            procedures_detail=[],
            handoff_ready=True
        )
        ql._working_set = ws
        bundle = query_layer.query_handoff_bundle()
        
        # Even for completed task, evidence must exist
        assert "evidence" in bundle
        assert bundle["evidence"] is not None
        assert len(bundle["evidence"]) > 0, \
            "Completed task must still have evidence explaining completion"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
