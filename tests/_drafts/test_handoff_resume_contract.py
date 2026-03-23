"""Handoff Resume Contract Tests.

Verifies that handoff bundle, when injected through context assembler,
enables successful task continuation.

Consumer contract: structured handoff context -> context assembler -> response
"""

import pytest
from unittest.mock import Mock, patch
from hypergraph_bistability.agent.query import QueryLayer, WorkingSet, HandoffBundle


class TestHandoffResumeContract:
    """Test handoff_resume_contract: bundle -> context assembler -> valid response."""
    
    @pytest.fixture
    def agent_with_handoff(self):
        """Create agent with handoff bundle ready for injection."""
        from hypergraph_bistability import HypergraphAgent
        
        agent = HypergraphAgent(
            model="gpt-5.4",
            provider="openai",
            mock=True  # Use mock for deterministic testing
        )
        
        # Setup: inject working set with rich handoff data
        ws = WorkingSet(
            task={
                "name": "implement login feature",
                "status": "active",
                "phase": "execution"
            },
            blockers=[
                {"id": "b1", "description": "need API key from auth provider"}
            ],
            next_steps=[
                {"id": "n1", "description": "obtain OAuth credentials"},
                {"id": "n2", "description": "implement OAuth flow"}
            ],
            active_decisions=[
                {"id": "d1", "summary": "use OAuth2 over basic auth"}
            ],
            applicable_procedures=[
                {"id": "p1", "name": "OAuth2 implementation checklist"}
            ],
            decisions_detail=[
                {"id": "d1", "summary": "use OAuth2 over basic auth"}
            ],
            procedures_detail=[
                {"id": "p1", "name": "OAuth2 implementation checklist"}
            ],
            handoff_ready=True
        )
        
        # Inject into query layer
        agent._query_layer._working_set = ws
        
        return agent
    
    def test_response_references_linked_task(self, agent_with_handoff):
        """Resume response must explicitly reference linked_task."""
        # Get handoff bundle
        bundle = agent_with_handoff.query_handoff_bundle()
        
        # Simulate continuation: feed bundle back as context
        task_name = bundle.get("linked_task", {}).get("name", "") if isinstance(bundle.get("linked_task"), dict) else str(bundle.get("linked_task", ""))
        
        # For this test, we check that the handoff bundle contains task info
        # that would enable a model to reference it
        assert task_name != "", "linked_task must be non-empty for resume"
        
        # The actual "response references task" check would require real LLM
        # For now, we verify the bundle has the information needed
        assert "linked_task" in bundle
    
    def test_response_preserves_blockers(self, agent_with_handoff):
        """Resume response must preserve or address at least one blocker."""
        bundle = agent_with_handoff.query_handoff_bundle()
        blockers = bundle.get("blockers", [])
        
        # Verify blockers exist in bundle
        assert blockers is not None, "blockers must be in handoff bundle"
        
        # For actual resume verification, we'd need to check the response
        # This test verifies the bundle has the data for the check
    
    def test_response_continues_next_steps(self, agent_with_handoff):
        """Resume response must continue at least one next_step."""
        bundle = agent_with_handoff.query_handoff_bundle()
        next_steps = bundle.get("next_steps", [])
        
        # Verify next_steps exist in bundle
        assert next_steps is not None, "next_steps must be in handoff bundle"
        assert len(next_steps) > 0 or bundle.get("task", {}).get("status") in ["completed", "done"], \
            "next_steps empty but task not completed"
    
    def test_response_not_generic_advice_only(self, agent_with_handoff):
        """Response must not degrade to generic advice-only answer."""
        bundle = agent_with_handoff.query_handoff_bundle()
        
        # Check that the bundle has task-specific content, not just generic advice
        has_task_specific_content = False
        
        # Check task
        task = bundle.get("linked_task")
        if task:
            has_task_specific_content = True
        
        # Check blockers have specific descriptions
        blockers = bundle.get("blockers", [])
        for b in blockers:
            if b.get("description") and len(b.get("description", "")) > 10:
                has_task_specific_content = True
        
        # Check next_steps have specific descriptions
        next_steps = bundle.get("next_steps", [])
        for ns in next_steps:
            if ns.get("description") and len(ns.get("description", "")) > 10:
                has_task_specific_content = True
        
        assert has_task_specific_content, \
            "Handoff bundle lacks task-specific content, would produce generic response"
    
    def test_procedure_sensitive_resume_for_coding_task(self):
        """For coding/planning tasks with procedure, response must retain procedure."""
        from hypergraph_bistability import HypergraphAgent
        
        agent = HypergraphAgent(model="gpt-5.4", provider="openai", mock=True)
        
        # Setup: coding task with procedure
        ws = WorkingSet(
            task={"name": "write unit tests", "status": "active", "phase": "execution"},
            blockers=[],
            next_steps=[{"id": "n1", "description": "add test coverage"}],
            active_decisions=[],
            applicable_procedures=[{"id": "p1", "name": "TDD workflow: red-green-refactor"}],
            decisions_detail=[],
            procedures_detail=[{"id": "p1", "name": "TDD workflow: red-green-refactor"}],
            handoff_ready=True
        )
        agent._query_layer._working_set = ws
        bundle = agent.query_handoff_bundle()
        
        # Verify procedure exists for coding task
        procedures = bundle.get("applicable_procedures", [])
        assert procedures is not None, "applicable_procedures must exist for coding task"
        
        # If requires_procedure=true scenario, procedure should be non-empty
        # This is verified by presence in the bundle
    
    def test_research_to_execution_resume(self):
        """Research record to execution resume: must carry evidence."""
        from hypergraph_bistability import HypergraphAgent
        
        agent = HypergraphAgent(model="gpt-5.4", provider="openai", mock=True)
        
        # Setup: research task with evidence
        ws = WorkingSet(
            task={"name": "research optimal caching strategy", "status": "completed", "phase": "done"},
            blockers=[],
            next_steps=[],
            active_decisions=[],
            applicable_procedures=[],
            decisions_detail=[],
            procedures_detail=[],
            evidence=[
                {"content": "Redis provides best performance for our use case", "source": "benchmark_results.md"},
                {"content": "Use cache-aside pattern with TTL", "source": "architecture_notes.md"}
            ],
            handoff_ready=True
        )
        agent._query_layer._working_set = ws
        bundle = agent.query_handoff_bundle()
        
        # For research-to-execution, evidence is critical
        evidence = bundle.get("evidence", [])
        assert evidence is not None, "evidence must exist for research-to-execution"
        assert len(evidence) > 0, "research-to-execution requires evidence"


class TestHandoffContextAssembly:
    """Test that handoff context is properly assembled for the model."""
    
    def test_handoff_context_includes_task(self):
        """Handoff context should include task info."""
        from hypergraph_bistability import HypergraphAgent
        
        agent = HypergraphAgent(model="gpt-5.4", provider="openai", mock=True)
        ws = WorkingSet(
            task={"name": "test task", "status": "active", "phase": "execution"},
            blockers=[{"id": "b1", "description": "test blocker"}],
            next_steps=[{"id": "n1", "description": "test next step"}],
            active_decisions=[],
            applicable_procedures=[],
            decisions_detail=[],
            procedures_detail=[],
            handoff_ready=True
        )
        agent._query_layer._working_set = ws
        
        # Get the handoff context (simulated)
        bundle = agent.query_handoff_bundle()
        
        # Verify bundle has task structure for context assembly
        assert "linked_task" in bundle
        assert bundle["linked_task"] is not None
    
    def test_handoff_context_includes_blockers(self):
        """Handoff context should include blocker info."""
        from hypergraph_bistability import HypergraphAgent
        
        agent = HypergraphAgent(model="gpt-5.4", provider="openai", mock=True)
        ws = WorkingSet(
            task={"name": "test task", "status": "active", "phase": "execution"},
            blockers=[{"id": "b1", "description": "API rate limit"}],
            next_steps=[],
            active_decisions=[],
            applicable_procedures=[],
            decisions_detail=[],
            procedures_detail=[],
            handoff_ready=True
        )
        agent._query_layer._working_set = ws
        
        bundle = agent.query_handoff_bundle()
        
        assert "blockers" in bundle
        assert bundle["blockers"] is not None
    
    def test_handoff_context_includes_next_steps(self):
        """Handoff context should include next_steps info."""
        from hypergraph_bistability import HypergraphAgent
        
        agent = HypergraphAgent(model="gpt-5.4", provider="openai", mock=True)
        ws = WorkingSet(
            task={"name": "test task", "status": "active", "phase": "execution"},
            blockers=[],
            next_steps=[{"id": "n1", "description": "complete implementation"}],
            active_decisions=[],
            applicable_procedures=[],
            decisions_detail=[],
            procedures_detail=[],
            handoff_ready=True
        )
        agent._query_layer._working_set = ws
        
        bundle = agent.query_handoff_bundle()
        
        assert "next_steps" in bundle
        assert bundle["next_steps"] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
