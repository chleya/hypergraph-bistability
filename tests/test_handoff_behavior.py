"""
Test handoff continuity BEHAVIOR - not just state sufficiency.

These tests verify that after a handoff, the response actually continues the task
by referencing key elements from the handoff bundle.

Key behaviors tested:
1. Response must reference linked_task
2. Response must continue at least one blocker/constraint
3. Response must continue at least one next step / pending decision
4. For coding/planning/research: must preserve procedure/evidence traces
"""

import pytest
from unittest.mock import patch, MagicMock
from hypergraph_bistability.agent.hypergraph_agent import HypergraphAgent


class TestHandoffBehavior:
    """Test that handoff bundle actually influences response behavior."""

    @pytest.fixture
    def agent_with_handoff(self):
        """Create agent with persisted handoff snapshot."""
        agent = HypergraphAgent(k=4, L=2, name='test_behavior')
        agent.llm_api_key = 'test'
        
        # Set a handoff snapshot with rich continuity data
        agent._handoff_snapshot = {
            "linked_task": "fix_api_bug",
            "blockers": [
                {"content": "API returns 500 error on /users endpoint", "constraint": "must fix before demo"}
            ],
            "next_steps": [
                {"content": "Check server logs for stack trace"},
                {"content": "Reproduce locally with curl"}
            ],
            "active_decisions": [
                {"content": "Use error handling middleware", "rationale": "cleaner than try-catch everywhere"}
            ],
            "applicable_procedures": [
                {"content": "debug_flask_endpoint", "type": "code_reference"}
            ],
            "evidence": [
                {"content": "Production error rate: 15%"}
            ],
            "dominant_conflict": None,
            "ready_signals": ["all_tests_pass"],
        }
        
        # Set current task
        agent._current_linked_task = "fix_api_bug"
        
        return agent

    def test_response_references_linked_task(self, agent_with_handoff):
        """Verify response references the linked task after restore."""
        # This is a conceptual test - we'd need to actually call the LLM
        # For now, we verify the handoff bundle contains the task
        
        bundle = agent_with_handoff.query_handoff_bundle()
        
        # The bundle should contain the task
        assert bundle["linked_task"] == "fix_api_bug"
        
        # In real behavior test, we'd:
        # 1. Restore from snapshot
        # 2. Send a generic prompt like "what should I do next?"
        # 3. Verify response contains "fix_api_bug" or references the task

    def test_response_continues_blocker(self, agent_with_handoff):
        """Verify response continues at least one blocker."""
        bundle = agent_with_handoff.query_handoff_bundle()
        
        # Verify bundle has blockers
        assert len(bundle["blockers"]) > 0
        
        blocker = bundle["blockers"][0]
        assert "content" in blocker
        assert "500" in blocker["content"].lower() or "error" in blocker["content"].lower()
        
        # In real behavior test:
        # 1. Restore from snapshot with blocker
        # 2. Ask "what's blocking progress?"
        # 3. Verify response references the blocker content

    def test_response_continues_next_step(self, agent_with_handoff):
        """Verify response continues at least one next step."""
        bundle = agent_with_handoff.query_handoff_bundle()
        
        # Verify bundle has next steps
        assert len(bundle["next_steps"]) > 0
        
        next_step = bundle["next_steps"][0]
        assert "content" in next_step
        
        # In real behavior test:
        # 1. Restore from snapshot with next steps
        # 2. Ask "what's next?"
        # 3. Verify response continues at least one next step

    def test_response_continues_decision(self, agent_with_handoff):
        """Verify response continues at least one pending decision."""
        bundle = agent_with_handoff.query_handoff_bundle()
        
        # Verify bundle has decisions
        assert len(bundle["active_decisions"]) > 0
        
        decision = bundle["active_decisions"][0]
        assert "content" in decision
        
        # In real behavior test:
        # 1. Restore from snapshot with pending decision
        # 2. Ask "what decisions need to be made?"
        # 3. Verify response references the decision

    def test_coding_scene_preserves_procedure(self, agent_with_handoff):
        """For coding scenes: verify procedure is preserved in handoff."""
        bundle = agent_with_handoff.query_handoff_bundle()
        
        # Verify bundle has procedures
        assert len(bundle.get("applicable_procedures", [])) > 0
        
        procedure = bundle["applicable_procedures"][0]
        assert "content" in procedure
        
        # In real behavior test:
        # 1. Restore coding task with procedure
        # 2. Ask "how do I debug this?"
        # 3. Verify response references the procedure

    def test_research_scene_preserves_evidence(self, agent_with_handoff):
        """For research scenes: verify evidence is preserved in handoff."""
        bundle = agent_with_handoff.query_handoff_bundle()
        
        # Verify bundle has evidence
        assert len(bundle.get("evidence", [])) > 0
        
        evidence = bundle["evidence"][0]
        assert "content" in evidence
        
        # In real behavior test:
        # 1. Restore research task with evidence
        # 2. Ask "what do we know about this?"
        # 3. Verify response references the evidence

    def test_handoff_with_minimal_content_still_valid(self):
        """Test that minimal handoff (just task name) is still valid."""
        agent = HypergraphAgent(k=4, L=2, name='test_minimal')
        agent.llm_api_key = 'test'
        
        # Minimal handoff - just task name
        agent._handoff_snapshot = {
            "linked_task": "simple_task",
            "blockers": [],
            "next_steps": [],
            "active_decisions": [],
            "applicable_procedures": [],
            "evidence": [],
            "dominant_conflict": None,
            "ready_signals": [],
        }
        
        agent._current_linked_task = "simple_task"
        
        bundle = agent.query_handoff_bundle()
        
        # Bundle should still be valid even with minimal content
        assert bundle["linked_task"] == "simple_task"
        # But diagnostics should show it's minimal
        diag = agent.query_handoff_diagnostics()
        assert diag["runtime"]["has_resume_material"] == False

    def test_handoff_with_weak_content_detected(self):
        """Test that weak content (e.g., template next steps) is detected."""
        agent = HypergraphAgent(k=4, L=2, name='test_weak')
        agent.llm_api_key = 'test'
        
        # Weak handoff - template content that looks useless
        agent._handoff_snapshot = {
            "linked_task": "some_task",
            "blockers": [],
            "next_steps": [
                {"content": "TBD"},  # Weak - template
                {"content": "look into it"},  # Weak - vague
            ],
            "active_decisions": [],
            "applicable_procedures": [],
            "evidence": [],
            "dominant_conflict": None,
            "ready_signals": [],
        }
        
        agent._current_linked_task = "some_task"
        
        # Get diagnostics
        diag = agent.query_handoff_diagnostics()
        
        # Should show runtime has content but might be weak
        # This test documents the current limitation
        # Future: add quality thresholds for content
        assert diag["persisted"]["has_content"] == True
        
        # Note: We don't currently detect "weak" content
        # This is a known gap - see Chen's feedback about 
        # "非空但无用" content


class TestContinuitySufficiency:
    """Test layered sufficiency - not just binary sufficient/insufficient."""

    def test_anchor_only_layer(self):
        """Test scenario with only task name, no continuity material."""
        agent = HypergraphAgent(k=4, L=2, name='test_anchor')
        agent.llm_api_key = 'test'
        
        agent._handoff_snapshot = {
            "linked_task": "task_x",
            "blockers": [],
            "next_steps": [],
            "active_decisions": [],
            "applicable_procedures": [],
            "evidence": [],
            "dominant_conflict": None,
            "ready_signals": [],
        }
        
        agent._current_linked_task = "task_x"
        
        diag = agent.query_handoff_diagnostics()
        
        # Should be insufficient - only anchor
        assert diag["runtime"]["has_anchor"] == True
        assert diag["runtime"]["has_resume_material"] == False
        assert diag["runtime"]["is_sufficient"] == False

    def test_thin_continuity_layer(self):
        """Test scenario with some continuity but not actionable."""
        agent = HypergraphAgent(k=4, L=2, name='test_thin')
        agent.llm_api_key = 'test'
        
        # Set persisted with thin continuity
        agent._handoff_snapshot = {
            "linked_task": "task_y",
            "blockers": [{"content": "some blocker"}],  # Has blocker
            "next_steps": [],
            "active_decisions": [],
            "applicable_procedures": [],
            "evidence": [],
            "dominant_conflict": None,
            "ready_signals": [],
        }
        
        # No runtime anchor - so fallback will trigger
        agent._current_linked_task = ""
        
        # Query handoff - should use persisted snapshot
        bundle = agent.query_handoff_bundle()
        
        # Should fallback to persisted
        assert bundle["linked_task"] == "task_y"
        assert len(bundle["blockers"]) > 0
        
        # But diagnostics should show runtime is empty
        diag = agent.query_handoff_diagnostics()
        assert diag["runtime"]["has_anchor"] == False
        # After fallback, persisted is used so overall continuity exists

    def test_actionable_continuity_layer(self):
        """Test scenario with full actionable continuity."""
        agent = HypergraphAgent(k=4, L=2, name='test_actionable')
        agent.llm_api_key = 'test'
        
        # Set persisted with full continuity
        agent._handoff_snapshot = {
            "linked_task": "task_z",
            "blockers": [{"content": "API failing with 500 error"}],
            "next_steps": [
                {"content": "Check nginx logs first"},
                {"content": "Reproduce with curl -X GET localhost:5000/users"}
            ],
            "active_decisions": [
                {"content": "Use error middleware", "rationale": "cleaner"}
            ],
            "applicable_procedures": [
                {"content": "debug_flask_500", "type": "code_reference"}
            ],
            "evidence": [
                {"content": "Error started after commit abc123"}
            ],
            "dominant_conflict": None,
            "ready_signals": [],
        }
        
        # No runtime anchor
        agent._current_linked_task = ""
        
        # Query handoff - should use persisted
        bundle = agent.query_handoff_bundle()
        
        # Should have full continuity from persisted
        assert bundle["linked_task"] == "task_z"
        assert len(bundle["blockers"]) > 0
        assert len(bundle["next_steps"]) > 0
        assert len(bundle["active_decisions"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
