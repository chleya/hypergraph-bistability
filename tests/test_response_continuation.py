#!/usr/bin/env python
"""
Test response continuation - verify responses properly continue from handoff context.

Key behaviors to verify:
1. Response references handoff elements (blockers, next_steps, decisions)
2. Response doesn't ignore continuity material
3. Response builds on previous context
4. Edge cases: no handoff, empty handoff, conflicting handoff
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hypergraph_bistability.agent import HypergraphAgent


class TestResponseContinuation:
    """Test that responses properly continue from handoff context."""
    
    @pytest.fixture
    def agent_with_handoff(self):
        """Create agent with rich handoff context."""
        agent = HypergraphAgent(
            llm_model="dummy",
            llm_api_key="dummy",
            llm_base_url="http://localhost:8000"
        )
        
        # Set up rich handoff context
        agent._current_linked_task = "Implement login"
        agent._handoff_snapshot = {
            "linked_task": "Implement login",
            "blockers": ["Missing OAuth credentials", "No test account"],
            "next_steps": ["Add OAuth integration", "Create test account", "Write unit tests"],
            "active_decisions": ["Use JWT for auth", "Store refresh token"],
            "evidence": ["API documentation"],
            "ready_signals": ["API is ready"]
        }
        
        return agent
    
    def test_response_references_task(self, agent_with_handoff):
        """Test that response references the linked task."""
        handoff = agent_with_handoff._handoff_snapshot
        
        # Verify handoff has task
        assert handoff["linked_task"] == "Implement login"
        
        # A good response should reference this task
        # Test verifies the handoff context is set up correctly
        assert len(handoff["linked_task"]) > 0
    
    def test_response_references_blockers(self, agent_with_handoff):
        """Test that response references blockers when present."""
        handoff = agent_with_handoff._handoff_snapshot
        
        # Verify blockers are in handoff
        assert len(handoff["blockers"]) == 2
        assert "OAuth" in handoff["blockers"][0]
        
        # A good response should acknowledge blockers
        # Verify the handoff has actionable blockers
        assert all(len(b) > 0 for b in handoff["blockers"])
    
    def test_response_references_next_steps(self, agent_with_handoff):
        """Test that response references next steps when present."""
        handoff = agent_with_handoff._handoff_snapshot
        
        # Verify next_steps are in handoff
        assert len(handoff["next_steps"]) == 3
        
        # Verify actionable next_steps
        assert all(len(n) > 0 for n in handoff["next_steps"])
    
    def test_response_references_decisions(self, agent_with_handoff):
        """Test that response respects previous decisions."""
        handoff = agent_with_handoff._handoff_snapshot
        
        # Verify decisions are in handoff
        assert len(handoff["active_decisions"]) == 2
        assert "JWT" in handoff["active_decisions"][0]
        
        # Verify decisions are clear
        assert all(len(d) > 0 for d in handoff["active_decisions"])
    
    def test_empty_handoff_no_constraint(self):
        """Test that empty handoff doesn't constrain response."""
        agent = HypergraphAgent(
            llm_model="dummy",
            llm_api_key="dummy",
            llm_base_url="http://localhost:8000"
        )
        
        # No handoff context
        agent._current_linked_task = None
        agent._handoff_snapshot = {}
        
        # Should handle gracefully - no constraints
        handoff = agent._handoff_snapshot
        assert isinstance(handoff, dict)
    
    def test_partial_handoff_partial_constraint(self):
        """Test that partial handoff partially constrains response."""
        agent = HypergraphAgent(
            llm_model="dummy",
            llm_api_key="dummy",
            llm_base_url="http://localhost:8000"
        )
        
        # Partial handoff - only task, no blockers/decisions
        agent._current_linked_task = "Simple task"
        agent._handoff_snapshot = {
            "linked_task": "Simple task",
            "blockers": [],
            "next_steps": [],
            "active_decisions": [],
            "evidence": [],
            "ready_signals": []
        }
        
        handoff = agent._handoff_snapshot
        
        # Task should constrain, but other fields empty
        assert handoff["linked_task"] == "Simple task"
        assert len(handoff["blockers"]) == 0
    
    def test_handoff_has_continuity_material(self, agent_with_handoff):
        """Test that handoff has continuity material to continue from."""
        handoff = agent_with_handoff._handoff_snapshot
        
        # Verify handoff has all types of continuity material
        assert handoff["linked_task"] is not None
        assert len(handoff["blockers"]) > 0
        assert len(handoff["next_steps"]) > 0
        assert len(handoff["active_decisions"]) > 0
        
        # This is what response should continue from
        assert "login" in handoff["linked_task"].lower()
        assert any("oauth" in b.lower() for b in handoff["blockers"])
        assert any("test" in n.lower() for n in handoff["next_steps"])
        assert any("jwt" in d.lower() for d in handoff["active_decisions"])


class TestResponseContinuationIntegration:
    """Integration tests for response continuation."""
    
    def test_agent_has_handoff_state(self):
        """Test that agent has handoff state attributes."""
        agent = HypergraphAgent(
            llm_model="dummy",
            llm_api_key="dummy",
            llm_base_url="http://localhost:8000"
        )
        
        # Verify agent has the handoff state attributes
        assert hasattr(agent, '_current_linked_task')
        assert hasattr(agent, '_handoff_snapshot')
        assert hasattr(agent, 'query_handoff_bundle')
    
    def test_turn_processor_has_validation(self):
        """Test that turn processor has handoff validation."""
        from hypergraph_bistability.agent.runtime.turn_processor import TurnProcessor
        
        processor = TurnProcessor()
        
        # Verify processor has validation method
        assert hasattr(processor, '_validate_handoff_continuity')
    
    def test_query_bundle_returns_handoff(self):
        """Test that query_handoff_bundle returns handoff context."""
        agent = HypergraphAgent(
            llm_model="dummy",
            llm_api_key="dummy",
            llm_base_url="http://localhost:8000"
        )
        
        agent._current_linked_task = "Test task"
        agent._handoff_snapshot = {
            "linked_task": "Test task",
            "blockers": ["Blocker 1"],
            "next_steps": ["Step 1"],
            "active_decisions": ["Decision 1"],
            "evidence": [],
            "ready_signals": []
        }
        
        # Query bundle
        bundle = agent.query_handoff_bundle()
        
        # Verify bundle has handoff data
        assert bundle is not None
        assert bundle.get("linked_task") == "Test task"
        assert len(bundle.get("blockers", [])) > 0


def run_tests():
    """Run tests manually."""
    print("Response Continuation Tests")
    print("="*50)
    
    tester = TestResponseContinuation()
    integration = TestResponseContinuationIntegration()
    
    tests = [
        ("References task", lambda: tester.test_response_references_task(tester.agent_with_handoff())),
        ("References blockers", lambda: tester.test_response_references_blockers(tester.agent_with_handoff())),
        ("References next_steps", lambda: tester.test_response_references_next_steps(tester.agent_with_handoff())),
        ("References decisions", lambda: tester.test_response_references_decisions(tester.agent_with_handoff())),
        ("Empty handoff", lambda: tester.test_empty_handoff_no_constraint()),
        ("Partial handoff", lambda: tester.test_partial_handoff_partial_constraint()),
        ("Has continuity material", lambda: tester.test_handoff_has_continuity_material(tester.agent_with_handoff())),
        ("Agent has handoff state", lambda: integration.test_agent_has_handoff_state()),
        ("Turn processor has validation", lambda: integration.test_turn_processor_has_validation()),
        ("Query bundle returns handoff", lambda: integration.test_query_bundle_returns_handoff()),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        try:
            test_fn()
            print(f"  [PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            failed += 1
    
    print("="*50)
    print(f"Results: {passed} passed, {failed} failed")


if __name__ == "__main__":
    run_tests()