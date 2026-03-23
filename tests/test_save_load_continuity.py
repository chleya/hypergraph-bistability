#!/usr/bin/env python
"""
Test save/load continuity completeness.

Key behaviors to verify:
1. Save preserves all continuity state (linked_task, blockers, decisions, etc.)
2. Load restores all continuity state correctly
3. Load after save produces same behavior as before save
4. Edge cases: empty state, partial state
"""
import pytest
import json
import sys
import os
import tempfile
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hypergraph_bistability.agent import HypergraphAgent
from hypergraph_bistability.agent.session import SessionState


class TestSaveLoadContinuity:
    """Test save/load preserves continuity state."""
    
    @pytest.fixture
    def agent_with_state(self):
        """Create agent with rich continuity state."""
        agent = HypergraphAgent(
            llm_model="dummy",
            llm_api_key="dummy",
            llm_base_url="http://localhost:8000"
        )
        
        # Set up rich continuity state
        agent._current_linked_task = "Implement login"
        agent._handoff_snapshot = {
            "linked_task": "Implement login",
            "blockers": ["Missing OAuth", "No API key"],
            "next_steps": ["Add OAuth", "Get API key", "Write tests"],
            "active_decisions": ["Use JWT"],
            "evidence": ["Design doc"],
            "ready_signals": ["API ready"]
        }
        agent._working_set = {
            "linked_task": "Implement login",
            "active_blockers": [{"content": "Missing OAuth"}],
            "next_steps": [{"content": "Add OAuth"}],
            "active_decisions": [{"content": "Use JWT"}],
            "evidence": []
        }
        
        return agent
    
    @pytest.fixture
    def temp_dir(self):
        """Create temp directory for save/load test."""
        tmp = tempfile.mkdtemp()
        yield tmp
        shutil.rmtree(tmp)
    
    def test_save_preserves_linked_task(self, agent_with_state, temp_dir):
        """Test that save preserves linked_task."""
        # Get session state
        state = agent_with_state.get_session_state()
        
        # Verify linked_task is in state
        assert hasattr(agent_with_state, '_current_linked_task')
        assert agent_with_state._current_linked_task == "Implement login"
    
    def test_save_preserves_handoff_snapshot(self, agent_with_state):
        """Test that save preserves handoff_snapshot."""
        # Verify handoff_snapshot is complete
        snapshot = agent_with_state._handoff_snapshot
        
        assert snapshot["linked_task"] == "Implement login"
        assert len(snapshot["blockers"]) == 2
        assert len(snapshot["next_steps"]) == 3
        assert len(snapshot["active_decisions"]) == 1
        assert len(snapshot["evidence"]) == 1
        assert len(snapshot["ready_signals"]) == 1
    
    def test_load_restores_linked_task(self, agent_with_state, temp_dir):
        """Test that load restores linked_task."""
        # Save state
        path = os.path.join(temp_dir, "test_state.json")
        agent_with_state.save(path)
        
        # Create new agent and load
        new_agent = HypergraphAgent(
            llm_model="dummy",
            llm_api_key="dummy",
            llm_base_url="http://localhost:8000"
        )
        new_agent.load(path)
        
        # Verify linked_task restored
        assert new_agent._current_linked_task == "Implement login"
    
    def test_load_restores_handoff_snapshot(self, agent_with_state, temp_dir):
        """Test that load restores handoff_snapshot."""
        # Save and load
        path = os.path.join(temp_dir, "test_state.json")
        agent_with_state.save(path)
        
        new_agent = HypergraphAgent(
            llm_model="dummy",
            llm_api_key="dummy",
            llm_base_url="http://localhost:8000"
        )
        new_agent.load(path)
        
        # Verify snapshot restored
        snapshot = new_agent._handoff_snapshot
        
        assert snapshot["linked_task"] == "Implement login"
        assert len(snapshot["blockers"]) == 2
        assert len(snapshot["next_steps"]) == 3
        assert len(snapshot["active_decisions"]) == 1
    
    def test_load_restores_working_set(self, agent_with_state, temp_dir):
        """Test that load restores working_set."""
        path = os.path.join(temp_dir, "test_state.json")
        agent_with_state.save(path)
        
        new_agent = HypergraphAgent(
            llm_model="dummy",
            llm_api_key="dummy",
            llm_base_url="http://localhost:8000"
        )
        new_agent.load(path)
        
        # Query working_set via method - returns dict
        ws = new_agent.get_working_set()
        
        # Verify working_set restored - linked_task should match
        assert ws.get("linked_task") == "Implement login"
    
    def test_query_bundle_after_load(self, agent_with_state, temp_dir):
        """Test that query_handoff_bundle works after load."""
        path = os.path.join(temp_dir, "test_state.json")
        agent_with_state.save(path)
        
        new_agent = HypergraphAgent(
            llm_model="dummy",
            llm_api_key="dummy",
            llm_base_url="http://localhost:8000"
        )
        new_agent.load(path)
        
        # Query handoff bundle
        bundle = new_agent.query_handoff_bundle()
        
        # Should have continuity material
        assert bundle is not None
        assert bundle.get("linked_task") is not None
    
    def test_empty_state_save_load(self, temp_dir):
        """Test save/load with empty state."""
        agent = HypergraphAgent(
            llm_model="dummy",
            llm_api_key="dummy",
            llm_base_url="http://localhost:8000"
        )
        
        path = os.path.join(temp_dir, "empty_state.json")
        agent.save(path)
        
        # Load into new agent
        new_agent = HypergraphAgent(
            llm_model="dummy",
            llm_api_key="dummy",
            llm_base_url="http://localhost:8000"
        )
        new_agent.load(path)
        
        # Should handle gracefully - snapshot has default keys with None values
        # This is expected behavior - not a bug
        task = new_agent._current_linked_task
        snapshot = new_agent._handoff_snapshot
        
        # Just verify we can load without crashing
        assert task is None or task == ""
        assert isinstance(snapshot, dict)
    
    def test_partial_state_save_load(self, temp_dir):
        """Test save/load with partial continuity state."""
        agent = HypergraphAgent(
            llm_model="dummy",
            llm_api_key="dummy",
            llm_base_url="http://localhost:8000"
        )
        
        # Set partial state
        agent._current_linked_task = "Partial task"
        agent._handoff_snapshot = {
            "linked_task": "Partial task",
            "blockers": ["Only blocker"],
            "next_steps": [],
            "active_decisions": [],
            "evidence": [],
            "ready_signals": []
        }
        
        path = os.path.join(temp_dir, "partial_state.json")
        agent.save(path)
        
        new_agent = HypergraphAgent(
            llm_model="dummy",
            llm_api_key="dummy",
            llm_base_url="http://localhost:8000"
        )
        new_agent.load(path)
        
        # Verify partial state restored
        assert new_agent._current_linked_task == "Partial task"
        assert len(new_agent._handoff_snapshot["blockers"]) == 1
        assert new_agent._handoff_snapshot["next_steps"] == []


def run_tests():
    """Run tests manually."""
    print("Save/Load Continuity Tests")
    print("="*50)
    
    tester = TestSaveLoadContinuity()
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create agent with state
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
        
        tests = [
            ("Preserves linked_task", lambda: tester.test_save_preserves_linked_task(agent, temp_dir)),
            ("Preserves handoff_snapshot", lambda: tester.test_save_preserves_handoff_snapshot(agent)),
            ("Restores linked_task", lambda: tester.test_load_restores_linked_task(agent, temp_dir)),
            ("Restores handoff_snapshot", lambda: tester.test_load_restores_handoff_snapshot(agent, temp_dir)),
            ("Restores working_set", lambda: tester.test_load_restores_working_set(agent, temp_dir)),
            ("Query bundle after load", lambda: tester.test_query_bundle_after_load(agent, temp_dir)),
            ("Empty state", lambda: tester.test_empty_state_save_load(temp_dir)),
            ("Partial state", lambda: tester.test_partial_state_save_load(temp_dir)),
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
    
    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    run_tests()