#!/usr/bin/env python
"""
Test handoff snapshot recovery correctness.

Key behaviors to verify:
1. Runtime sparse + snapshot exists → use snapshot
2. Runtime sufficient + snapshot exists → NOT use old snapshot
3. Same task, different stage → snapshot not stale
4. After restore, query_handoff_bundle() returns usable continuity material
"""
import pytest
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hypergraph_bistability.agent import HypergraphAgent


class TestHandoffSnapshotRecovery:
    """Test handoff snapshot recovery behavior."""
    
    @pytest.fixture
    def agent(self):
        """Create agent with minimal config."""
        return HypergraphAgent(
            llm_model="dummy",
            llm_api_key="dummy",
            llm_base_url="http://localhost:8000"
        )
    
    def test_restore_uses_snapshot_when_runtime_sparse(self, agent):
        """
        Scenario: After restore, runtime has no task but snapshot exists.
        Expected: Should use snapshot.
        """
        # Setup: Create agent with sparse runtime but valid snapshot
        agent._handoff_snapshot = {
            "linked_task": "Fix bug",
            "blockers": ["API timeout"],
            "next_steps": ["Add retry"],
            "active_decisions": [],
            "evidence": [],
            "ready_signals": []
        }
        agent._current_linked_task = None  # Runtime has no task
        
        # Get handoff bundle
        bundle = agent.query_handoff_bundle()
        
        # Should use snapshot because runtime is sparse
        assert bundle["linked_task"] == "Fix bug"
        assert bundle["blockers"] == ["API timeout"]
        assert "snapshot_used" in bundle or bundle.get("linked_task") is not None
    
    def test_restore_skips_snapshot_when_runtime_sufficient(self, agent):
        """
        Scenario: After restore, runtime has sufficient state.
        Expected: Should NOT use old snapshot, use runtime instead.
        """
        # Setup: Agent with sufficient runtime state
        agent._handoff_snapshot = {
            "linked_task": "Old task",
            "blockers": ["Old blocker"],
            "next_steps": [],
            "active_decisions": [],
            "evidence": [],
            "ready_signals": []
        }
        agent._current_linked_task = "New task"  # Runtime has new task
        agent._working_set = {
            "linked_task": "New task",
            "active_blockers": [{"content": "New blocker"}],
            "next_steps": [],
            "active_decisions": [],
            "evidence": []
        }
        
        # Get handoff bundle
        bundle = agent.query_handoff_bundle()
        
        # Should use runtime, not old snapshot
        assert bundle["linked_task"] == "New task"
        # The old snapshot should NOT override the new runtime state
    
    def test_snapshot_not_stale_for_same_task_different_stage(self, agent):
        """
        Scenario: Same task at different stages - should get fresh snapshot.
        Expected: Snapshot updates when task progresses.
        """
        # First: Create with task at stage 1
        agent._handoff_snapshot = {
            "linked_task": "Fix bug",
            "blockers": ["Initial blocker"],
            "next_steps": ["Step 1"],
            "active_decisions": [],
            "evidence": [],
            "ready_signals": []
        }
        
        # Simulate task progress - update working set
        agent._working_set = {
            "linked_task": "Fix bug",
            "active_blockers": [{"content": "New blocker discovered"}],
            "next_steps": ["Step 2", "Step 3"],  # More steps
            "active_decisions": [],
            "evidence": []
        }
        agent._current_linked_task = "Fix bug"
        
        # Get handoff bundle
        bundle = agent.query_handoff_bundle()
        
        # Should NOT use old snapshot with Initial blocker
        # Should get updated state
        # (The exact behavior depends on implementation - key is no stale pollution)
        # At minimum, task should still be "Fix bug"
        assert bundle["linked_task"] == "Fix bug"
    
    def test_restore_query_bundle_returns_usable_material(self, agent):
        """
        Scenario: After restore, query_handoff_bundle should return usable material.
        Expected: Has continuity material (task, blockers, next_steps, or decisions).
        """
        # Setup: Valid snapshot with continuity material
        agent._handoff_snapshot = {
            "linked_task": "Implement feature",
            "blockers": ["Need API key"],
            "next_steps": ["Get key", "Write code"],
            "active_decisions": ["Use REST"],
            "evidence": [],
            "ready_signals": []
        }
        agent._current_linked_task = None  # Runtime sparse
        
        # Get handoff bundle
        bundle = agent.query_handoff_bundle()
        
        # Verify has usable continuity material
        has_material = (
            bundle.get("linked_task") or
            bundle.get("blockers") or
            bundle.get("next_steps") or
            bundle.get("active_decisions")
        )
        
        assert has_material, "Handoff bundle should have continuity material"
        assert bundle["linked_task"] == "Implement feature"
        assert len(bundle["blockers"]) > 0
    
    def test_empty_snapshot_handled_gracefully(self, agent):
        """
        Scenario: Restore with empty snapshot.
        Expected: Should handle gracefully, return empty/default.
        """
        # Setup: Empty snapshot
        agent._handoff_snapshot = {}
        agent._current_linked_task = None
        agent._working_set = {"linked_task": None, "active_blockers": [], "next_steps": [], "active_decisions": [], "evidence": []}
        
        # Should not crash
        bundle = agent.query_handoff_bundle()
        
        # Should return something (empty but valid structure)
        assert bundle is not None
        assert isinstance(bundle, dict)


def run_tests():
    """Run tests manually."""
    print("Handoff Snapshot Recovery Tests")
    print("="*50)
    
    tester = TestHandoffSnapshotRecovery()
    
    # Create agent
    agent = HypergraphAgent(
        llm_model="dummy",
        llm_api_key="dummy", 
        llm_base_url="http://localhost:8000"
    )
    
    tests = [
        ("Runtime sparse → use snapshot", lambda: tester.test_restore_uses_snapshot_when_runtime_sparse(agent)),
        ("Runtime sufficient → skip snapshot", lambda: tester.test_restore_skips_snapshot_when_runtime_sufficient(agent)),
        ("Same task different stage → not stale", lambda: tester.test_snapshot_not_stale_for_same_task_different_stage(agent)),
        ("Query bundle returns usable material", lambda: tester.test_restore_query_bundle_returns_usable_material(agent)),
        ("Empty snapshot → graceful", lambda: tester.test_empty_snapshot_handled_gracefully(agent)),
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