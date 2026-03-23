"""Test task switching behavior for handoff continuity."""
import pytest
import sys
import os
sys.path.insert(0, 'src')

os.environ['OPENAI_API_KEY'] = 'test-key'


class TestTaskSwitch:
    """Test task switching scenarios."""
    
    def test_explicit_task_overwrites_stored(self):
        """When explicit task is passed, it overwrites stored _current_linked_task."""
        from hypergraph_bistability.agent.hypergraph_agent import HypergraphAgent
        
        agent = HypergraphAgent(k=4, L=2, name='test_task_switch')
        agent.llm_api_key = 'test'
        
        # Set initial task
        agent._current_linked_task = "old_task"
        
        # Call get_working_set with explicit new task
        ws1 = agent.get_working_set(linked_task="new_task")
        
        # Should have new task
        assert ws1["linked_task"] == "new_task"
        
        # Stored should be updated
        assert agent._current_linked_task == "new_task"
        
    def test_task_switch_clears_old_task(self):
        """Switching to new explicit task should clear old task."""
        from hypergraph_bistability.agent.hypergraph_agent import HypergraphAgent
        
        agent = HypergraphAgent(k=4, L=2, name='test_task_switch2')
        agent.llm_api_key = 'test'
        
        # Set initial task
        agent._current_linked_task = "task_A"
        
        # Switch to new explicit task
        agent.get_working_set(linked_task="task_B")
        
        # Should have task B, not A
        assert agent._current_linked_task == "task_B"
        
    def test_empty_task_does_not_clear_stored(self):
        """Calling with no explicit task should preserve stored task."""
        from hypergraph_bistability.agent.hypergraph_agent import HypergraphAgent
        
        agent = HypergraphAgent(k=4, L=2, name='test_task_switch3')
        agent.llm_api_key = 'test'
        
        # Set initial task
        agent._current_linked_task = "stored_task"
        
        # Call without explicit task
        ws = agent.get_working_set()
        
        # Should preserve stored task
        assert ws["linked_task"] == "stored_task"
        
    def test_restore_preserves_task_across_session(self):
        """Save/load should preserve linked_task."""
        from hypergraph_bistability.agent.hypergraph_agent import HypergraphAgent
        import tempfile
        import os
        
        agent = HypergraphAgent(k=4, L=2, name='test_restore')
        agent.llm_api_key = 'test'
        
        # Set task
        agent._current_linked_task = "persisted_task"
        
        # Save
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as f:
            filepath = f.name
        try:
            agent.save(filepath)
            
            # Load into new agent
            agent2 = HypergraphAgent(k=4, L=2, name='test_restore2')
            agent2.llm_api_key = 'test'
            agent2.load(filepath)
            
            # Task should be restored
            assert agent2._current_linked_task == "persisted_task"
            
        finally:
            # Cleanup
            if os.path.exists(filepath):
                os.remove(filepath)
            history = filepath.replace('.json', '_history.json')
            if os.path.exists(history):
                os.remove(history)
                
    def test_fallback_requires_no_runtime_anchor(self):
        """Fallback only when NO linked_task in runtime, not just empty nodes."""
        from hypergraph_bistability.agent.hypergraph_agent import HypergraphAgent
        
        # Create agent with persisted snapshot for task_A
        agent = HypergraphAgent(k=4, L=2, name='test_fallback')
        agent.llm_api_key = 'test'
        
        # Set persisted snapshot for task_A
        agent._handoff_snapshot = {
            "linked_task": "task_A",
            "blockers": ["old_blocker"],
            "next_steps": ["old_step"],
            "active_decisions": ["old_decision"],
            "applicable_procedures": ["old_proc"],
            "evidence": ["old_evidence"],
            "dominant_conflict": {"content": "conflict"},
            "ready_signals": ["signal1"],
        }
        
        # Query handoff when runtime has task_B (fresh anchor)
        agent._current_linked_task = "task_B"
        bundle = agent.query_handoff_bundle()
        
        # Should NOT fallback - runtime has fresh anchor
        assert bundle["linked_task"] == "task_B"
        
    def test_fallback_blocked_for_different_task(self):
        """Fallback blocked when snapshot task differs from runtime task."""
        from hypergraph_bistability.agent.hypergraph_agent import HypergraphAgent
        
        # Create agent with persisted snapshot for task_A
        agent = HypergraphAgent(k=4, L=2, name='test_diff_task')
        agent.llm_api_key = 'test'
        
        # Set persisted snapshot for task_A
        agent._handoff_snapshot = {
            "linked_task": "task_A",
            "blockers": ["old_blocker"],
            "next_steps": ["old_step"],
            "active_decisions": [],
            "applicable_procedures": [],
            "evidence": [],
            "dominant_conflict": None,
            "ready_signals": [],
        }
        
        # Set current task to task_B (different from snapshot)
        agent._current_linked_task = "task_B"
        
        # Query handoff - should NOT use snapshot from different task
        bundle = agent.query_handoff_bundle()
        
        # Should return task_B, not task_A's snapshot
        assert bundle["linked_task"] == "task_B"
        
    def test_task_switch_invalidates_old_snapshot(self):
        """Switching to new task should invalidate old snapshot."""
        from hypergraph_bistability.agent.hypergraph_agent import HypergraphAgent
        
        agent = HypergraphAgent(k=4, L=2, name='test_switch')
        agent.llm_api_key = 'test'
        
        # Set initial task with snapshot
        agent._current_linked_task = "task_A"
        agent._handoff_snapshot = {
            "linked_task": "task_A",
            "blockers": ["blocker_A"],
            "next_steps": [],
            "active_decisions": [],
            "applicable_procedures": [],
            "evidence": [],
            "dominant_conflict": None,
            "ready_signals": [],
        }
        
        # Switch to new task
        agent.get_working_set(linked_task="task_B")
        
        # Old snapshot should be invalidated
        assert agent._handoff_snapshot == {}
        
    def test_fallback_uses_snapshot_when_no_runtime_anchor(self):
        """Fallback uses snapshot when runtime has NO anchor at all."""
        from hypergraph_bistability.agent.hypergraph_agent import HypergraphAgent
        
        agent = HypergraphAgent(k=4, L=2, name='test_no_anchor')
        agent.llm_api_key = 'test'
        
        # Set persisted snapshot but NO runtime anchor
        agent._handoff_snapshot = {
            "linked_task": "previous_task",
            "blockers": ["blocker1"],
            "next_steps": ["step1"],
            "active_decisions": ["decision1"],
            "applicable_procedures": [],
            "evidence": [],
            "dominant_conflict": None,
            "ready_signals": [],
        }
        
        # No runtime task set
        agent._current_linked_task = ""
        
        # Query handoff - should use snapshot since no runtime anchor
        bundle = agent.query_handoff_bundle()
        
        # Should use persisted snapshot
        assert bundle["linked_task"] == "previous_task"
        assert bundle["blockers"] == ["blocker1"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
