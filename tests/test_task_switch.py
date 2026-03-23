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
        agent._handoff_snapshot = {
            "linked_task": "persisted_task",
            "blockers": ["blocker1"],
            "next_steps": ["step1"],
            "active_decisions": [],
            "applicable_procedures": [],
            "evidence": [],
            "dominant_conflict": None,
            "ready_signals": [],
        }
        
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
            
            # Query handoff should return persisted data
            bundle = agent2.query_handoff_bundle()
            assert bundle["linked_task"] == "persisted_task"
            assert bundle["blockers"] == ["blocker1"]
            
        finally:
            # Cleanup
            if os.path.exists(filepath):
                os.remove(filepath)
            history = filepath.replace('.json', '_history.json')
            if os.path.exists(history):
                os.remove(history)
                
    def test_fallback_uses_persisted_when_sparse(self):
        """When runtime state is sparse, should fallback to persisted snapshot."""
        from hypergraph_bistability.agent.hypergraph_agent import HypergraphAgent
        import tempfile
        import os
        
        # Create agent with persisted snapshot
        agent = HypergraphAgent(k=4, L=2, name='test_fallback')
        agent.llm_api_key = 'test'
        
        # Set persisted snapshot
        agent._handoff_snapshot = {
            "linked_task": "old_task",
            "blockers": ["old_blocker"],
            "next_steps": ["old_step"],
            "active_decisions": ["old_decision"],
            "applicable_procedures": ["old_proc"],
            "evidence": ["old_evidence"],
            "dominant_conflict": {"content": "conflict"},
            "ready_signals": ["signal1"],
        }
        
        # Query handoff when runtime is sparse (no current task)
        # Since _current_linked_task is empty and hypergraph is empty
        bundle = agent.query_handoff_bundle()
        
        # Should fallback to persisted
        assert bundle.get("_fallback") == True
        assert bundle["blockers"] == ["old_blocker"]
        assert bundle["next_steps"] == ["old_step"]
        assert bundle["active_decisions"] == ["old_decision"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
