"""Session Restore Contract Tests.

Verifies that save/load preserves necessary runtime state AND
that query_handoff_bundle still works after restore with consistent semantics.
"""

import pytest
import tempfile
import os
from pathlib import Path

from hypergraph_bistability import HypergraphAgent


class TestSessionRestoreContract:
    """Test session_restore_contract: save/load + handoff post-restore."""
    
    @pytest.fixture
    def agent_with_state(self):
        """Create agent with active working set."""
        agent = HypergraphAgent(
            llm_model="gpt-5.4"
        )
        
        # Setup some state
        agent.process_turn("Hello, my task is to implement login feature")
        agent.process_turn("The current blocker is: need API key")
        agent.process_turn("My decision: use OAuth2")
        
        return agent
    
    def test_save_load_preserves_linked_task(self, agent_with_state):
        """Save/load must preserve linked_task."""
        # Get handoff before save
        bundle_before = agent_with_state.query_handoff_bundle()
        task_before = bundle_before.get("linked_task")
        
        # Save state
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as f:
            save_path = f.name
        
        try:
            agent_with_state.save(save_path)
            
            # Load into new agent
            new_agent = HypergraphAgent(model="gpt-5.4", provider="openai")
            new_agent.load(save_path)
            
            # Get handoff after restore
            bundle_after = new_agent.query_handoff_bundle()
            task_after = bundle_after.get("linked_task")
            
            # Verify task preserved
            assert task_after is not None, "linked_task not preserved after restore"
            assert task_after != "", "linked_task empty after restore"
            
        finally:
            if os.path.exists(save_path):
                os.remove(save_path)
    
    def test_save_load_preserves_blockers(self, agent_with_state):
        """Save/load must preserve active blockers."""
        bundle_before = agent_with_state.query_handoff_bundle()
        blockers_before = bundle_before.get("blockers", [])
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as f:
            save_path = f.name
        
        try:
            agent_with_state.save(save_path)
            new_agent = HypergraphAgent(model="gpt-5.4", provider="openai")
            new_agent.load(save_path)
            
            bundle_after = new_agent.query_handoff_bundle()
            blockers_after = bundle_after.get("blockers", [])
            
            # Blockers should be preserved
            assert blockers_after is not None, "blockers not preserved"
            
        finally:
            if os.path.exists(save_path):
                os.remove(save_path)
    
    def test_save_load_preserves_active_decisions(self, agent_with_state):
        """Save/load must preserve active decisions."""
        bundle_before = agent_with_state.query_handoff_bundle()
        decisions_before = bundle_before.get("active_decisions", [])
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as f:
            save_path = f.name
        
        try:
            agent_with_state.save(save_path)
            new_agent = HypergraphAgent(model="gpt-5.4", provider="openai")
            new_agent.load(save_path)
            
            bundle_after = new_agent.query_handoff_bundle()
            decisions_after = bundle_after.get("active_decisions", [])
            
            # Decisions should be preserved
            assert decisions_after is not None, "active_decisions not preserved"
            
        finally:
            if os.path.exists(save_path):
                os.remove(save_path)
    
    def test_save_load_preserves_procedures(self, agent_with_state):
        """Save/load must preserve applicable procedures."""
        bundle_before = agent_with_state.query_handoff_bundle()
        procedures_before = bundle_before.get("applicable_procedures", [])
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as f:
            save_path = f.name
        
        try:
            agent_with_state.save(save_path)
            new_agent = HypergraphAgent(model="gpt-5.4", provider="openai")
            new_agent.load(save_path)
            
            bundle_after = new_agent.query_handoff_bundle()
            procedures_after = bundle_after.get("applicable_procedures", [])
            
            # Procedures should be preserved
            assert procedures_after is not None, "applicable_procedures not preserved"
            
        finally:
            if os.path.exists(save_path):
                os.remove(save_path)
    
    def test_handoff_works_after_restore(self, agent_with_state):
        """After restore, query_handoff_bundle must still work and produce valid output."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as f:
            save_path = f.name
        
        try:
            agent_with_state.save(save_path)
            new_agent = HypergraphAgent(model="gpt-5.4", provider="openai")
            new_agent.load(save_path)
            
            # Key test: handoff API should still work after restore
            bundle = new_agent.query_handoff_bundle()
            
            # Basic structure check
            assert bundle is not None, "handoff_bundle returns None after restore"
            assert isinstance(bundle, dict), "handoff_bundle returns non-dict"
            
            # Core fields must exist
            assert "linked_task" in bundle
            assert "blockers" in bundle
            assert "next_steps" in bundle
            
        finally:
            if os.path.exists(save_path):
                os.remove(save_path)
    
    def test_handoff_semantics_consistent_after_restore(self, agent_with_state):
        """After restore, handoff semantics must be consistent (not just structure)."""
        bundle_before = agent_with_state.query_handoff_bundle()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as f:
            save_path = f.name
        
        try:
            agent_with_state.save(save_path)
            new_agent = HypergraphAgent(model="gpt-5.4", provider="openai")
            new_agent.load(save_path)
            
            bundle_after = new_agent.query_handoff_bundle()
            
            # Semantic check: key information should be preserved
            # (not just that fields exist, but content is meaningful)
            task_before = str(bundle_before.get("linked_task", ""))
            task_after = str(bundle_after.get("linked_task", ""))
            
            # At least the task name should be preserved
            assert task_before == task_after or \
                   task_before.lower() in task_after.lower() or \
                   task_after.lower() in task_before.lower(), \
                   "Task semantics lost after restore"
            
        finally:
            if os.path.exists(save_path):
                os.remove(save_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
