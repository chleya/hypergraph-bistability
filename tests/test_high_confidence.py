#!/usr/bin/env python
"""Quick test to verify the new high-confidence strong_pass logic."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hypergraph_bistability.agent.runtime.context_assembler import ContextAssembler


def test_new_logic():
    """Test the new high-confidence strict strong_pass logic."""
    assembler = ContextAssembler()
    
    # Test case 1: Should be WEAK_PASS (no exact phrase match)
    handoff1 = {
        "linked_task": "Fix login bug",
        "blockers": ["OAuth token expired"],
        "next_steps": ["Check credentials"],
        "active_decisions": []
    }
    # This response mentions "login bug" and "OAuth token" but NOT the full phrases
    response1 = "I see the login bug is related to OAuth token issues. Let me check credentials."
    
    result1 = assembler.validate_handoff_continuity(response1, handoff1)
    print(f"Test 1 (keyword only, no phrase match): {result1['status']}")
    print(f"  task_ref: {result1.get('task_reference')}")
    print(f"  blocker_ref: {result1.get('blocker_reference')}")
    print(f"  blocker_high_conf:: {result1.get('blocker_high_confidence')}")
    print(f"  next_step_ref: {result1.get('next_step_reference')}")
    print(f"  next_step_high_conf: {result1.get('next_step_high_confidence')}")
    print()
    
    # Test case 2: Should be STRONG_PASS (exact phrase match)  
    handoff2 = {
        "linked_task": "Fix login bug",
        "blockers": ["OAuth token expired"],
        "next_steps": [],
        "active_decisions": []
    }
    # This response contains the full phrase "OAuth token expired"
    response2 = "The issue is OAuth token expired. I will fix login bug now."
    
    result2 = assembler.validate_handoff_continuity(response2, handoff2)
    print(f"Test 2 (exact phrase match): {result2['status']}")
    print(f"  task_ref: {result2.get('task_reference')}")
    print(f"  blocker_ref: {result2.get('blocker_reference')}")
    print(f"  blocker_high_conf: {result2.get('blocker_high_confidence')}")
    print()


if __name__ == "__main__":
    test_new_logic()