#!/usr/bin/env python
"""Debug why round1_003 is being classified as strong_pass."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hypergraph_bistability.agent.runtime.context_assembler import ContextAssembler


def debug_sample(sample_id):
    # Load fixture
    with open("fixtures/handoff_validation_round1.json", "r", encoding="utf-8") as f:
        fixture = json.load(f)
    
    sample = [s for s in fixture["samples"] if s["sample_id"] == sample_id][0]
    
    assembler = ContextAssembler()
    result = assembler.validate_handoff_continuity(sample["response"], sample["handoff_bundle"])
    
    print(f"Sample: {sample_id}")
    print(f"Human label: {sample['human_label']}")
    print(f"Old validator status: {sample['validator_status']}")
    print(f"New validator status: {result['status']}")
    print()
    print("Handoff bundle:")
    print(f"  linked_task: {sample['handoff_bundle'].get('linked_task')}")
    print(f"  blockers: {sample['handoff_bundle'].get('blockers')}")
    print(f"  next_steps: {sample['handoff_bundle'].get('next_steps')}")
    print()
    print("Response (first 200 chars):")
    print(sample["response"][:200])
    print()
    print("Validation result:")
    print(f"  task_reference: {result.get('task_reference')}")
    print(f"  blocker_reference: {result.get('blocker_reference')}")
    print(f"  blocker_high_conf: {result.get('blocker_high_confidence')}")
    print(f"  next_step_reference: {result.get('next_step_reference')}")
    print(f"  next_step_high_conf: {result.get('next_step_high_confidence')}")
    print(f"  decision_reference: {result.get('decision_reference')}")
    print(f"  decision_high_conf: {result.get('decision_high_confidence')}")
    print(f"  content_quality: {result.get('content_quality')}")


if __name__ == "__main__":
    debug_sample("round1_003")
    print("\n" + "="*60)
    debug_sample("round1_011")