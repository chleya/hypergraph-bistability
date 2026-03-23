#!/usr/bin/env python
"""Compare strong_pass changes between old and new validator."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hypergraph_bistability.agent.runtime.context_assembler import ContextAssembler


def compare_strong_changes():
    """Compare which samples changed strong_pass status."""
    # Load fixture
    fixture_path = os.path.join(
        os.path.dirname(__file__), 
        "..", 
        "fixtures", 
        "handoff_validation_round1.json"
    )
    
    with open(fixture_path, "r", encoding="utf-8") as f:
        fixture = json.load(f)
    
    assembler = ContextAssembler()
    
    print("\n" + "="*80)
    print("STRONG_PASS CHANGES ANALYSIS")
    print("="*80)
    
    # Old strong_pass samples
    old_strong = [s for s in fixture["samples"] if s["validator_status"] == "strong_pass"]
    print(f"\nOld strong_pass count: {len(old_strong)}")
    
    # Run new validator on these samples
    strong_became_weak = 0
    strong_stayed_strong = 0
    
    for sample in old_strong:
        result = assembler.validate_handoff_continuity(sample["response"], sample["handoff_bundle"])
        new_status = result["status"]
        
        if new_status == "strong_pass":
            strong_stayed_strong += 1
            print(f"\n  {sample['sample_id']}: STAYED strong_pass (human={sample['human_label']})")
        else:
            strong_became_weak += 1
            print(f"\n  {sample['sample_id']}: became {new_status} (human={sample['human_label']})")
            print(f"    task_ref: {result.get('task_reference')}")
            print(f"    blocker_high_conf: {result.get('blocker_high_confidence')}")
            print(f"    next_step_high_conf: {result.get('next_step_high_confidence')}")
    
    print(f"\n\nSummary:")
    print(f"  Old strong -> stayed strong: {strong_stayed_strong}")
    print(f"  Old strong -> became weak/fail: {strong_became_weak}")
    
    # Now check new strong_pass (including samples that weren't strong before)
    print("\n" + "-"*40)
    print("New strong_pass samples:")
    
    new_strong_samples = []
    for sample in fixture["samples"]:
        result = assembler.validate_handoff_continuity(sample["response"], sample["handoff_bundle"])
        if result["status"] == "strong_pass":
            new_strong_samples.append((sample["sample_id"], sample["human_label"]))
            was_old = sample["validator_status"] == "strong_pass"
            print(f"  {sample['sample_id']}: {was_old and 'STAYED' or 'NEW'} strong (human={sample['human_label']})")


if __name__ == "__main__":
    compare_strong_changes()