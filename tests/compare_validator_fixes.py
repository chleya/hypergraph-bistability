#!/usr/bin/env python
"""Run validator on fixture samples and compare with ground truth."""
import json
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hypergraph_bistability.agent.runtime.context_assembler import ContextAssembler


def run_validator_on_fixture():
    """Run the validator on fixture samples and compare results."""
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
    print("VALIDATOR FIX COMPARISON - RUNNING ON FIXTURE")
    print("="*80)
    
    improved = 0
    same = 0
    worse = 0
    
    for sample in fixture["samples"]:
        sample_id = sample["sample_id"]
        handoff_bundle = sample["handoff_bundle"]
        response = sample["response"]
        human_label = sample["human_label"]
        old_validator_status = sample["validator_status"]
        
        # Run validator
        result = assembler.validate_handoff_continuity(response, handoff_bundle)
        new_status = result["status"]
        
        # Compare
        old_correct = (old_validator_status == human_label or 
                     (old_validator_status == "weak_pass" and human_label == "marginal"))
        new_correct = (new_status == human_label or
                      (new_status == "weak_pass" and human_label == "marginal"))
        
        if new_correct and not old_correct:
            status_change = "[IMPROVED]"
            improved += 1
        elif old_correct and not new_correct:
            status_change = "[WORSE]"
            worse += 1
        else:
            status_change = "[SAME]"
            same += 1
        
        if new_status != old_validator_status:
            print(f"\n{sample_id}: {old_validator_status} -> {new_status} ({human_label}) {status_change}")
            if "error_reason" in sample:
                print(f"   Reason: {sample['error_reason']}")
            print(f"   Task ref: {result.get('task_reference')}")
            print(f"   Blocker ref: {result.get('blocker_reference')}")
            print(f"   Next step ref: {result.get('next_step_reference')}")
    
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total samples: {len(fixture['samples'])}")
    print(f"[IMPROVED]: {improved}")
    print(f"[SAME]: {same}")
    print(f"[WORSE]: {worse}")
    
    # Calculate new agreement rate
    correct = sum(1 for s in fixture["samples"] 
                 if result.get("status") == s["human_label"] or
                 (result.get("status") == "weak_pass" and s["human_label"] == "marginal"))
    
    new_agreement_rate = correct / len(fixture["samples"]) * 100
    old_agreement_rate = fixture["summary"]["agreement_rate"].replace("%", "")
    
    print(f"\nAgreement Rate:")
    print(f"  Before (round 1): {old_agreement_rate}%")
    print(f"  After (with fixes): {new_agreement_rate:.1f}%")
    print(f"  Change: {new_agreement_rate - float(old_agreement_rate):+.1f}%")
    
    return {
        "improved": improved,
        "same": same,
        "worse": worse,
        "new_agreement_rate": new_agreement_rate
    }


if __name__ == "__main__":
    run_validator_on_fixture()
