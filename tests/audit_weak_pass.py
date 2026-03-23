#!/usr/bin/env python
"""Audit weak_pass internals to see composition."""
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hypergraph_bistability.agent.runtime.context_assembler import ContextAssembler


def audit_weak_pass():
    with open("fixtures/handoff_validation_round1.json", "r", encoding="utf-8") as f:
        fixture = json.load(f)
    
    assembler = ContextAssembler()
    
    # Get all weak_pass samples
    weak_samples = [s for s in fixture["samples"] if s["human_label"] in ["fail", "weak_pass", "marginal"]]
    
    print("="*80)
    print("WEAK_PASS INTERNAL AUDIT")
    print("="*80)
    
    for i, sample in enumerate(weak_samples):
        result = assembler.validate_handoff_continuity(sample["response"], sample["handoff_bundle"])
        
        human = sample["human_label"]
        validator = result["status"]
        
        # Determine if this is correctly classified or misclassified
        if human == "fail" and validator == "weak_pass":
            classification = "SHOULD BE FAIL"
        elif human == "strong_pass" and validator == "weak_pass":
            classification = "SHOULD BE STRONG"
        elif human == "marginal":
            classification = "BOUNDARY (marginal)"
        else:
            classification = "OK - true boundary"
        
        print(f"\n[{i+1}] {sample['sample_id']}")
        print(f"    Human: {human} | Validator: {validator}")
        print(f"    Classification: {classification}")
        print(f"    Reason: {result.get('reason', 'N/A')}")
        
        # Show what was matched
        print(f"    Task ref: {result.get('task_reference')}")
        print(f"    Blocker ref: {result.get('blocker_reference')} (high: {result.get('blocker_high_confidence')})")
        print(f"    Next step ref: {result.get('next_step_reference')} (high: {result.get('next_step_high_confidence')})")
        print(f"    Decision ref: {result.get('decision_reference')} (high: {result.get('decision_high_confidence')})")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    # Count
    should_be_fail = sum(1 for s in weak_samples 
                        if s["human_label"] == "fail" and assembler.validate_handoff_continuity(s["response"], s["handoff_bundle"])["status"] == "weak_pass")
    should_be_strong = sum(1 for s in weak_samples 
                          if s["human_label"] == "strong_pass" and assembler.validate_handoff_continuity(s["response"], s["handoff_bundle"])["status"] == "weak_pass")
    true_boundary = sum(1 for s in weak_samples if s["human_label"] in ["weak_pass", "marginal"])
    
    print(f"Total weak_pass samples: {len(weak_samples)}")
    print(f"  - Should be fail (misclassified): {should_be_fail}")
    print(f"  - Should be strong (misclassified): {should_be_strong}")
    print(f"  - True boundary: {true_boundary}")


if __name__ == "__main__":
    audit_weak_pass()