#!/usr/bin/env python
"""Calculate actual agreement with human labels using new logic."""
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hypergraph_bistability.agent.runtime.context_assembler import ContextAssembler


def calculate_real_agreement():
    with open("fixtures/handoff_validation_round1.json", "r", encoding="utf-8") as f:
        fixture = json.load(f)
    
    assembler = ContextAssembler()
    
    correct = 0
    partial = 0
    wrong = 0
    
    print("Sample-by-sample comparison:")
    print("="*60)
    
    for sample in fixture["samples"]:
        result = assembler.validate_handoff_continuity(sample["response"], sample["handoff_bundle"])
        new_status = result["status"]
        human = sample["human_label"]
        
        # Agreement logic
        if new_status == human:
            correct += 1
            status = "OK"
        elif new_status == "weak_pass" and human == "marginal":
            partial += 1
            status = "PARTIAL"
        else:
            wrong += 1
            status = "WRONG"
        
        if status != "OK":
            print(f"{sample['sample_id']}: {new_status} vs {human} [{status}]")
    
    total = len(fixture["samples"])
    print("="*60)
    print(f"Correct: {correct}/{total} ({100*correct/total:.1f}%)")
    print(f"Partial: {partial}/{total}")
    print(f"Wrong: {wrong}/{total}")
    
    # Break down by status
    print("\nBy status breakdown:")
    for status in ["fail", "weak_pass", "strong_pass"]:
        matches = sum(1 for s in fixture["samples"] 
                     if (result := assembler.validate_handoff_continuity(s["response"], s["handoff_bundle"])) 
                     and result["status"] == status)
        print(f"  {status}: {matches}")


if __name__ == "__main__":
    calculate_real_agreement()