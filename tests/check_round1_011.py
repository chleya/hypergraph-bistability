#!/usr/bin/env python
"""Check what round1_011 used to match for strong."""
import json

with open("fixtures/handoff_validation_round1.json", "r", encoding="utf-8") as f:
    fixture = json.load(f)

sample = [s for s in fixture["samples"] if s["sample_id"] == "round1_011"][0]

print("Sample: round1_011")
print(f"Old validator status: {sample['validator_status']}")
print(f"Human label: {sample['human_label']}")
print(f"Old validator details: {sample.get('validator_details', 'N/A')}")
print()
print("Handoff:")
print(f"  linked_task: {sample['handoff_bundle'].get('linked_task')}")
print(f"  blockers: {sample['handoff_bundle'].get('blockers')}")
print(f"  next_steps: {sample['handoff_bundle'].get('next_steps')}")
print(f"  decisions: {sample['handoff_bundle'].get('active_decisions')}")
print()
print("Response (full):")
print(sample['response'][:500])