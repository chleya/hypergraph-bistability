#!/usr/bin/env python
"""Analyze handoff validation results manually."""
import json

# Read log file
with open('src/logs/handoff_validation.jsonl', 'r') as f:
    lines = f.readlines()

print(f"Total samples: {len(lines)}\n")

# Parse each line
samples = []
for i, line in enumerate(lines):
    data = json.loads(line)
    samples.append({
        'id': i,
        'timestamp': data.get('timestamp', ''),
        'status': data.get('status', ''),
        'reason': data.get('reason', ''),
        'task_ref': data.get('task_reference', False),
        'blocker_ref': data.get('blocker_reference', False),
        'next_step_ref': data.get('next_step_reference', False),
        'decision_ref': data.get('decision_reference', False),
        'quality': data.get('content_quality', ''),
        'response_len': data.get('response_length', 0),
        'response_preview': data.get('response_preview', '')[:150],
    })

# Show all samples with their status
print("=" * 80)
print("ALL SAMPLES FROM LOG")
print("=" * 80)

for s in samples:
    print(f"\n--- Sample {s['id']} ---")
    print(f"Status: {s['status']} ({s['reason']})")
    print(f"Refs: task={s['task_ref']}, blocker={s['blocker_ref']}, next_step={s['next_step_ref']}, decision={s['decision_ref']}")
    print(f"Quality: {s['quality']}, Length: {s['response_len']}")
    print(f"Response: {s['response_preview']}...")
