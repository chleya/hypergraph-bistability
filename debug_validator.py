#!/usr/bin/env python
"""Debug why task_reference is False."""
import os
import sys

os.environ["MINIMAX_API_KEY"] = "sk-cp-32X4yB3hv4uMfzdBmke7EyaxE2pXmHkAGisoBxm1bTlSnUKXcH3lGRgWYcD62Nre5AacJpbi0E5yOx92m5rkIth9HioW2aCHP5r3LeCKBuf-wdr1TVgeFxY"

print("Debugging task reference...")

from hypergraph_bistability import HypergraphAgent

agent = HypergraphAgent(
    llm_model="abab6.5s-chat",
    llm_base_url="https://api.minimax.chat/v1",
    llm_api_key=os.environ["MINIMAX_API_KEY"],
)

# Set task context
agent._current_linked_task = "Fix login bug"
agent._handoff_snapshot = {
    "linked_task": "Fix login bug",
    "blockers": ["API timeout issue", "Missing error handling"],
    "next_steps": ["Add retry logic", "Update error messages"],
    "active_decisions": ["Use exponential backoff"],
    "applicable_procedures": [],
}

# Get the handoff bundle
bundle = agent.query_handoff_bundle()
print(f"Handoff bundle: {bundle}")

# Check what the validator sees
from hypergraph_bistability.agent.runtime.context_assembler import ContextAssembler
assembler = ContextAssembler()

response = "We are currently working on fixing a login bug. The status of this task is unknown..."

result = assembler.validate_handoff_continuity(response, bundle)
print(f"\nValidation result: {result}")

# Debug the task matching
task_lower = bundle.get("linked_task", "").lower().replace("_", " ")
task_words = [w for w in task_lower.split() if len(w) > 3]
print(f"\nTask: {bundle.get('linked_task')}")
print(f"Task words (len>3): {task_words}")
print(f"Response: {response.lower()}")
print(f"Word matches: {sum(1 for tw in task_words if tw in response.lower())}")
