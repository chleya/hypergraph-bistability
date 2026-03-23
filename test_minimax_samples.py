#!/usr/bin/env python
"""Run agent with real MiniMax to collect validation samples."""
import os
import sys

# Set API key directly
os.environ["MINIMAX_API_KEY"] = "sk-cp-32X4yB3hv4uMfzdBmke7EyaxE2pXmHkAGisoBxm1bTlSnUKXcH3lGRgWYcD62Nre5AacJpbi0E5yOx92m5rkIth9HioW2aCHP5r3LeCKBuf-wdr1TVgeFxY"

print("Testing MiniMax API with task context...")

try:
    from hypergraph_bistability import HypergraphAgent
    
    # Create agent
    agent = HypergraphAgent(
        llm_model="abab6.5s-chat",
        llm_base_url="https://api.minimax.chat/v1",
        llm_api_key=os.environ["MINIMAX_API_KEY"],
    )
    print("Agent created")
    
    # Set task context manually
    agent._current_linked_task = "Fix login bug"
    agent._handoff_snapshot = {
        "linked_task": "Fix login bug",
        "blockers": ["API timeout issue", "Missing error handling"],
        "next_steps": ["Add retry logic", "Update error messages"],
        "active_decisions": ["Use exponential backoff"],
        "applicable_procedures": [],
    }
    
    # Test: With task context
    print("\n=== Test: With task context ===")
    result = agent.process_turn("What are we working on?")
    response = result.assistant_response if hasattr(result, 'assistant_response') else str(result)
    print(f"Response: {response[:200]}...")
    
    val = agent.turn_processor._last_handoff_validation
    print(f"Validation: {val}")
    
    # Test 2: Another with task context - more specific
    print("\n=== Test 2: What should we do next? ===")
    agent._current_linked_task = "Fix login bug"
    agent._handoff_snapshot = {
        "linked_task": "Fix login bug",
        "blockers": ["API timeout issue", "Missing error handling"],
        "next_steps": ["Add retry logic", "Update error messages"],
        "active_decisions": ["Use exponential backoff"],
        "applicable_procedures": [],
    }
    result2 = agent.process_turn("What should we do next?")
    response2 = result2.assistant_response if hasattr(result2, 'assistant_response') else str(result2)
    print(f"Response: {response2[:200]}...")
    
    val2 = agent.turn_processor._last_handoff_validation
    print(f"Validation: {val2}")
    
    print("\n=== Done ===")
    print(f"Check logs/handoff_validation.jsonl for recorded results")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
