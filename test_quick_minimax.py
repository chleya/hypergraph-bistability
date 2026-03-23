#!/usr/bin/env python
"""Quick test to run agent with real MiniMax and collect validation samples."""
import os
import sys

# Set up MiniMax API
os.environ.setdefault("MINIMAX_API_KEY", os.environ.get("MINIMAX_API_KEY", ""))

# Check API key
api_key = os.environ.get("MINIMAX_API_KEY", "")
if not api_key or api_key == "your_api_key_here":
    print("No MINIMAX_API_KEY found!")
    print(f"Current: {api_key}")
    sys.exit(1)

print(f"API key found: {api_key[:20]}...")

# Try to create agent
try:
    from hypergraph_bistability import HypergraphAgent
    
    agent = HypergraphAgent(
        llm_api="minimax",
        llm_model="MiniMax-M2.1",
    )
    print("Agent created successfully")
    
    # Run a simple turn
    response = agent.process_turn("What is 2+2?")
    print(f"Response: {response[:100]}...")
    
    # Check validation result
    validation = agent.turn_processor._last_handoff_validation
    print(f"Validation: {validation}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
