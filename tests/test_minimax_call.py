#!/usr/bin/env python
"""Simple test to call MiniMax API via the agent."""

import os
import sys

# Set API key
os.environ['ANTHROPIC_API_KEY'] = 'sk-cp-32X4yB3hv4uMfzdBmke7EyaxE2pXmHkAGisoBxm1bTlSnUKXcH3lGRgWYcD62Nre5AacJpbi0E5yOx92m5rkIth9HioW2aCHP5r3LeCKBuf-wdr1TVgeFxY'

print("=" * 50)
print("Testing MiniMax API via Agent")
print("=" * 50)

# Import after setting env
from hypergraph_bistability.agent import HypergraphAgent

# Create agent
agent = HypergraphAgent(
    k=3, 
    L=2, 
    use_embeddings=False,
    llm_base_url='https://api.minimaxi.com/anthropic',
    llm_model='MiniMax-M2.7',
    llm_force_powershell_transport=True
)

print(f"\nAgent created:")
print(f"  - Model: {agent.llm_model}")
print(f"  - Base URL: {agent.llm_base_url}")
print(f"  - Transport: {agent.llm_transport}")

# Test single turn
print("\n--- Testing single turn ---")
try:
    result = agent.process_turn("Hi, how are you?")
    print(f"User: {result.user_input}")
    print(f"Assistant: {result.assistant_response[:200]}...")
    print(f"Retrieved: {len(result.retrieved_items)} items")
    print(f"Writes: {len(result.writes)} items")
    print("\n[OK] Single turn test PASSED!")
except Exception as e:
    print(f"\n[X] Error: {e}")
    import traceback
    traceback.print_exc()

# Test multi-turn
print("\n--- Testing multi-turn conversation ---")
try:
    response1 = agent.chat("My name is Alice.")
    print(f"Turn 1: {response1[:100]}...")
    
    response2 = agent.chat("What's my name?")
    print(f"Turn 2: {response2[:100]}...")
    
    print("\n[OK] Multi-turn test PASSED!")
except Exception as e:
    print(f"\n[X] Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 50)
print("Test Complete")
print("=" * 50)
