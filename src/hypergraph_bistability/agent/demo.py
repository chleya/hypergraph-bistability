"""
Demo script for HypergraphAgent
================================

Usage:
    python -m hypergraph_bistability.agent.demo
"""

from hypergraph_bistability.agent.hypergraph_agent import HypergraphAgent


def demo():
    """Demo of HypergraphAgent."""
    print("=" * 60)
    print("HypergraphAgent Demo")
    print("=" * 60)
    
    agent = HypergraphAgent(k=4, L=2, use_embeddings=False)
    agent.group_labels = ["work", "personal", "technical", "creative"]
    
    print("\n1. Testing adaptive mode switching:")
    
    scenarios = [
        "Let's brainstorm some startup ideas",
        "I need to decide between job offers",
        "Help me debug this Python code",
        "Write me a short poem",
    ]
    
    for input_text in scenarios:
        response = agent.chat(input_text)
        state = agent.get_memory_state()
        print(f"\nInput: {input_text}")
        print(f"Mode: {state['controller']['mode']}, λ={state['memory']['lambda']:.4f}")
        print(f"Response: {response[:80]}...")
    
    print("\n" + agent.visualize_state())
    
    print("\n2. Testing conflict mitigation:")
    response = agent.chat("I want to go to the beach but I also want to finish this project")
    state = agent.get_memory_state()
    print(f"\nInput: I want to go to the beach but I also want to finish this project")
    print(f"Mode: {state['controller']['mode']}")
    print(f"Response: {response[:100]}...")


if __name__ == "__main__":
    demo()
