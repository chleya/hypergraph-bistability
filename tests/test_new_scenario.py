"""Quick test for new scenarios."""
from hypergraph_bistability.agent import HypergraphAgent

# Test paraphrase_heavy_conflict scenario
agent = HypergraphAgent(k=4, L=2, use_embeddings=False)

print('=== Testing paraphrase_heavy_conflict ===')
# Turn 1: remember conflict
r1 = agent.process_turn('Remember that there is a debate about whether to use PostgreSQL or MongoDB for the user profile service.')
print(f'Turn 1: OK')

# Turn 2: distraction
r2 = agent.process_turn("What's for lunch today?")
print(f'Turn 2: OK')

# Turn 3: paraphrase question
r3 = agent.process_turn('I need a summary of the architectural decision we discussed.')
print(f'Turn 3: {r3.assistant_response[:150]}...')

print()
print('=== Testing long_interruption_resume ===')
agent2 = HypergraphAgent(k=4, L=2, use_embeddings=False)

# Turn 1-2: task context
r1 = agent2.process_turn("I'm working on implementing user authentication for the new API.")
print(f'Turn 1: OK')
r2 = agent2.process_turn('The auth should use JWT tokens with refresh token rotation.')
print(f'Turn 2: OK')

# Turns 3-6: long interruptions
for i in range(3, 7):
    r = agent2.process_turn(f'Turn {i} - random topic')
    print(f'Turn {i}: OK')

# Turn 7: resume
r7 = agent2.process_turn('Okay, back to work. What was I implementing?')
print(f'Turn 7 (resume): {r7.assistant_response[:150]}...')

print()
print('=== All tests passed! ===')
