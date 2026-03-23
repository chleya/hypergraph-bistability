"""Check what's in save state."""
import json
from hypergraph_bistability import HypergraphAgent

agent = HypergraphAgent(llm_model='gpt-4o-mini')
agent.process_turn('Task: implement login feature')
agent.process_turn('Blocker: need OAuth credentials')

# Check what gets saved
save_data = agent._state
print('Keys in save:', list(save_data.keys()))
print('task:', save_data.get('task'))
print('memory keys:', list(save_data.get('memory', {}).keys()) if save_data.get('memory') else 'None')
