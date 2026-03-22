import sys
sys.path.insert(0, 'F:/hypergraph_bistability/src')
from hypergraph_bistability.memory.unified_node import UnifiedNodeManager

m = UnifiedNodeManager('F:/hypergraph_bistability/unified_memory.db')

# Get all tables
tables = m.store.conn.execute('SELECT name FROM sqlite_master WHERE type=?', ('table',)).fetchall()
print('Tables:', [t[0] for t in tables])

# Get all skill nodes
all_nodes = m.store.conn.execute('SELECT id, node_type, content FROM nodes').fetchall()
print('\nAll nodes in DB:')
print('='*50)
for r in all_nodes:
    print(f'ID {r[0]}: [{r[1]}] {r[2][:40]}')
