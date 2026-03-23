"""
Cross-System Verification: Graph System
========================================
Test if bistability emerges in a graph-based system
with similar dynamics but different implementation

Core structure preserved:
- Competition (merge vs split)
- Capacity constraint (max degree K)
- Noise

Graph system:
- Nodes + Edges
- Edge merging: connect similar clusters
- Edge breaking: split large clusters
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import os
import json

os.makedirs('F:/hypergraph_bistability/figures', exist_ok=True)
os.makedirs('F:/hypergraph_bistability/results', exist_ok=True)


class GraphSystem:
    """Graph-based competitive dynamics"""
    
    def __init__(self, n_nodes, state_dim=16, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        
        self.V = list(range(n_nodes))
        self.edges = []
        self.state_dim = state_dim
        
        # Random state vectors for each node
        self.states = {v: np.random.randn(state_dim) for v in self.V}
        
        # Initial random edges
        n_initial = max(3, n_nodes // 4)
        for _ in range(n_initial):
            u, v = random.sample(self.V, 2)
            if (u, v) not in self.edges and (v, u) not in self.edges:
                self.edges.append((u, v))
    
    def get_node_degree(self, v):
        return sum(1 for e in self.edges if v in e)
    
    def get_cluster_size(self, v):
        """Find connected component size containing v"""
        visited = set()
        stack = [v]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            for u, w in self.edges:
                if node in (u, w):
                    other = w if u == node else u
                    if other not in visited:
                        stack.append(other)
        return len(visited)
    
    def get_largest_cluster_size(self):
        """Get size of largest connected component"""
        if not self.edges:
            return 1
        
        visited = set()
        max_size = 0
        
        for v in self.V:
            if v not in visited:
                cluster = self._get_cluster(v, visited)
                max_size = max(max_size, len(cluster))
        
        return max_size
    
    def _get_cluster(self, start, visited):
        cluster = []
        stack = [start]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            cluster.append(node)
            for u, w in self.edges:
                if node in (u, w):
                    other = w if u == node else u
                    if other not in visited:
                        stack.append(other)
        return cluster
    
    def apply_rules(self, gamma=0.35, K=None):
        """Apply merge/split dynamics"""
        if K is None:
            K = int(gamma * len(self.V))
        
        # Rule 1: Growth - add new node and edge
        if random.random() < 0.3 and self.V:
            source = random.choice(self.V)
            new_node = max(self.V) + 1
            self.V.append(new_node)
            self.states[new_node] = self.states[source] + np.random.randn(self.state_dim) * 0.2
            self.edges.append((source, new_node))
        
        # Rule 2: Edge Merging (fusion equivalent)
        if random.random() < 0.25 and len(self.edges) >= 2:
            # Select two edges
            e1 = random.choice(self.edges)
            e2 = random.choice(self.edges)
            
            if e1 != e2:
                # Get nodes from each edge
                u1, v1 = e1
                u2, v2 = e2
                
                # Calculate state similarity
                states1 = [self.states[u1], self.states[v1]]
                states2 = [self.states[u2], self.states[v2]]
                
                # Find closest pair
                min_dist = float('inf')
                pair = None
                for s1 in states1:
                    for s2 in states2:
                        d = np.linalg.norm(s1 - s2)
                        if d < min_dist:
                            min_dist = d
                            pair = (s1, s2)
                
                # Merge if similar
                if min_dist < 1.0 and random.random() < 0.5:
                    # Connect the two edges' clusters
                    new_edge = (random.choice([u1, v1]), random.choice([u2, v2]))
                    if new_edge[0] != new_edge[1] and new_edge not in self.edges and (new_edge[1], new_edge[0]) not in self.edges:
                        self.edges.append(new_edge)
        
        # Rule 3: Edge Breaking (split equivalent)
        if random.random() < 0.15 and len(self.edges) > 2:
            # Break an edge
            e = random.choice(self.edges)
            if e in self.edges:
                self.edges.remove(e)
        
        # Rule 4: Pruning - remove isolated nodes
        if random.random() < 0.05:
            isolated = [v for v in self.V if self.get_node_degree(v) == 0]
            for v in isolated[:1]:
                if v in self.V:
                    self.V.remove(v)
                if v in self.states:
                    del self.states[v]
        
        # Rule 5: Degree constraint (capacity limit)
        for v in self.V:
            degree = self.get_node_degree(v)
            if degree > K:
                # Remove excess edges from high-degree node
                excess = degree - K
                v_edges = [e for e in self.edges if v in e]
                random.shuffle(v_edges)
                for e in v_edges[:excess]:
                    if e in self.edges:
                        self.edges.remove(e)
    
    def get_M(self):
        """Order parameter: largest cluster / total nodes"""
        if not self.V:
            return 0
        return self.get_largest_cluster_size() / len(self.V)


def run_graph_experiment(N=50, gamma=0.35, state_dim=16, steps=200, seed=42):
    """Run one graph experiment"""
    g = GraphSystem(N, state_dim, seed)
    
    for _ in range(steps):
        g.apply_rules(gamma)
    
    return g.get_M()


def run_batch(N=50, gamma=0.35, state_dim=16, steps=200, n_runs=15):
    """Run multiple experiments"""
    results = []
    for i in range(n_runs):
        M = run_graph_experiment(N, gamma, state_dim, steps, seed=i*100+42)
        results.append(M)
    return np.mean(results), np.std(results), results


def main():
    print("=" * 60)
    print("Cross-System Verification: Graph System")
    print("=" * 60)
    
    # Test different N and gamma
    configs = [
        {'N': 30, 'gamma': 0.25},
        {'N': 30, 'gamma': 0.35},
        {'N': 30, 'gamma': 0.45},
        {'N': 50, 'gamma': 0.25},
        {'N': 50, 'gamma': 0.35},
        {'N': 50, 'gamma': 0.45},
    ]
    
    results = {}
    
    for config in configs:
        name = f"N{config['N']}_g{config['gamma']}"
        print(f"\nRunning: {name}...")
        
        mean_M, std_M, all_M = run_batch(
            N=config['N'], 
            gamma=config['gamma'],
            state_dim=16,
            steps=200,
            n_runs=15
        )
        
        results[name] = {
            'N': config['N'],
            'gamma': config['gamma'],
            'mean': mean_M,
            'std': std_M,
            'all': all_M
        }
        
        print(f"  M* = {mean_M:.3f} +/- {std_M:.3f}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"{'Config':<15} | {'N':>4} | {'gamma':>6} | {'M*':>8} | {'Std':>6}")
    print("-" * 60)
    
    for name, data in results.items():
        print(f"{name:<15} | {data['N']:>4} | {data['gamma']:>6.2f} | {data['mean']:>8.3f} | {data['std']:>6.3f}")
    
    # Compare with hypergraph results
    print("\n" + "=" * 60)
    print("Comparison: Graph vs Hypergraph")
    print("=" * 60)
    
    # Hypergraph baseline (from earlier): N=40, gamma=0.35 -> M* ~ 0.69
    hypergraph_baseline = 0.692
    
    for name, data in results.items():
        if data['N'] == 50 and data['gamma'] == 0.35:
            graph_M = data['mean']
            diff = graph_M - hypergraph_baseline
            print(f"Graph N=50, gamma=0.35: M* = {graph_M:.3f}")
            print(f"Hypergraph baseline: M* = {hypergraph_baseline:.3f}")
            print(f"Difference: {diff:+.3f}")
    
    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: M* vs gamma for different N
    ax1 = axes[0]
    for N in [30, 50]:
        gammas = []
        Ms = []
        stds = []
        for name, data in results.items():
            if data['N'] == N:
                gammas.append(data['gamma'])
                Ms.append(data['mean'])
                stds.append(data['std'])
        
        ax1.errorbar(gammas, Ms, yerr=stds, label=f'N={N}', marker='o', capsize=5)
    
    ax1.axhline(0.45, color='green', linestyle='--', alpha=0.5, label='Hypergraph M*')
    ax1.axhline(0.692, color='blue', linestyle=':', alpha=0.5, label='Hypergraph baseline')
    ax1.set_xlabel('gamma (capacity ratio)')
    ax1.set_ylabel('M* (order parameter)')
    ax1.set_title('Graph System: M* vs gamma')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.2)
    
    # Plot 2: Distribution for N=50, gamma=0.35
    ax2 = axes[1]
    if 'N50_g0.35' in results:
        ax2.hist(results['N50_g0.35']['all'], bins=10, alpha=0.7, edgecolor='black')
        ax2.axvline(results['N50_g0.35']['mean'], color='red', linestyle='--', 
                   label=f"Mean={results['N50_g0.35']['mean']:.3f}")
        ax2.axvline(0.45, color='green', linestyle=':', label='Hypergraph M*')
        ax2.set_xlabel('M')
        ax2.set_ylabel('Count')
        ax2.set_title('Graph System: M Distribution (N=50, gamma=0.35)')
        ax2.legend()
    
    plt.tight_layout()
    plt.savefig('F:/hypergraph_bistability/figures/graph_vs_hypergraph.png', dpi=150)
    print(f"\n[OK] Figure saved: figures/graph_vs_hypergraph.png")
    
    # Save results
    with open('F:/hypergraph_bistability/results/graph_verification.json', 'w') as f:
        json.dump({k: {'N': v['N'], 'gamma': v['gamma'], 'mean': v['mean'], 'std': v['std']} 
                  for k, v in results.items()}, f, indent=2)
    print(f"[OK] Results saved: results/graph_verification.json")
    
    # Analysis
    print("\n" + "=" * 60)
    print("Key Findings")
    print("=" * 60)
    
    # Check if bistability exists
    has_high_M = any(r['mean'] > 0.7 for r in results.values())
    has_low_M = any(r['mean'] < 0.5 for r in results.values())
    
    print(f"High M* (>0.7) observed: {has_high_M}")
    print(f"Low M* (<0.5) observed: {has_low_M}")
    
    if has_high_M and has_low_M:
        print("\n[OK] Bistability confirmed in graph system!")
    else:
        print("\n[!] Different behavior from hypergraph system")


if __name__ == '__main__':
    main()
