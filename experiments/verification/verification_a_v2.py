"""
Path A v2: Coagulation Bias for Emergent Bistability
======================================================

Key insight: Current rules produce fragmentation because:
- Growth is random and always possible
- Splitting is common

To create bistability, we need:
- When M is LOW: splitting dominates (fragmentation)
- When M is HIGH: fusion dominates (coagulation)

This creates two stable basins: LOW state and HIGH state
"""

import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json

os.makedirs('F:/hypergraph_bistability/results/verification_a', exist_ok=True)


class CoagulationBiasedHypergraph:
    """
    Hypergraph with coagulation bias:
    - Fusion rate increases with M
    - Split rate decreases with M
    """
    
    def __init__(self, N=50, n_groups=1, gamma=0.35, state_dim=16,
                 fusion_boost=5.0, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        
        self.N = N
        self.n_groups = n_groups
        self.gamma = gamma
        self.state_dim = state_dim
        self.K = int(gamma * N)
        self.fusion_boost = fusion_boost
        
        self.group_assignments = {}
        nodes_per_group = N // n_groups
        for i in range(n_groups):
            start = i * nodes_per_group
            end = start + nodes_per_group if i < n_groups - 1 else N
            for v in range(start, end):
                self.group_assignments[v] = i
        
        self.V = list(range(N))
        self.s = {v: np.random.randn(state_dim) for v in self.V}
        
        self.E = []
        n_initial = max(4, N // 4)
        for _ in range(n_initial):
            group = random.randint(0, n_groups - 1)
            group_nodes = [v for v in self.V if self.group_assignments[v] == group]
            if len(group_nodes) >= 2:
                size = random.randint(2, min(5, len(group_nodes) // 2))
                e = frozenset(random.sample(group_nodes, size))
                self.E.append(e)
    
    def get_node_group(self, v):
        return self.group_assignments[v]
    
    def get_group_nodes(self, g):
        return [v for v in self.V if self.group_assignments[v] == g]
    
    def get_node_degree(self, v):
        return sum(1 for e in self.E if v in e)
    
    def get_group_order_parameter(self, g):
        group_nodes = self.get_group_nodes(g)
        if not group_nodes:
            return 0.0
        degrees = [self.get_node_degree(v) for v in group_nodes]
        return np.mean(degrees) / max(self.K, 1)
    
    def get_avg_distance(self, nodes):
        if len(nodes) < 2:
            return 0.1
        samples = min(20, len(nodes) * (len(nodes) - 1) // 2)
        if samples <= 0:
            return 0.1
        dist_sum = 0
        for _ in range(samples):
            u, v = random.sample(nodes, 2)
            dist_sum += np.linalg.norm(self.s[u] - self.s[v])
        return max(dist_sum / samples, 0.01)
    
    def apply_rules(self):
        avg_dist_overall = self.get_avg_distance(self.V)
        
        for g in range(self.n_groups):
            group_nodes = self.get_group_nodes(g)
            if not group_nodes:
                continue
            
            avg_dist_g = self.get_avg_distance(group_nodes)
            M_g = self.get_group_order_parameter(g)
            
            fusion_factor = 1.0 + self.fusion_boost * M_g
            split_factor = 1.0 - 0.5 * M_g
            
            # Rule 1: Growth
            if random.random() < 0.35 and self.E:
                group_edges = [e for e in self.E
                             if all(self.get_node_group(v) == g for v in e)]
                if group_edges:
                    e = random.choice(group_edges)
                    nodes = list(e)
                    w = len(self.V)
                    self.V.append(w)
                    self.s[w] = self.s[nodes[0]] + np.random.randn(self.state_dim) * 0.2
                    self.group_assignments[w] = g
                    new_e = frozenset([nodes[0], w])
                    self.E.append(new_e)
            
            # Rule 2: Fusion with bias
            if random.random() < 0.3 * fusion_factor and len(self.E) >= 2:
                e1, e2 = random.sample(self.E, 2)
                if len(e1 & e2) >= 1:
                    g1 = self.get_node_group(list(e1)[0])
                    g2 = self.get_node_group(list(e2)[0])
                    
                    if g1 == g2:
                        weights1 = np.array([1.0 / (1 + self.get_node_degree(v)) for v in e1])
                        weights1 = weights1 / weights1.sum()
                        c1 = np.average([self.s[u] for u in e1], axis=0, weights=weights1)
                        
                        weights2 = np.array([1.0 / (1 + self.get_node_degree(v)) for v in e2])
                        weights2 = weights2 / weights2.sum()
                        c2 = np.average([self.s[u] for u in e2], axis=0, weights=weights2)
                        
                        dist = np.linalg.norm(c1 - c2)
                        threshold = avg_dist_g * 0.8
                        
                        if dist < threshold:
                            new_e = frozenset(e1 | e2)
                            if new_e not in self.E:
                                self.E.append(new_e)
                                self.E.remove(e1)
                                self.E.remove(e2)
            
            # Rule 3: Split (reduced when M is high)
            if random.random() < 0.15 * max(0.1, split_factor) and self.E:
                group_edges = [e for e in self.E
                             if all(self.get_node_group(v) == g for v in e)]
                large_edges = [e for e in group_edges if len(e) > 2]
                if large_edges:
                    e = random.choice(large_edges)
                    nodes = list(e)
                    if len(nodes) >= 2:
                        split_point = len(nodes) // 2
                        e1 = frozenset(nodes[:split_point])
                        e2 = frozenset(nodes[split_point:])
                        if e1 not in self.E and e2 not in self.E:
                            self.E.append(e1)
                            self.E.append(e2)
                            self.E.remove(e)
            
            # Rule 4: Delete
            if random.random() < 0.2:
                small_edges = [e for e in self.E if len(e) == 2]
                if small_edges:
                    e = random.choice(small_edges)
                    self.E.remove(e)


def test_bistability(fusion_boost, n_trials=10):
    """Test if bistability emerges with given fusion_boost"""
    
    low_results = []
    high_results = []
    
    for trial in range(n_trials):
        seed = 42 + trial * 100
        
        # LOW initial condition
        hg_low = CoagulationBiasedHypergraph(
            N=50, n_groups=1, gamma=0.35,
            fusion_boost=fusion_boost, seed=seed
        )
        hg_low.E = hg_low.E[:3]
        for _ in range(120):
            hg_low.apply_rules()
        M_low = hg_low.get_group_order_parameter(0)
        low_results.append(M_low)
        
        # HIGH initial condition
        hg_high = CoagulationBiasedHypergraph(
            N=50, n_groups=1, gamma=0.35,
            fusion_boost=fusion_boost, seed=seed
        )
        for _ in range(50):
            group_nodes = hg_high.get_group_nodes(0)
            if len(group_nodes) >= 2:
                size = random.randint(5, min(15, len(group_nodes)))
                e = frozenset(random.sample(group_nodes, size))
                if e not in hg_high.E:
                    hg_high.E.append(e)
        for _ in range(120):
            hg_high.apply_rules()
        M_high = hg_high.get_group_order_parameter(0)
        high_results.append(M_high)
    
    return {
        'low_mean': np.mean(low_results),
        'low_std': np.std(low_results),
        'high_mean': np.mean(high_results),
        'high_std': np.std(high_results),
        'bistable': np.mean(high_results) > np.mean(low_results) * 2
    }


if __name__ == '__main__':
    print("=" * 60)
    print("Test: Emergent Bistability with Coagulation Bias")
    print("=" * 60)
    
    results = {}
    for fusion_boost in [0.0, 2.0, 5.0, 10.0, 20.0]:
        print(f"\nfusion_boost = {fusion_boost}:")
        r = test_bistability(fusion_boost, n_trials=10)
        results[fusion_boost] = r
        print(f"  LOW initial -> M = {r['low_mean']:.4f} +/- {r['low_std']:.4f}")
        print(f"  HIGH initial -> M = {r['high_mean']:.4f} +/- {r['high_std']:.4f}")
        print(f"  Bistable: {r['bistable']}")
    
    # Save results
    with open('F:/hypergraph_bistability/results/verification_a/coagulation_bias_test.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for boost, r in results.items():
        status = "BISTABLE" if r['bistable'] else "monostable"
        print(f"fusion_boost={boost}: {status} (M_low={r['low_mean']:.3f}, M_high={r['high_mean']:.3f})")
