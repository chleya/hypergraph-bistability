"""
Edge Tracker with Parent Chain for Rigorous Identity Tracking
============================================================

This module implements a hypergraph where every edge tracks its true origin:
- edge_id: unique identifier
- birth_time: simulation step when edge was created
- death_time: step when edge was deleted (None if alive)
- nodes: frozenset of node IDs in this edge
- edge_type: "growth", "split", or "fusion"
- parents: list of parent edge IDs that this edge originated from

Key principle: An edge is "new" (in-situ) if its edge_type is "growth" or "split".
An edge is "fusion" if it came from merging two parent edges.
"""

import numpy as np
import random
from typing import List, Dict, Optional, Tuple, Set


class Edge:
    """Edge with full ancestry tracking."""

    def __init__(
        self,
        edge_id: int,
        birth_time: int,
        nodes: frozenset,
        edge_type: str,
        parents: List[int] = None
    ):
        self.id = edge_id
        self.birth_time = birth_time
        self.death_time = None
        self.nodes = nodes
        self.edge_type = edge_type  # "growth", "split", "fusion"
        self.parents = parents or []  # parent edge IDs

    def is_alive(self) -> bool:
        return self.death_time is None

    def get_ancestry_depth(self, edge_history: Dict[int, 'Edge']) -> int:
        """Recursively trace ancestry back to root edges."""
        if not self.parents:
            return 0  # This is a root edge
        max_depth = 0
        for parent_id in self.parents:
            if parent_id in edge_history:
                parent = edge_history[parent_id]
                depth = 1 + parent.get_ancestry_depth(edge_history)
                max_depth = max(max_depth, depth)
        return max_depth

    def __repr__(self):
        return f"Edge(id={self.id}, type={self.edge_type}, nodes={len(self.nodes)}, birth={self.birth_time})"


class HypergraphIdentityTracker:
    """Hypergraph with rigorous edge identity tracking."""

    def __init__(self, N: int = 50, p_pair: float = 0.5, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)

        self.N = N
        self.p_pair = p_pair
        self.K = int(0.35 * N)
        self.next_edge_id = 0

        self.V = list(range(N))
        self.edges: Dict[int, Edge] = {}
        self.time = 0

        for _ in range(15):
            if random.random() < p_pair:
                nodes = frozenset(random.sample(self.V, 2))
            else:
                nodes = frozenset(random.sample(self.V, min(3, N)))
            e = Edge(
                edge_id=self.next_edge_id,
                birth_time=0,
                nodes=nodes,
                edge_type="initial",
                parents=[]
            )
            self.edges[self.next_edge_id] = e
            self.next_edge_id += 1

    def get_node_degree(self, v: int) -> int:
        return sum(1 for e in self.edges.values() if v in e.nodes and e.is_alive())

    def get_alive_edges(self) -> List[Edge]:
        return [e for e in self.edges.values() if e.is_alive()]

    def get_clusters(self) -> Tuple[List[Set[int]], Set[int]]:
        """Return (clusters, max_cluster)."""
        alive_edges = self.get_alive_edges()
        if not self.V or not alive_edges:
            return [], set()

        adj = {v: set() for v in self.V}
        for e in alive_edges:
            nodes = list(e.nodes)
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    adj[nodes[i]].add(nodes[j])
                    adj[nodes[j]].add(nodes[i])

        visited = set()
        clusters = []

        for s in self.V:
            if s in visited:
                continue
            stack = [s]
            c = set()
            while stack:
                n = stack.pop()
                if n in visited:
                    continue
                visited.add(n)
                c.add(n)
                if n in adj:
                    for nb in adj[n]:
                        if nb not in visited:
                            stack.append(nb)
            clusters.append(c)

        if not clusters:
            return [], set()

        max_cluster = max(clusters, key=len)
        return clusters, max_cluster

    def get_core_edges(self) -> List[Edge]:
        """Get edges that belong to the maximum cluster (core)."""
        _, max_cluster = self.get_clusters()
        if not max_cluster:
            return []
        alive_edges = self.get_alive_edges()
        core_edges = [e for e in alive_edges if e.nodes & max_cluster]
        return core_edges

    def compute_H(self, node_set: Set[int]) -> float:
        relevant_edges = [e for e in self.get_alive_edges() if len(e.nodes & node_set) > 0]
        if not relevant_edges:
            return 0.0
        sizes = [len(e.nodes) for e in relevant_edges]
        counts = {}
        for s in sizes:
            counts[s] = counts.get(s, 0) + 1
        total = len(relevant_edges)
        H = 0.0
        for c in counts.values():
            p = c / total
            if p > 0:
                H -= p * np.log2(p)
        return H

    def get_M(self) -> float:
        _, max_cluster = self.get_clusters()
        return len(max_cluster) / len(self.V) if self.V and max_cluster else 0.0

    def apply_rules(self, steps: int = 1):
        """Apply growth, fusion, split, deletion rules with full tracking."""
        for _ in range(steps):
            self._growth_rule()
            self._fusion_rule()
            self._split_rule()
            self._deletion_rule()
            self.time += 1

    def _growth_rule(self):
        """Add new node connected to existing edge."""
        if random.random() < 0.3 and self.edges:
            alive_edges = self.get_alive_edges()
            if not alive_edges:
                return
            parent = random.choice(alive_edges)
            v = random.choice(list(parent.nodes))
            w = max(self.V) + 1
            self.V.append(w)

            if random.random() < self.p_pair:
                new_nodes = frozenset([v, w])
            else:
                new_nodes = frozenset([v, w, random.choice(list(parent.nodes) + [w])])

            new_edge = Edge(
                edge_id=self.next_edge_id,
                birth_time=self.time,
                nodes=new_nodes,
                edge_type="growth",
                parents=[parent.id]
            )
            self.edges[self.next_edge_id] = new_edge
            self.next_edge_id += 1

    def _fusion_rule(self):
        """Merge two edges that share nodes."""
        alive_edges = self.get_alive_edges()
        if len(alive_edges) < 2:
            return

        if random.random() < 0.25:
            e1, e2 = random.sample(alive_edges, 2)
            if len(e1.nodes & e2.nodes) >= 1 and random.random() < 0.5:
                new_nodes = e1.nodes | e2.nodes
                if len(new_nodes) >= 2:
                    e1.death_time = self.time
                    e2.death_time = self.time

                    new_edge = Edge(
                        edge_id=self.next_edge_id,
                        birth_time=self.time,
                        nodes=new_nodes,
                        edge_type="fusion",
                        parents=[e1.id, e2.id]
                    )
                    self.edges[self.next_edge_id] = new_edge
                    self.next_edge_id += 1

    def _split_rule(self):
        """Split large edge into two."""
        alive_edges = self.get_alive_edges()
        large_edges = [e for e in alive_edges if len(e.nodes) > 2]

        if random.random() < 0.12 and large_edges:
            parent = random.choice(large_edges)
            nodes_list = list(parent.nodes)
            if len(nodes_list) >= 4:
                split = len(nodes_list) // 2
                parent.death_time = self.time

                child1_nodes = frozenset(nodes_list[:split])
                child1 = Edge(
                    edge_id=self.next_edge_id,
                    birth_time=self.time,
                    nodes=child1_nodes,
                    edge_type="split",
                    parents=[parent.id]
                )
                self.edges[self.next_edge_id] = child1
                self.next_edge_id += 1

                child2_nodes = frozenset(nodes_list[split:])
                child2 = Edge(
                    edge_id=self.next_edge_id,
                    birth_time=self.time,
                    nodes=child2_nodes,
                    edge_type="split",
                    parents=[parent.id]
                )
                self.edges[self.next_edge_id] = child2
                self.next_edge_id += 1

    def _deletion_rule(self):
        """Remove excess degree via K-cap."""
        for v in list(self.V):
            d = self.get_node_degree(v)
            if d > self.K:
                excess = d - self.K
                v_edges = [(eid, e) for eid, e in self.edges.items() if v in e.nodes and e.is_alive()]
                for eid, e in v_edges[:excess]:
                    if len(e.nodes) > 2:
                        new_nodes = e.nodes - {v}
                        if len(new_nodes) >= 2:
                            e.death_time = self.time
                            new_edge = Edge(
                                edge_id=self.next_edge_id,
                                birth_time=self.time,
                                nodes=new_nodes,
                                edge_type="deletion_prune",
                                parents=[e.id]
                            )
                            self.edges[self.next_edge_id] = new_edge
                            self.next_edge_id += 1

    def compute_reconstruction_metrics(self) -> Dict:
        """Compute new_ratio and related metrics for current core edges."""
        core_edges = self.get_core_edges()
        if not core_edges:
            return {
                'new_ratio': 0.0,
                'local_birth': 0,
                'fusion_birth': 0,
                'total_core_edges': 0,
                'mean_ancestry_depth': 0.0
            }

        local_birth = 0
        fusion_birth = 0
        depth_sum = 0

        for e in core_edges:
            if e.edge_type in ["growth", "split", "initial"]:
                local_birth += 1
            elif e.edge_type == "fusion":
                fusion_birth += 1
            depth_sum += e.get_ancestry_depth(self.edges)

        total = local_birth + fusion_birth
        new_ratio = local_birth / total if total > 0 else 0.0
        mean_depth = depth_sum / len(core_edges) if core_edges else 0.0

        return {
            'new_ratio': new_ratio,
            'local_birth': local_birth,
            'fusion_birth': fusion_birth,
            'total_core_edges': len(core_edges),
            'mean_ancestry_depth': mean_depth
        }


def run_ancestry_experiment(n_runs: int = 10, T: int = 80) -> Dict:
    """Run rigorous ancestry tracking experiment."""
    print("=" * 70)
    print("Rigorous Identity Tracking with Parent Chain")
    print("=" * 70)
    print(f"Configuration: n_runs={n_runs}, T={T}")
    print()

    all_results = []

    for run in range(n_runs):
        h = HypergraphIdentityTracker(N=50, p_pair=0.5, seed=run * 100 + 42)

        for t in range(T):
            h.apply_rules()

        metrics = h.compute_reconstruction_metrics()
        metrics['run'] = run
        metrics['M_final'] = h.get_M()

        all_results.append(metrics)

        print(f"Run {run + 1}/{n_runs}: "
              f"new_ratio={metrics['new_ratio']:.1%}, "
              f"local={metrics['local_birth']}, "
              f"fusion={metrics['fusion_birth']}, "
              f"core_edges={metrics['total_core_edges']}, "
              f"depth={metrics['mean_ancestry_depth']:.2f}")

    new_ratios = [r['new_ratio'] for r in all_results]
    depths = [r['mean_ancestry_depth'] for r in all_results]
    local_births = [r['local_birth'] for r in all_results]
    fusion_births = [r['fusion_birth'] for r in all_results]

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"new_ratio:      {np.mean(new_ratios):.1%} ± {np.std(new_ratios):.1%}")
    print(f"mean_depth:     {np.mean(depths):.2f} ± {np.std(depths):.2f}")
    print(f"local_birth:    {np.mean(local_births):.1f} ± {np.std(local_births):.1f}")
    print(f"fusion_birth:   {np.mean(fusion_births):.1f} ± {np.std(fusion_births):.1f}")
    print(f"M_final:        {np.mean([r['M_final'] for r in all_results]):.3f}")

    summary = {
        'n_runs': n_runs,
        'T': T,
        'new_ratio_mean': float(np.mean(new_ratios)),
        'new_ratio_std': float(np.std(new_ratios)),
        'depth_mean': float(np.mean(depths)),
        'depth_std': float(np.std(depths)),
        'local_birth_mean': float(np.mean(local_births)),
        'fusion_birth_mean': float(np.mean(fusion_births)),
        'individual_results': all_results
    }

    return summary


if __name__ == '__main__':
    results = run_ancestry_experiment(n_runs=10, T=80)
