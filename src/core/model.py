"""
Core hypergraph model classes.
"""

import numpy as np
import random
from typing import Dict, List, Optional, Tuple


class MultiGroupHypergraph:
    """Multi-group competitive hypergraph dynamics system."""

    def __init__(
        self,
        N: int = 50,
        n_groups: int = 3,
        gamma: float = 0.35,
        state_dim: int = 16,
        inter_group_competition: float = 0.5,
        p_pair: float = 0.3,
        seed: int = 42
    ):
        """
        Args:
            N: Total number of nodes
            n_groups: Number of competing groups (k >= 3 for multistability)
            gamma: Capacity constraint parameter
            state_dim: State vector dimensionality
            inter_group_competition: Cross-group competition strength lambda (0~1)
            p_pair: Probability of creating binary edge (vs ternary); lower = more clustering
            seed: Random seed
        """
        random.seed(seed)
        np.random.seed(seed)

        self.N = N
        self.n_groups = n_groups
        self.gamma = gamma
        self.state_dim = state_dim
        self.K = int(gamma * N)
        self.lambda_ij = inter_group_competition
        self.p_pair = p_pair

        self._init_groups()
        self._init_nodes()
        self._init_hyperedges()

    def _init_groups(self):
        """Initialize group assignments."""
        self.group_assignments = {}
        nodes_per_group = self.N // self.n_groups
        for i in range(self.n_groups):
            start = i * nodes_per_group
            end = start + nodes_per_group if i < self.n_groups - 1 else self.N
            for v in range(start, end):
                self.group_assignments[v] = i

    def _init_nodes(self):
        """Initialize nodes and their states."""
        self.V = list(range(self.N))
        self.s = {v: np.random.randn(self.state_dim) for v in self.V}
        self.group_gamma = {i: self.gamma for i in range(self.n_groups)}

    def _init_hyperedges(self):
        """Initialize hyperedges per group."""
        self.E = []
        n_initial = max(4, self.N // 4)
        for _ in range(n_initial):
            group = random.randint(0, self.n_groups - 1)
            group_nodes = self.get_group_nodes(group)
            if len(group_nodes) >= 2:
                size = random.randint(2, min(5, len(group_nodes) // 2))
                e = frozenset(random.sample(group_nodes, size))
                self.E.append(e)

    def get_node_group(self, v: int) -> int:
        """Get group assignment of node v."""
        return self.group_assignments[v]

    def get_group_nodes(self, g: int) -> List[int]:
        """Get all nodes in group g."""
        return [v for v in self.V if self.group_assignments[v] == g]

    def get_node_degree(self, v: int) -> int:
        """Get degree of node v."""
        return sum(1 for e in self.E if v in e)

    def get_group_order_parameter(self, g: int) -> float:
        """Compute order parameter M_g for group g."""
        group_nodes = self.get_group_nodes(g)
        if not group_nodes:
            return 0.0
        degrees = [self.get_node_degree(v) for v in group_nodes]
        return float(np.mean(degrees)) / float(max(self.K, 1))

    def get_avg_distance(self, nodes: List[int]) -> float:
        """Compute average distance between nodes in list."""
        if len(nodes) < 2:
            return 0.1
        samples = min(20, len(nodes) * (len(nodes) - 1) // 2)
        if samples <= 0:
            return 0.1
        dist_sum = 0.0
        for _ in range(samples):
            u, v = random.sample(nodes, 2)
            dist_sum += float(np.linalg.norm(self.s[u] - self.s[v]))
        dist_avg = dist_sum / samples
        return max(dist_avg, 0.01)

    def apply_rules(self):
        """Apply growth, fusion, split, and deletion rules."""
        avg_dist_overall = self.get_avg_distance(self.V)

        for g in range(self.n_groups):
            group_nodes = self.get_group_nodes(g)
            if not group_nodes:
                continue

            avg_dist_g = self.get_avg_distance(group_nodes)
            gamma_g = self.group_gamma[g]
            K_g = int(gamma_g * len(group_nodes))

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
                    if random.random() < self.p_pair:
                        new_e = frozenset([nodes[0], w])
                    else:
                        third = random.choice([v for v in group_nodes if v != nodes[0]])
                        new_e = frozenset([nodes[0], w, third])
                    self.E.append(new_e)

            if random.random() < 0.3 and len(self.E) >= 2:
                e1, e2 = random.sample(self.E, 2)
                if len(e1 & e2) >= 1:
                    g1 = self.get_node_group(list(e1)[0])
                    g2 = self.get_node_group(list(e2)[0])

                    weights1 = np.array([1.0 / (1 + self.get_node_degree(v)) for v in e1])
                    weights1 = weights1 / weights1.sum()
                    c1 = np.average([self.s[u] for u in e1], axis=0, weights=weights1)

                    weights2 = np.array([1.0 / (1 + self.get_node_degree(v)) for v in e2])
                    weights2 = weights2 / weights2.sum()
                    c2 = np.average([self.s[u] for u in e2], axis=0, weights=weights2)

                    dist = np.linalg.norm(c1 - c2)

                    if g1 == g2:
                        threshold = avg_dist_g * 0.8
                        if dist < threshold:
                            new_e = frozenset(e1 | e2)
                            if new_e not in self.E:
                                self.E.append(new_e)
                                self.E.remove(e1)
                                self.E.remove(e2)
                    else:
                        if random.random() < (1 - self.lambda_ij):
                            threshold = avg_dist_g * 0.8
                            if dist < threshold:
                                new_e = frozenset(e1 | e2)
                                if new_e not in self.E:
                                    self.E.append(new_e)
                                    self.E.remove(e1)
                                    self.E.remove(e2)

            if random.random() < 0.15:
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

            if random.random() < 0.2:
                small_edges = [e for e in self.E if len(e) == 2]
                if small_edges:
                    e = random.choice(small_edges)
                    self.E.remove(e)

    def run_dynamics(self, steps: int = 80) -> Dict[int, List[float]]:
        """Run dynamics for specified steps."""
        M_history = {g: [] for g in range(self.n_groups)}

        for _ in range(steps):
            self.apply_rules()
            for g in range(self.n_groups):
                M_history[g].append(self.get_group_order_parameter(g))

        return M_history

    def get_stable_states(self) -> Dict[int, float]:
        """Get stable state order parameters for all groups."""
        return {g: self.get_group_order_parameter(g) for g in range(self.n_groups)}


class MultiLayerHypergraph:
    """Multi-layer hypergraph with inter-layer coupling."""

    def __init__(
        self,
        N: int = 50,
        n_groups: int = 3,
        n_layers: int = 2,
        gamma: float = 0.35,
        state_dim: int = 16,
        inter_group_competition: float = 0.5,
        inter_layer_coupling: float = 0.2,
        p_pair: float = 0.3,
        seed: int = 42
    ):
        """
        Args:
            N: Total nodes
            n_groups: Groups per layer
            n_layers: Number of layers (L >= 2)
            gamma: Capacity constraint
            state_dim: State dimensionality
            inter_group_competition: Cross-group competition lambda
            inter_layer_coupling: Cross-layer coupling mu (-0.5 ~ 0.5)
            p_pair: Probability of binary edge creation
            seed: Random seed
        """
        self.N = N
        self.n_groups = n_groups
        self.n_layers = n_layers
        self.gamma = gamma
        self.state_dim = state_dim
        self.lambda_ij = inter_group_competition
        self.mu = inter_layer_coupling
        self.p_pair = p_pair

        self.layers = []
        for l in range(n_layers):
            layer = MultiGroupHypergraph(
                N=N,
                n_groups=n_groups,
                gamma=gamma,
                state_dim=state_dim,
                inter_group_competition=inter_group_competition,
                p_pair=p_pair,
                seed=seed + l * 100
            )
            self.layers.append(layer)

    def apply_cross_layer_rules(self):
        """Apply cross-layer interaction rules."""
        if self.mu == 0:
            return

        for l1 in range(self.n_layers):
            for l2 in range(self.n_layers):
                if l1 >= l2:
                    continue

                for v in self.layers[l1].V:
                    if v in self.layers[l2].V:
                        s1 = self.layers[l1].s[v]
                        s2 = self.layers[l2].s[v]

                        if self.mu > 0:
                            self.layers[l1].s[v] = (1 - self.mu) * s1 + self.mu * s2
                            self.layers[l2].s[v] = (1 - self.mu) * s2 + self.mu * s1
                        else:
                            self.layers[l1].s[v] = s1 - self.mu * s2
                            self.layers[l2].s[v] = s2 - self.mu * s1

    def run_dynamics(self, steps: int = 80) -> Dict:
        """Run multi-layer dynamics."""
        M_history = {l: {g: [] for g in range(self.n_groups)}
                    for l in range(self.n_layers)}

        for _ in range(steps):
            for layer in self.layers:
                layer.apply_rules()
            self.apply_cross_layer_rules()

            for l in range(self.n_layers):
                for g in range(self.n_groups):
                    M_history[l][g].append(self.layers[l].get_group_order_parameter(g))

        return M_history

    def get_all_stable_states(self) -> Dict[Tuple[int, int], float]:
        """Get stable states for all layer-group combinations."""
        return {(l, g): self.layers[l].get_group_order_parameter(g)
                for l in range(self.n_layers)
                for g in range(self.n_groups)}
