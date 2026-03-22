"""
Core dynamics functions for hypergraph system.

Note on dynamics equations:
The group-level dynamics use a cubic polynomial f(M) = a*M³ + b*M² + c*M
with a = -3/Kc, b = 4.5/Kc, c = -1.5/Kc. This is equivalent (up to 
Kc-scaling) to the canonical form f(m) = m(1-m)(2m-1) used in the paper,
up to a factor absorbed in the time scale. The parameter Kc controls the
capacity asymmetry across groups.
"""

import numpy as np
import random
from typing import Dict, List, Optional, Tuple


class HypergraphDynamics:
    """Container for hypergraph dynamics operations."""

    def __init__(self, L: int = 2, k: int = 3, Kc_list: Optional[List[float]] = None):
        """
        Args:
            L: Number of layers
            k: Number of groups
            Kc_list: Critical capacity values for each group
        """
        self.L = L
        self.k = k
        self.Kc_list = Kc_list if Kc_list else [0.32, 0.40, 0.48]

        self.a = [-3 / kc for kc in self.Kc_list]
        self.b = [4.5 / kc for kc in self.Kc_list]
        self.c = [-1.5 / kc for kc in self.Kc_list]

    def compute_dM(self, M: np.ndarray, lam: float = 0.0, mu: float = 0.0) -> np.ndarray:
        """
        Compute time derivative of order parameter.

        Args:
            M: Order parameter array of shape (L, k)
            lam: Intra-layer competition strength
            mu: Inter-layer coupling strength

        Returns:
            dM/dt array of shape (L, k)
        """
        M = M.reshape((self.L, self.k))
        dM = np.zeros((self.L, self.k))

        for l in range(self.L):
            for i in range(self.k):
                Mi = M[l, i]
                st = self.a[i] * Mi**3 + self.b[i] * Mi**2 + self.c[i] * Mi
                cr = -lam * Mi * np.sum(M[l, np.arange(self.k) != i])
                cl = mu * np.sum(M[np.arange(self.L) != l, i])
                dM[l, i] = st + cr + cl

        return dM.flatten()

    def dynamics_ode(self, M_flat: np.ndarray, t: float,
                     lam: float, mu: float,
                     boost_active: bool = False,
                     boost_target: int = 0,
                     boost_factor: float = 1.0) -> np.ndarray:
        """ODE function for scipy.integrate.odeint with optional boost."""
        M = M_flat.reshape((self.L, self.k))
        dM = np.zeros((self.L, self.k))

        a_eff = self.a.copy()
        b_eff = self.b.copy()
        c_eff = self.c.copy()

        if boost_active:
            a_eff[boost_target] /= boost_factor
            b_eff[boost_target] /= boost_factor
            c_eff[boost_target] /= boost_factor

        for l in range(self.L):
            for i in range(self.k):
                Mi = M[l, i]
                st = a_eff[i] * Mi**3 + b_eff[i] * Mi**2 + c_eff[i] * Mi
                cr = -lam * Mi * np.sum(M[l, np.arange(self.k) != i])
                cl = mu * np.sum(M[np.arange(self.L) != l, i])
                dM[l, i] = st + cr + cl

        return dM.flatten()


def compute_order_parameter(
    m: np.ndarray,
    ga: np.ndarray,
    la: np.ndarray,
    L: int,
    k: int
) -> np.ndarray:
    """
    Compute order parameter M[l,i] from node states.

    Args:
        m: Node state array of length N
        ga: Group assignments of length N
        la: Layer assignments of length N
        L: Number of layers
        k: Number of groups

    Returns:
        M array of shape (L, k)
    """
    N = len(m)
    M = np.zeros((L, k))
    cnt = np.zeros((L, k))

    for v in range(N):
        i, l = ga[v], la[v]
        M[l, i] += m[v]
        cnt[l, i] += 1

    return M / (cnt + 1e-8)


def apply_growth_rule(
    E: List[frozenset],
    V: List[int],
    s: Dict[int, np.ndarray],
    group_assignments: Dict[int, int],
    n_groups: int,
    probability: float = 0.35,
    state_dim: int = 16
) -> Tuple[List[frozenset], List[int], Dict[int, np.ndarray], Dict[int, int]]:
    """Apply growth rule: add new node connected to existing node."""
    if random.random() < probability and E:
        e = random.choice(E)
        nodes = list(e)
        w = len(V)
        V.append(w)
        s[w] = s[nodes[0]] + np.random.randn(state_dim) * 0.2
        group_assignments[w] = group_assignments[nodes[0]]
        new_e = frozenset([nodes[0], w])
        E.append(new_e)
    return E, V, s, group_assignments


def apply_fusion_rule(
    E: List[frozenset],
    V: List[int],
    s: Dict[int, np.ndarray],
    group_assignments: Dict[int, int],
    get_node_degree,
    probability: float = 0.3,
    lambda_ij: float = 0.5
) -> List[frozenset]:
    """Apply fusion rule: merge two hyperedges."""
    if random.random() < probability and len(E) >= 2:
        e1, e2 = random.sample(E, 2)
        if len(e1 & e2) >= 1:
            g1 = group_assignments[list(e1)[0]]
            g2 = group_assignments[list(e2)[0]]

            weights1 = np.array([1.0 / (1 + get_node_degree(v)) for v in e1])
            weights1 = weights1 / weights1.sum()
            c1 = np.average([s[u] for u in e1], axis=0, weights=weights1)

            weights2 = np.array([1.0 / (1 + get_node_degree(v)) for v in e2])
            weights2 = weights2 / weights2.sum()
            c2 = np.average([s[u] for u in e2], axis=0, weights=weights2)

            dist = np.linalg.norm(c1 - c2)
            threshold = 0.8

            if g1 == g2 and dist < threshold:
                new_e = frozenset(e1 | e2)
                if new_e not in E:
                    E.append(new_e)
                    E.remove(e1)
                    E.remove(e2)
            elif g1 != g2 and random.random() < (1 - lambda_ij) and dist < threshold:
                new_e = frozenset(e1 | e2)
                if new_e not in E:
                    E.append(new_e)
                    E.remove(e1)
                    E.remove(e2)

    return E


def apply_split_rule(
    E: List[frozenset],
    group_assignments: Dict[int, int],
    probability: float = 0.15
) -> List[frozenset]:
    """Apply split rule: split large hyperedge into two."""
    if random.random() < probability:
        large_edges = [e for e in E if len(e) > 2]
        if large_edges:
            e = random.choice(large_edges)
            nodes = list(e)
            if len(nodes) >= 2:
                split_point = len(nodes) // 2
                e1 = frozenset(nodes[:split_point])
                e2 = frozenset(nodes[split_point:])
                if e1 not in E and e2 not in E:
                    E.append(e1)
                    E.append(e2)
                    E.remove(e)
    return E


def apply_deletion_rule(
    E: List[frozenset],
    probability: float = 0.2
) -> List[frozenset]:
    """Apply deletion rule: remove small hyperedges."""
    if random.random() < probability:
        small_edges = [e for e in E if len(e) == 2]
        if small_edges:
            e = random.choice(small_edges)
            E.remove(e)
    return E

