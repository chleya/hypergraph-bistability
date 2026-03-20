"""
Noise analysis functions for hypergraph dynamics.
"""

import numpy as np
from typing import List, Tuple


def add_gaussian_noise(
    m: np.ndarray,
    sigma: float
) -> np.ndarray:
    """
    Add Gaussian noise to node states.

    Args:
        m: Node state array
        sigma: Standard deviation of noise

    Returns:
        Noisy state array
    """
    return m + np.random.normal(0, sigma, len(m))


def compute_noise_effect(
    states: np.ndarray,
    threshold: float = 0.1
) -> int:
    """
    Compute number of unique attractor states.

    Args:
        states: Array of final states, shape (n_runs, N)
        threshold: Distance threshold for considering states same

    Returns:
        Number of unique attractors
    """
    uniq = []
    for s in states:
        is_new = True
        for u in uniq:
            if np.sqrt(np.mean((s - u)**2)) < threshold:
                is_new = False
                break
        if is_new:
            uniq.append(s)
    return len(uniq)


def noise_driven_transition(
    m: np.ndarray,
    H: np.ndarray,
    ga: np.ndarray,
    la: np.ndarray,
    Kc_list: List[float],
    lam: float,
    L: int,
    k: int,
    sigma: float,
    dt: float = 0.05
) -> np.ndarray:
    """
    Apply noise-driven dynamics step.

    Args:
        m: Current states (N,)
        H: Hypergraph adjacency (N, E)
        ga: Group assignments (N,)
        la: Layer assignments (N,)
        Kc_list: Critical capacities
        lam: Competition strength
        L: Number of layers
        k: Number of groups
        sigma: Noise strength
        dt: Time step

    Returns:
        Updated states
    """
    N = len(m)
    M = np.zeros((L, k))
    cnt = np.zeros((L, k))

    for v in range(N):
        i, l = ga[v], la[v]
        M[l, i] += m[v]
        cnt[l, i] += 1
    M = M / (cnt + 1e-8)

    a = [-3.0 / kc for kc in Kc_list]
    b = [4.5 / kc for kc in Kc_list]
    c = [-1.5 / kc for kc in Kc_list]

    dM = np.zeros((L, k))
    for l in range(L):
        for i in range(k):
            Mi = M[l, i]
            st = a[i] * Mi**3 + b[i] * Mi**2 + c[i] * Mi
            cr = -lam * Mi * (np.sum(M[l, :]) - Mi)
            dM[l, i] = st + cr

    dm = np.zeros(N)
    for v in range(N):
        i, l = ga[v], la[v]
        dm[v] = dM[l, i] + np.random.normal(0, sigma)

    return np.clip(m + dt * dm, 0.01, 0.99)


def compute_switching_probability(
    initial_states: np.ndarray,
    final_states: np.ndarray,
    target_state: np.ndarray,
    threshold: float = 0.1
) -> float:
    """
    Compute probability of switching to target attractor.

    Args:
        initial_states: Starting states (n_runs, N)
        final_states: Ending states (n_runs, N)
        target_state: Target attractor (N,)
        threshold: Distance threshold

    Returns:
        Switching probability
    """
    count = 0
    for final in final_states:
        dist = np.sqrt(np.mean((final - target_state)**2))
        if dist < threshold:
            count += 1
    return count / len(final_states)
