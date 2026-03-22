"""
Potential function analysis for bistability/multistability.
"""

import numpy as np
from typing import List, Tuple, Optional


def double_well_potential(
    M: float,
    a: float = 1.0,
    b: float = 0.15
) -> float:
    """
    Double-well potential: V(M) = a*M^2*(M-1)^2 - b*M

    Args:
        M: Order parameter value
        a: Well depth parameter
        b: Asymmetry parameter

    Returns:
        Potential value
    """
    return a * M**2 * (M - 1)**2 - b * M


def multi_well_potential(
    M: np.ndarray,
    wells: List[Tuple[float, float]],
    a: float = 1.0
) -> float:
    """
    Multi-well potential from sum of Gaussian wells.

    Args:
        M: Order parameter value(s)
        wells: List of (center, depth) tuples
        a: Well width parameter

    Returns:
        Potential value
    """
    V = 0.0
    for center, depth in wells:
        V += depth * np.exp(-a * (M - center)**2)
    return V


def compute_barrier_height(
    x1: float,
    x2: float,
    potential_func,
    n_points: int = 100
) -> float:
    """
    Compute barrier height between two minima.

    Args:
        x1: First minimum position
        x2: Second minimum position
        potential_func: Function that computes V(x)
        n_points: Number of points to search

    Returns:
        Barrier height
    """
    x_range = np.linspace(min(x1, x2), max(x1, x2), n_points)
    V_values = [potential_func(x) for x in x_range]
    barrier = max(V_values)
    return barrier - min(potential_func(x1), potential_func(x2))


def find_fixed_points(
    a: float,
    b: float
) -> List[float]:
    """
    Find fixed points of minimal model: F(x) = a*x*(1-x) - b = 0

    Args:
        a: Competition strength
        b: Constraint strength

    Returns:
        List of fixed points in [0, 1]
    """
    discriminant = a**2 - 4 * a * b
    if discriminant < 0:
        return []

    sqrt_disc = np.sqrt(discriminant)
    x1 = (a - sqrt_disc) / (2 * a)
    x2 = (a + sqrt_disc) / (2 * a)

    roots = []
    if 0 <= x1 <= 1:
        roots.append(x1)
    if 0 <= x2 <= 1:
        roots.append(x2)

    return roots


def stability_derivative(
    x: float,
    a: float,
    b: float
) -> float:
    """
    Compute dF/dx for stability analysis.

    F(x) = a*x*(1-x) - b
    dF/dx = a*(1 - 2*x)

    Args:
        x: Position
        a: Competition strength
        b: Constraint strength

    Returns:
        dF/dx value (negative = stable)
    """
    return a * (1 - 2 * x)


def classify_stability(
    roots: List[float],
    a: float,
    b: float
) -> List[Tuple[float, str]]:
    """
    Classify fixed points as stable or unstable.

    Args:
        roots: List of fixed points
        a: Competition strength
        b: Constraint strength

    Returns:
        List of (root, stability) tuples
    """
    result = []
    for root in roots:
        dFdx = stability_derivative(root, a, b)
        stability = "stable" if dFdx < 0 else "unstable"
        result.append((root, stability))
    return result


def compute_kramers_rate(
    barrier_height: float,
    temperature: float = 1.0
) -> float:
    """
    Compute Kramers escape rate: Γ ∝ exp(-ΔV/T)

    Args:
        barrier_height: Barrier height ΔV
        temperature: Temperature T

    Returns:
        Escape rate
    """
    return np.exp(-barrier_height / temperature)


def simulate_dynamics(
    x0: float,
    a: float,
    b: float,
    steps: int = 100,
    dt: float = 0.1
) -> List[float]:
    """
    Simulate gradient dynamics: dx/dt = F(x)

    Args:
        x0: Initial condition
        a: Competition strength
        b: Constraint strength
        steps: Number of steps
        dt: Time step

    Returns:
        Trajectory
    """
    x = x0
    trajectory = [x]

    for _ in range(steps):
        F = a * x * (1 - x) - b
        x = x + dt * F
        x = np.clip(x, 0, 1)
        trajectory.append(x)

    return trajectory
