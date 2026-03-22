"""
Random utilities for reproducible experiments.
"""

import random
import numpy as np
from typing import List, Any, Optional


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed: Seed value
    """
    random.seed(seed)
    np.random.seed(seed)


def random_choice(population: List[Any], weights: Optional[List[float]] = None) -> Any:
    """
    Choose random element from population.

    Args:
        population: List of choices
        weights: Optional weights for each element

    Returns:
        Selected element
    """
    if weights is None:
        return random.choice(population)
    return random.choices(population, weights=weights, k=1)[0]


def random_sample(population: List[Any], k: int) -> List[Any]:
    """
    Sample k unique elements from population.

    Args:
        population: List to sample from
        k: Number of samples

    Returns:
        List of k samples
    """
    return random.sample(population, k)


def randuniform(low: float, high: float, size: Optional[int] = None) -> float:
    """
    Generate uniform random number(s).

    Args:
        low: Lower bound
        high: Upper bound
        size: If None, return scalar; if int, return array of size

    Returns:
        Random number(s)
    """
    if size is None:
        return random.uniform(low, high)
    return np.random.uniform(low, high, size)


def randn(size: Optional[int] = None) -> float:
    """
    Generate standard normal random number(s).

    Args:
        size: If None, return scalar; if tuple, return array

    Returns:
        Random number(s) from N(0,1)
    """
    if size is None:
        return float(np.random.randn())
    return np.random.randn(*size)
