"""
Math utilities wrapping numpy functions.
"""

import numpy as np
from typing import List, Union

ArrayLike = Union[List[float], np.ndarray, float]


def mean(arr: ArrayLike) -> float:
    """Compute mean of array."""
    return float(np.mean(arr))


def std(arr: ArrayLike) -> float:
    """Compute standard deviation of array."""
    return float(np.std(arr))


def clamp(x: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp value to [low, high] range."""
    return float(np.clip(x, low, high))


def linspace(start: float, stop: float, num: int) -> np.ndarray:
    """Generate evenly spaced numbers."""
    return np.linspace(start, stop, num)


def arange(start: float, stop: float, step: float) -> np.ndarray:
    """Generate.arange style sequence."""
    return np.arange(start, stop, step)


def euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Euclidean distance between two vectors."""
    return float(np.linalg.norm(x - y))


def normalize(x: np.ndarray) -> np.ndarray:
    """Normalize array to [0, 1] range."""
    x_min = x.min()
    x_max = x.max()
    if x_max - x_min == 0:
        return np.zeros_like(x)
    return (x - x_min) / (x_max - x_min)


def softmax(x: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """Compute softmax of array."""
    x_beta = beta * x
    x_beta -= x_beta.max()
    return np.exp(x_beta) / np.exp(x_beta).sum()
