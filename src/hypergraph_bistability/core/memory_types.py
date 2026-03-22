"""
Shared types for Agent Memory modules.
"""
import numpy as np
from dataclasses import dataclass


HIGH_THRESHOLD = 0.7
LOW_THRESHOLD = 0.3

DEFAULT_DT = 0.1
HIGH_TARGET = 0.95
LOW_TARGET = 0.05


@dataclass
class MemoryState:
    groups: np.ndarray
    layers: np.ndarray
    pattern: np.ndarray
    n_active: int
    mode_label: str
    
    def to_dict(self) -> dict:
        return {
            "groups": self.groups.tolist(),
            "layers": self.layers.tolist(),
            "pattern": self.pattern.tolist(),
            "n_active": self.n_active,
            "mode_label": self.mode_label
        }
