"""
I/O utilities for saving and loading data.
"""

import json
import os
from typing import Any, Dict


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def save_json(data: Any, filepath: str) -> None:
    """
    Save data to JSON file.

    Args:
        data: Data to save (must be JSON serializable)
        filepath: Path to output file
    """
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(filepath: str) -> Any:
    """
    Load data from JSON file.

    Args:
        filepath: Path to JSON file

    Returns:
        Loaded data
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def append_to_json(data: Dict, filepath: str) -> None:
    """
    Append data to existing JSON file (list format).

    Args:
        data: Dictionary to append
        filepath: Path to JSON file
    """
    ensure_dir(os.path.dirname(filepath))
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            existing = json.load(f)
        if isinstance(existing, list):
            existing.append(data)
            data = existing
        else:
            raise ValueError("JSON file does not contain a list")
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
