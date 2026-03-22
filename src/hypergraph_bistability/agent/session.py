"""Structured session state for practical agent persistence."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class SessionState:
    """Serializable session state for a HypergraphAgent instance."""

    schema_version: int = 2
    system_prompt: str = ""
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    turn_log: List[Dict[str, Any]] = field(default_factory=list)
    controller_state: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a plain dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionState":
        """Create session state from a plain dictionary."""
        return cls(
            schema_version=int(data.get("schema_version", 1)),
            system_prompt=data.get("system_prompt", ""),
            conversation_history=list(data.get("conversation_history", [])),
            turn_log=list(data.get("turn_log", [])),
            controller_state=dict(data.get("controller_state", {})),
        )
