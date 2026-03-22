"""Evaluation scenarios for agent memory testing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional
import json


@dataclass
class Scenario:
    """Base class for evaluation scenarios."""
    
    name: str
    description: str
    turns: list[dict] = field(default_factory=list)
    expected_memories: list[str] = field(default_factory=list)
    ground_truth: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "turns": self.turns,
            "expected_memories": self.expected_memories,
            "ground_truth": self.ground_truth,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> Scenario:
        return cls(**data)


class PreferenceRecallScenario(Scenario):
    """Scenario testing preference retention over many turns."""
    
    def __init__(self):
        super().__init__(
            name="preference_recall",
            description="Test if agent remembers user preferences over 30 turns",
            turns=[
                {"role": "user", "content": "I prefer responses in Chinese."},
                {"role": "assistant", "content": "好的，我会用中文回复。"},
                # ... filler turns
                {"role": "user", "content": "What programming language do I like?"},
            ],
            expected_memories=["prefer Chinese"],
            ground_truth={"preferred_language": "chinese"}
        )


class TaskContinuityScenario(Scenario):
    """Scenario testing task continuity across interruptions."""
    
    def __init__(self):
        super().__init__(
            name="task_continuity",
            description="Test if agent maintains task context across interruptions",
            turns=[
                {"role": "user", "content": "Fix the login bug in user_service.py"},
                {"role": "assistant", "content": "I'll investigate the login issue. Let me check the code."},
                # ... some turns later
                {"role": "user", "content": "What was I working on?"},
            ],
            expected_memories=["login bug", "user_service.py"],
            ground_truth={"current_task": "fix login bug"}
        )


class ContextSwitchScenario(Scenario):
    """Scenario testing context switching between topics."""
    
    def __init__(self):
        super().__init__(
            name="context_switch",
            description="Test if agent can switch between personal and technical topics",
            turns=[
                {"role": "user", "content": "My name is Alice."},
                {"role": "assistant", "content": "好的，Alice，我记住了。"},
                # ... technical turns
                {"role": "user", "content": "How do I fix a Python ImportError?"},
                {"role": "assistant", "content": "ImportError in Python can be fixed by..."},
                # ... switch back
                {"role": "user", "content": "What's my name?"},
            ],
            expected_memories=["Alice", "ImportError solution"],
            ground_truth={"user_name": "alice"}
        )


class CodingAssistantScenario(Scenario):
    """Scenario testing coding assistant memory."""
    
    def __init__(self):
        super().__init__(
            name="coding_assistant",
            description="Test if coding assistant remembers project constraints",
            turns=[
                {"role": "user", "content": "We're using Python 3.10 and FastAPI. Don't use Django."},
                {"role": "assistant", "content": "Got it. I'll use Python 3.10 and FastAPI, no Django."},
                # ... later
                {"role": "user", "content": "Create a new API endpoint."},
            ],
            expected_memories=["Python 3.10", "FastAPI", "no Django"],
            ground_truth={"language": "python_3.10", "framework": "fastapi", "forbidden": ["django"]}
        )


class ConflictResolutionScenario(Scenario):
    """Scenario testing conflict/hypothesis tracking."""
    
    def __init__(self):
        super().__init__(
            name="conflict_resolution",
            description="Test if agent tracks conflicting hypotheses",
            turns=[
                {"role": "user", "content": "The API is returning 500 errors. Could be database or cache issue."},
                {"role": "assistant", "content": "Let me investigate both possibilities."},
                # ... investigation
                {"role": "user", "content": "The database connection looks fine. What about the cache?"},
            ],
            expected_memories=["500 error", "database hypothesis", "cache hypothesis"],
            ground_truth={"issue": "500_errors", "hypotheses": ["database", "cache"]}
        )


class SessionRecoveryScenario(Scenario):
    """Scenario testing recovery after session gap."""
    
    def __init__(self):
        super().__init__(
            name="session_recovery",
            description="Test context recovery after session gap",
            turns=[
                {"role": "user", "content": "Working on feature X. Need to add authentication."},
                {"role": "assistant", "content": "OK, adding authentication to feature X."},
                # ... simulate gap (new session)
                {"role": "user", "content": "What was I doing?"},
            ],
            expected_memories=["feature X", "authentication"],
            ground_truth={"last_task": "add authentication to feature X"}
        )


# Registry of all scenarios
SCENARIOS: dict[str, type[Scenario]] = {
    "preference_recall": PreferenceRecallScenario,
    "task_continuity": TaskContinuityScenario,
    "context_switch": ContextSwitchScenario,
    "coding_assistant": CodingAssistantScenario,
    "conflict_resolution": ConflictResolutionScenario,
    "session_recovery": SessionRecoveryScenario,
}


def get_scenario(name: str) -> Optional[Scenario]:
    """Get a scenario by name."""
    scenario_class = SCENARIOS.get(name)
    if scenario_class:
        return scenario_class()
    return None


def list_scenarios() -> list[str]:
    """List all available scenario names."""
    return list(SCENARIOS.keys())


def create_scenario(name: str, **kwargs) -> Scenario:
    """Create a custom scenario."""
    return Scenario(name=name, **kwargs)
