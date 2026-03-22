"""Baseline agents for comparison."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional
import random


@dataclass
class BaselineResult:
    """Result from a baseline agent run."""
    
    name: str
    recall_precision: float
    recall_usefulness: float
    continuity_score: float
    metadata: dict


class BaselineAgent(ABC):
    """Base class for baseline memory agents."""
    
    def __init__(self, name: str):
        self.name = name
        self.memory: list[dict] = []
    
    @abstractmethod
    def add_to_memory(self, content: str, role: str = "user") -> None:
        """Add content to memory."""
        pass
    
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """Retrieve relevant memories."""
        pass
    
    def reset(self) -> None:
        """Reset memory."""
        self.memory = []


class RecentHistoryBaseline(BaselineAgent):
    """Baseline that only remembers recent N messages."""
    
    def __init__(self, history_size: int = 5):
        super().__init__("recent_history")
        self.history_size = history_size
    
    def add_to_memory(self, content: str, role: str = "user") -> None:
        self.memory.append({"content": content, "role": role})
        # Keep only recent history
        if len(self.memory) > self.history_size:
            self.memory = self.memory[-self.history_size:]
    
    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        # Return most recent
        return self.memory[-top_k:] if self.memory else []


class VectorSearchBaseline(BaselineAgent):
    """Baseline that uses simple keyword matching for retrieval."""
    
    def __init__(self):
        super().__init__("vector_search")
    
    def add_to_memory(self, content: str, role: str = "user") -> None:
        self.memory.append({
            "content": content, 
            "role": role,
            "keywords": set(content.lower().split())
        })
    
    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        query_words = set(query.lower().split())
        
        if not query_words or not self.memory:
            return self.memory[-top_k:]
        
        # Score by keyword overlap
        scored = []
        for item in self.memory:
            item_words = item.get("keywords", set())
            overlap = len(query_words & item_words)
            if overlap > 0:
                scored.append((overlap, item))
        
        # Sort by score
        scored.sort(key=lambda x: x[0], reverse=True)
        
        return [item for _, item in scored[:top_k]]


class RandomRecallBaseline(BaselineAgent):
    """Baseline that randomly recalls memories."""
    
    def __init__(self):
        super().__init__("random_recall")
    
    def add_to_memory(self, content: str, role: str = "user") -> None:
        self.memory.append({"content": content, "role": role})
    
    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        if len(self.memory) <= top_k:
            return self.memory[:]
        
        return random.sample(self.memory, top_k)


class SummaryBaseline(BaselineAgent):
    """Baseline that uses simple summarization."""
    
    def __init__(self):
        super().__init__("summary")
        self.summary = ""
    
    def add_to_memory(self, content: str, role: str = "user") -> None:
        # Simple: keep last 100 chars as summary
        self.summary = content[-100:] if len(content) > 100 else content
    
    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        # Return the summary
        if self.summary:
            return [{"content": self.summary, "role": "summary"}]
        return []


# Registry of baselines
BASELINES: dict[str, type[BaselineAgent]] = {
    "recent_history": RecentHistoryBaseline,
    "vector_search": VectorSearchBaseline,
    "random_recall": RandomRecallBaseline,
    "summary": SummaryBaseline,
}


def get_baseline(name: str, **kwargs) -> BaselineAgent:
    """Get a baseline agent by name."""
    baseline_class = BASELINES.get(name)
    if not baseline_class:
        raise ValueError(f"Unknown baseline: {name}")
    return baseline_class(**kwargs)


def run_baseline_comparison(
    scenario_turns: list[dict],
    baselines: list[str] = None,
) -> dict[str, BaselineResult]:
    """Run comparison across baselines."""
    if baselines is None:
        baselines = list(BASELINES.keys())
    
    results = {}
    
    for baseline_name in baselines:
        try:
            agent = get_baseline(baseline_name)
            
            # Simulate the scenario
            for turn in scenario_turns:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                
                # Add to memory
                agent.add_to_memory(content, role)
                
                # If assistant turn, try to retrieve relevant memories
                if role == "assistant":
                    retrieved = agent.retrieve(content)
                    
                    # Calculate simple metrics
                    precision = len(retrieved) / max(1, 5)  # Simplified
                    usefulness = 0.5  # Placeholder
                    
                    results[baseline_name] = BaselineResult(
                        name=baseline_name,
                        recall_precision=precision,
                        recall_usefulness=usefulness,
                        continuity_score=0.5,
                        metadata={"retrieved_count": len(retrieved)}
                    )
                    
        except Exception as e:
            results[baseline_name] = BaselineResult(
                name=baseline_name,
                recall_precision=0.0,
                recall_usefulness=0.0,
                continuity_score=0.0,
                metadata={"error": str(e)}
            )
    
    return results
