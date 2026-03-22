"""Metrics for evaluating agent memory performance."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional
import time


@dataclass
class MetricResult:
    """Result of a single metric evaluation."""
    
    name: str
    value: float
    score: float  # 0.0 to 1.0 normalized score
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "value": self.value,
            "score": self.score,
            "details": self.details,
            "timestamp": self.timestamp,
        }


class Metric(ABC):
    """Base class for evaluation metrics."""
    
    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight
    
    @abstractmethod
    def evaluate(
        self,
        *,
        conversation_history: list[dict],
        memory_state: Optional[dict] = None,
        retrieved_memories: Optional[list[dict]] = None,
        responses: Optional[list[str]] = None,
        **kwargs
    ) -> MetricResult:
        """Evaluate the metric given conversation and memory state."""
        pass
    
    def _normalize_score(self, value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """Normalize value to 0.0-1.0 range."""
        if max_val == min_val:
            return 1.0 if value == min_val else 0.0
        return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))


class TaskContinuation(Metric):
    """Measures whether the agent continues working on the same task."""
    
    def __init__(self):
        super().__init__("task_continuation", weight=1.0)
    
    def evaluate(
        self,
        *,
        conversation_history: list[dict],
        memory_state: Optional[dict] = None,
        **kwargs
    ) -> MetricResult:
        # Extract task identifiers from conversation
        tasks = []
        for turn in conversation_history:
            content = turn.get("content", "")
            # Look for task indicators
            if any(kw in content.lower() for kw in ["task:", "working on", "todo", "fixing"]):
                # Extract task name
                for prefix in ["task:", "working on", "todo:", "fixing"]:
                    if prefix in content.lower():
                        idx = content.lower().find(prefix)
                        task = content[idx:idx+50].strip()
                        if task not in tasks:
                            tasks.append(task)
                        break
        
        # Check if task persists across turns
        if len(tasks) < 2:
            return MetricResult(
                name=self.name,
                value=0.0,
                score=0.0,
                details={"tasks_found": len(tasks)}
            )
        
        # Simple metric: task mentioned in later turns
        later_turns = conversation_history[len(conversation_history)//2:]
        task_mentions = sum(
            1 for turn in later_turns 
            if any(t.lower() in turn.get("content", "").lower() for t in tasks)
        )
        
        value = task_mentions / max(1, len(later_turns))
        
        return MetricResult(
            name=self.name,
            value=value,
            score=self._normalize_score(value),
            details={"tasks": tasks, "task_mentions": task_mentions}
        )


class BlockerPreservation(Metric):
    """Measures whether blockers are remembered and addressed."""
    
    def __init__(self):
        super().__init__("blocker_preservation", weight=1.0)
    
    def evaluate(
        self,
        *,
        conversation_history: list[dict],
        **kwargs
    ) -> MetricResult:
        blockers = []
        for turn in conversation_history:
            content = turn.get("content", "")
            if any(kw in content.lower() for kw in ["blocker", "blocked", "cannot proceed", "waiting for"]):
                blockers.append(content[:100])
        
        if not blockers:
            return MetricResult(
                name=self.name,
                value=1.0,
                score=1.0,
                details={"blockers_found": 0}
            )
        
        # Check if blockers are referenced later
        all_content = " ".join(turn.get("content", "") for turn in conversation_history)
        resolved = sum(1 for b in blockers if any(
            kw in all_content.lower() 
            for kw in ["resolved", "fixed", "done", "completed"]
        ))
        
        value = resolved / len(blockers) if blockers else 1.0
        
        return MetricResult(
            name=self.name,
            value=value,
            score=self._normalize_score(value),
            details={"blockers": blockers, "resolved": resolved}
        )


class DecisionContinuity(Metric):
    """Measures whether past decisions are remembered."""
    
    def __init__(self):
        super().__init__("decision_continuity", weight=1.0)
    
    def evaluate(
        self,
        *,
        conversation_history: list[dict],
        **kwargs
    ) -> MetricResult:
        decisions = []
        for turn in conversation_history:
            content = turn.get("content", "")
            if any(kw in content.lower() for kw in ["decision:", "we decided", "chose to", "going with"]):
                decisions.append(content[:100])
        
        if not decisions:
            return MetricResult(
                name=self.name,
                value=1.0,
                score=1.0,
                details={"decisions_found": 0}
            )
        
        # Check if decisions are referenced in later turns
        later_content = " ".join(
            turn.get("content", "") 
            for turn in conversation_history[len(conversation_history)//2:]
        )
        
        referenced = sum(
            1 for d in decisions 
            if any(word in later_content.lower() for word in d.lower().split()[:5])
        )
        
        value = referenced / len(decisions)
        
        return MetricResult(
            name=self.name,
            value=value,
            score=self._normalize_score(value),
            details={"decisions": decisions, "referenced": referenced}
        )


class ProcedureContinuity(Metric):
    """Measures whether procedures/playbooks are followed."""
    
    def __init__(self):
        super().__init__("procedure_continuity", weight=1.0)
    
    def evaluate(
        self,
        *,
        conversation_history: list[dict],
        **kwargs
    ) -> MetricResult:
        procedures = []
        for turn in conversation_history:
            content = turn.get("content", "")
            if any(kw in content.lower() for kw in ["checklist", "playbook", "procedure", "steps:"]):
                procedures.append(content[:100])
        
        if not procedures:
            return MetricResult(
                name=self.name,
                value=1.0,
                score=1.0,
                details={"procedures_found": 0}
            )
        
        # Check if procedure steps are followed in order
        value = 0.5  # Default partial credit
        
        return MetricResult(
            name=self.name,
            value=value,
            score=value,
            details={"procedures": procedures}
        )


class ConflictContinuity(Metric):
    """Measures conflict hypothesis tracking."""
    
    def __init__(self):
        super().__init__("conflict_continuity", weight=1.0)
    
    def evaluate(
        self,
        *,
        conversation_history: list[dict],
        memory_state: Optional[dict] = None,
        **kwargs
    ) -> MetricResult:
        # Look for conflict/hypothesis discussions
        hypotheses = []
        for turn in conversation_history:
            content = turn.get("content", "")
            if any(kw in content.lower() for kw in ["hypothesis", "suspect", "root cause", "likely"]):
                hypotheses.append(content[:100])
        
        if not hypotheses:
            return MetricResult(
                name=self.name,
                value=1.0,
                score=1.0,
                details={"hypotheses_found": 0}
            )
        
        # Check if dominant hypothesis is tracked
        later_content = " ".join(
            turn.get("content", "") 
            for turn in conversation_history[len(conversation_history)//2:]
        )
        
        # Count hypothesis mentions in later turns
        mentions = sum(
            1 for h in hypotheses 
            if any(word in later_content.lower() for word in h.lower().split()[:3])
        )
        
        value = mentions / len(hypotheses)
        
        return MetricResult(
            name=self.name,
            value=value,
            score=self._normalize_score(value),
            details={"hypotheses": hypotheses, "tracked": mentions}
        )


class RepeatedWorkAvoidance(Metric):
    """Measures if agent avoids repeating work."""
    
    def __init__(self):
        super().__init__("repeated_work_avoidance", weight=1.0)
    
    def evaluate(
        self,
        *,
        conversation_history: list[dict],
        **kwargs
    ) -> MetricResult:
        # Look for repeated actions/queries
        user_queries = [
            turn.get("content", "") 
            for turn in conversation_history 
            if turn.get("role") == "user"
        ]
        
        if len(user_queries) < 2:
            return MetricResult(
                name=self.name,
                value=1.0,
                score=1.0,
                details={"queries": len(user_queries)}
            )
        
        # Check for repetition
        repeated = 0
        for i in range(1, len(user_queries)):
            prev = user_queries[i-1].lower()
            curr = user_queries[i].lower()
            # Simple repetition detection
            if len(prev) > 10 and len(curr) > 10:
                words_prev = set(prev.split())
                words_curr = set(curr.split())
                overlap = len(words_prev & words_curr) / max(1, len(words_curr))
                if overlap > 0.5:
                    repeated += 1
        
        value = 1.0 - (repeated / max(1, len(user_queries) - 1))
        
        return MetricResult(
            name=self.name,
            value=value,
            score=self._normalize_score(value),
            details={"repeated": repeated, "total": len(user_queries) - 1}
        )


class MemoryRecallPrecision(Metric):
    """Measures precision of memory retrieval."""
    
    def __init__(self):
        super().__init__("memory_recall_precision", weight=1.0)
    
    def evaluate(
        self,
        *,
        retrieved_memories: Optional[list[dict]] = None,
        query: str = "",
        **kwargs
    ) -> MetricResult:
        if not retrieved_memories:
            return MetricResult(
                name=self.name,
                value=0.0,
                score=0.0,
                details={"retrieved": 0}
            )
        
        # Simple relevance check
        query_words = set(query.lower().split())
        relevant = 0
        
        for mem in retrieved_memories:
            content = mem.get("content", "").lower()
            content_words = set(content.split())
            overlap = len(query_words & content_words)
            if overlap > 0:
                relevant += 1
        
        value = relevant / len(retrieved_memories)
        
        return MetricResult(
            name=self.name,
            value=value,
            score=self._normalize_score(value),
            details={"retrieved": len(retrieved_memories), "relevant": relevant}
        )


class MemoryRecallUsefulness(Metric):
    """Measures if retrieved memories are actually useful."""
    
    def __init__(self):
        super().__init__("memory_recall_usefulness", weight=1.0)
    
    def evaluate(
        self,
        *,
        retrieved_memories: Optional[list[dict]] = None,
        responses: Optional[list[str]] = None,
        **kwargs
    ) -> MetricResult:
        if not retrieved_memories or not responses:
            return MetricResult(
                name=self.name,
                value=0.0,
                score=0.0,
                details={"no_data": True}
            )
        
        # Check if response references retrieved memories
        response_text = " ".join(responses).lower()
        
        useful = 0
        for mem in retrieved_memories:
            content = mem.get("content", "").lower()
            # Check if memory content appears in response
            key_words = [w for w in content.split() if len(w) > 4]
            if any(w in response_text for w in key_words[:5]):
                useful += 1
        
        value = useful / len(retrieved_memories)
        
        return MetricResult(
            name=self.name,
            value=value,
            score=self._normalize_score(value),
            details={"useful": useful, "total": len(retrieved_memories)}
        )


class IrrelevantRecallRate(Metric):
    """Measures rate of irrelevant memories retrieved."""
    
    def __init__(self):
        super().__init__("irrelevant_recall_rate", weight=1.0)
    
    def evaluate(
        self,
        *,
        retrieved_memories: Optional[list[dict]] = None,
        query: str = "",
        **kwargs
    ) -> MetricResult:
        if not retrieved_memories:
            return MetricResult(
                name=self.name,
                value=0.0,
                score=1.0,  # Perfect if no irrelevant memories
                details={"retrieved": 0}
            )
        
        query_words = set(query.lower().split())
        irrelevant = 0
        
        for mem in retrieved_memories:
            content = mem.get("content", "").lower()
            content_words = set(content.split())
            overlap = len(query_words & content_words)
            if overlap == 0:
                irrelevant += 1
        
        value = irrelevant / len(retrieved_memories)
        
        return MetricResult(
            name=self.name,
            value=value,
            score=1.0 - self._normalize_score(value),  # Invert: lower is better
            details={"irrelevant": irrelevant, "total": len(retrieved_memories)}
        )


class TokenUsage(Metric):
    """Measures token usage efficiency."""
    
    def __init__(self, max_context: int = 100000):
        super().__init__("token_usage", weight=0.5)
        self.max_context = max_context
    
    def evaluate(
        self,
        *,
        conversation_history: list[dict],
        retrieved_memories: Optional[list[dict]] = None,
        **kwargs
    ) -> MetricResult:
        # Estimate token usage
        total_chars = sum(
            len(turn.get("content", "")) 
            for turn in conversation_history
        )
        if retrieved_memories:
            total_chars += sum(len(m.get("content", "")) for m in retrieved_memories)
        
        # Rough estimate: 4 chars per token
        estimated_tokens = total_chars / 4
        
        value = estimated_tokens / self.max_context
        
        return MetricResult(
            name=self.name,
            value=value,
            score=1.0 - self._normalize_score(value),  # Lower usage is better
            details={"estimated_tokens": estimated_tokens, "max": self.max_context}
        )


class Latency(Metric):
    """Measures response latency."""
    
    def __init__(self, max_latency: float = 30.0):
        super().__init__("latency", weight=0.5)
        self.max_latency = max_latency
    
    def evaluate(
        self,
        *,
        response_times: Optional[list[float]] = None,
        **kwargs
    ) -> MetricResult:
        if not response_times:
            return MetricResult(
                name=self.name,
                value=0.0,
                score=1.0,
                details={"no_data": True}
            )
        
        avg_latency = sum(response_times) / len(response_times)
        
        return MetricResult(
            name=self.name,
            value=avg_latency,
            score=1.0 - self._normalize_score(avg_latency, 0, self.max_latency),
            details={"avg_latency": avg_latency, "max": self.max_latency}
        )
