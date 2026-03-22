"""Memory decay/demotion policy for practical agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import time


@dataclass
class DecayResult:
    """Result of decay evaluation for a memory item."""
    
    memory_key: str
    should_decay: bool
    new_importance: float
    decay_reason: str
    action: str  # "keep", "demote", "remove"


class DecayPolicy:
    """Policy for decaying or demoting stale memories.
    
    The decay policy decides whether memories should be:
    - kept: actively used, high relevance
    - demoted: moved to lower priority/older storage
    - removed: completely evicted from memory
    
    Factors considered:
    - recency: time since last access
    - usage frequency: how often the memory was retrieved
    - importance: initial importance score
    - relevance: current relevance to active tasks
    - type: some types decay faster than others
    """
    
    # Decay rates for different memory types (per hour)
    DEFAULT_DECAY_RATES = {
        "context": 0.5,      # Fast decay - chat context
        "response": 0.4,     # Fast - assistant responses
        "task": 0.1,         # Slow - important tasks
        "preference": 0.05,  # Very slow - user preferences
        "fact": 0.1,         # Medium - facts
        "log": 0.3,          # Medium - logs
        "hypothesis": 0.15,   # Medium-slow - hypotheses
        "plan": 0.1,         # Slow - plans
        "decision": 0.05,     # Very slow - decisions
        "procedure": 0.08,    # Slow - procedures
    }
    
    # Importance thresholds
    KEEP_THRESHOLD = 0.5
    DEMOTE_THRESHOLD = 0.2
    REMOVE_THRESHOLD = 0.05
    
    def __init__(
        self,
        decay_rates: Optional[dict[str, float]] = None,
        keep_threshold: float = 0.5,
        demote_threshold: float = 0.2,
        remove_threshold: float = 0.05,
    ):
        self.decay_rates = decay_rates or self.DEFAULT_DECAY_RATES
        self.keep_threshold = keep_threshold
        self.demote_threshold = demote_threshold
        self.remove_threshold = remove_threshold
    
    def evaluate(
        self,
        memory_key: str,
        memory_data: dict,
        current_time: Optional[float] = None,
    ) -> DecayResult:
        """Evaluate whether a memory should decay."""
        
        current_time = current_time or time.time()
        
        # Get memory properties
        content = memory_data.get("content", "")
        kind = memory_data.get("kind", "context")
        importance = memory_data.get("importance", 0.5)
        
        # Get timestamps
        created_at = memory_data.get("created_at", current_time)
        last_accessed = memory_data.get("last_accessed", created_at)
        
        # Calculate age in hours
        age_hours = (current_time - max(created_at, last_accessed)) / 3600
        
        # Get decay rate for this type
        decay_rate = self.decay_rates.get(kind, 0.2)
        
        # Calculate decayed importance
        decayed_importance = importance * (1 - decay_rate) ** age_hours
        
        # Check access frequency bonus
        access_count = memory_data.get("access_count", 0)
        if access_count > 5:
            # Frequently accessed memories decay slower
            decayed_importance *= min(1.5, 1 + (access_count - 5) * 0.05)
        
        # Check if linked to active task
        linked_task = memory_data.get("linked_task")
        if linked_task:
            # Active task memories decay slower
            decayed_importance *= 1.2
        
        # Determine action based on decayed importance
        if decayed_importance >= self.keep_threshold:
            action = "keep"
            reason = f"importance {decayed_importance:.3f} >= keep threshold"
        elif decayed_importance >= self.demote_threshold:
            action = "demote"
            reason = f"importance {decayed_importance:.3f} >= demote threshold"
        elif decayed_importance >= self.remove_threshold:
            action = "demote"
            reason = f"importance {decayed_importance:.3f} >= remove threshold"
        else:
            action = "remove"
            reason = f"importance {decayed_importance:.3f} < remove threshold"
        
        return DecayResult(
            memory_key=memory_key,
            should_decay=action in ("demote", "remove"),
            new_importance=decayed_importance,
            decay_reason=reason,
            action=action,
        )
    
    def evaluate_batch(
        self,
        memories: dict[str, dict],
        current_time: Optional[float] = None,
    ) -> list[DecayResult]:
        """Evaluate decay for multiple memories."""
        
        results = []
        for key, data in memories.items():
            result = self.evaluate(key, data, current_time)
            results.append(result)
        
        # Sort by importance to prioritize what to keep
        results.sort(key=lambda r: r.new_importance, reverse=True)
        
        return results
    
    def should_evict(
        self,
        memories: dict[str, dict],
        max_memories: int,
        current_time: Optional[float] = None,
    ) -> list[str]:
        """Determine which memories should be evicted to stay under limit."""
        
        if len(memories) <= max_memories:
            return []
        
        # Evaluate all memories
        results = self.evaluate_batch(memories, current_time)
        
        # Sort by importance (lowest first = first to evict)
        results.sort(key=lambda r: r.new_importance)
        
        # Return keys to evict
        num_to_evict = len(memories) - max_memories
        return [r.memory_key for r in results[:num_to_evict]]
    
    def get_decay_stats(
        self,
        memories: dict[str, dict],
        current_time: Optional[float] = None,
    ) -> dict:
        """Get statistics about memory decay."""
        
        results = self.evaluate_batch(memories, current_time)
        
        stats = {
            "total": len(results),
            "keep": sum(1 for r in results if r.action == "keep"),
            "demote": sum(1 for r in results if r.action == "demote"),
            "remove": sum(1 for r in results if r.action == "remove"),
            "avg_importance": sum(r.new_importance for r in results) / len(results) if results else 0,
        }
        
        return stats


class AdaptiveDecayPolicy(DecayPolicy):
    """Decay policy that adapts based on usage patterns."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.type_stats: dict[str, dict] = {}  # Track stats per type
    
    def update_stats(self, memory_data: dict, retrieved: bool) -> None:
        """Update statistics based on retrieval."""
        
        kind = memory_data.get("kind", "context")
        
        if kind not in self.type_stats:
            self.type_stats[kind] = {
                "access_count": 0,
                "retrieval_count": 0,
                "avg_importance": 0,
            }
        
        stats = self.type_stats[kind]
        stats["access_count"] = stats.get("access_count", 0) + 1
        
        if retrieved:
            stats["retrieval_count"] = stats.get("retrieval_count", 0) + 1
        
        # Update average importance
        current_avg = stats.get("avg_importance", 0)
        count = stats["access_count"]
        new_importance = memory_data.get("importance", 0.5)
        stats["avg_importance"] = (current_avg * (count - 1) + new_importance) / count
    
    def evaluate(
        self,
        memory_key: str,
        memory_data: dict,
        current_time: Optional[float] = None,
    ) -> DecayResult:
        """Evaluate with adaptive decay rates."""
        
        kind = memory_data.get("kind", "context")
        
        # Adjust decay rate based on type statistics
        base_decay = self.decay_rates.get(kind, 0.2)
        
        if kind in self.type_stats:
            stats = self.type_stats[kind]
            retrieval_rate = stats.get("retrieval_count", 0) / max(1, stats.get("access_count", 1))
            
            # High retrieval rate = slower decay
            if retrieval_rate > 0.5:
                base_decay *= 0.5
            elif retrieval_rate > 0.3:
                base_decay *= 0.75
        
        # Temporarily modify decay rate
        original_decay = self.decay_rates.get(kind, 0.2)
        self.decay_rates[kind] = base_decay
        
        try:
            result = super().evaluate(memory_key, memory_data, current_time)
        finally:
            # Restore original decay rate
            self.decay_rates[kind] = original_decay
        
        return result


def create_decay_policy(
    policy_type: str = "default",
    **kwargs
) -> DecayPolicy:
    """Factory function to create decay policies."""
    
    if policy_type == "adaptive":
        return AdaptiveDecayPolicy(**kwargs)
    else:
        return DecayPolicy(**kwargs)
