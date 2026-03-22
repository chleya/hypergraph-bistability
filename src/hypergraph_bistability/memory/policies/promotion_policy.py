"""Memory promotion policy for practical agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import time


@dataclass
class PromotionResult:
    """Result of promotion evaluation for a memory item."""
    
    memory_key: str
    should_promote: bool
    new_layer: int  # 0 = working, 1 = episodic, 2 = durable
    promotion_reason: str
    confidence: float = 1.0


class PromotionPolicy:
    """Policy for promoting important memories to longer-term storage.
    
    The promotion policy decides whether memories should be promoted
    from working memory to more durable storage layers:
    
    - Layer 0 (working): Active, frequently accessed memories
    - Layer 1 (episodic): Important episodes worth keeping
    - Layer 2 (durable): Critical long-term memories (preferences, decisions)
    
    Promotion criteria:
    - frequency: how often the memory is accessed
    - importance: initial importance score
    - type: some types should always be promoted
    - age: time since creation
    - verification: whether the memory has been verified/confirmed
    """
    
    # Promotion criteria for different memory types
    AUTO_PROMOTE_TYPES = {"preference", "decision", "procedure"}  # Always promote
    NEVER_PROMOTE_TYPES = {"context", "response"}  # Never promote
    
    # Layer definitions
    LAYER_NAMES = {
        0: "working",
        1: "episodic", 
        2: "durable"
    }
    
    def __init__(
        self,
        promote_threshold: float = 0.7,
        access_count_threshold: int = 3,
        age_hours_threshold: float = 24.0,
    ):
        self.promote_threshold = promote_threshold
        self.access_count_threshold = access_count_threshold
        self.age_hours_threshold = age_hours_threshold
    
    def evaluate(
        self,
        memory_key: str,
        memory_data: dict,
        current_time: Optional[float] = None,
    ) -> PromotionResult:
        """Evaluate whether a memory should be promoted."""
        
        current_time = current_time or time.time()
        
        # Get memory properties
        content = memory_data.get("content", "")
        kind = memory_data.get("kind", "context")
        importance = memory_data.get("importance", 0.5)
        current_layer = memory_data.get("layer", 0)
        
        # Already at max layer?
        if current_layer >= 2:
            return PromotionResult(
                memory_key=memory_key,
                should_promote=False,
                new_layer=2,
                promotion_reason="already at max layer",
                confidence=1.0,
            )
        
        # Check auto-promote types
        if kind in self.AUTO_PROMOTE_TYPES:
            return PromotionResult(
                memory_key=memory_key,
                should_promote=True,
                new_layer=min(current_layer + 1, 2),
                promotion_reason=f"auto-promote type: {kind}",
                confidence=1.0,
            )
        
        # Check never-promote types
        if kind in self.NEVER_PROMOTE_TYPES:
            return PromotionResult(
                memory_key=memory_key,
                should_promote=False,
                new_layer=current_layer,
                promotion_reason=f"never-promote type: {kind}",
                confidence=1.0,
            )
        
        # Calculate promotion score
        score = 0.0
        reasons = []
        
        # Factor 1: Access frequency
        access_count = memory_data.get("access_count", 0)
        if access_count >= self.access_count_threshold:
            score += 0.4
            reasons.append(f"high access ({access_count})")
        
        # Factor 2: Importance
        if importance >= self.promote_threshold:
            score += 0.3
            reasons.append(f"high importance ({importance:.2f})")
        
        # Factor 3: Age
        created_at = memory_data.get("created_at", current_time)
        age_hours = (current_time - created_at) / 3600
        if age_hours >= self.age_hours_threshold:
            score += 0.2
            reasons.append(f"aged ({age_hours:.1f}h)")
        
        # Factor 4: Verification
        confidence_tag = memory_data.get("confidence_tag")
        if confidence_tag == "verified":
            score += 0.3
            reasons.append("verified")
        
        # Factor 5: Type-specific promotion
        if kind in {"task", "fact"}:
            score += 0.15
            reasons.append(f"promotable type: {kind}")
        
        # Determine if should promote
        should_promote = score >= 0.5
        
        if should_promote:
            new_layer = min(current_layer + 1, 2)
            reason = f"promote: {', '.join(reasons)}" if reasons else f"score={score:.2f}"
        else:
            new_layer = current_layer
            reason = f"no promote: score={score:.2f}"
        
        return PromotionResult(
            memory_key=memory_key,
            should_promote=should_promote,
            new_layer=new_layer,
            promotion_reason=reason,
            confidence=min(1.0, score),
        )
    
    def evaluate_batch(
        self,
        memories: dict[str, dict],
        current_time: Optional[float] = None,
    ) -> list[PromotionResult]:
        """Evaluate promotion for multiple memories."""
        
        results = []
        for key, data in memories.items():
            result = self.evaluate(key, data, current_time)
            results.append(result)
        
        # Sort by confidence (highest first)
        results.sort(key=lambda r: r.confidence, reverse=True)
        
        return results
    
    def get_promotions_needed(
        self,
        memories: dict[str, dict],
        current_time: Optional[float] = None,
    ) -> list[PromotionResult]:
        """Get list of memories that should be promoted."""
        
        results = self.evaluate_batch(memories, current_time)
        return [r for r in results if r.should_promote]
    
    def get_promotion_stats(
        self,
        memories: dict[str, dict],
        current_time: Optional[float] = None,
    ) -> dict:
        """Get statistics about promotion needs."""
        
        results = self.evaluate_batch(memories, current_time)
        
        layer_counts = {0: 0, 1: 0, 2: 0}
        for r in results:
            layer_counts[r.new_layer] = layer_counts.get(r.new_layer, 0) + 1
        
        stats = {
            "total": len(results),
            "should_promote": sum(1 for r in results if r.should_promote),
            "layer_distribution": layer_counts,
            "avg_confidence": sum(r.confidence for r in results) / len(results) if results else 0,
        }
        
        return stats


class SmartPromotionPolicy(PromotionPolicy):
    """Promotion policy with learned patterns."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.promotion_history: list[dict] = []
    
    def record_promotion(
        self,
        memory_key: str,
        memory_data: dict,
        promotion_result: PromotionResult,
    ) -> None:
        """Record promotion decision for learning."""
        
        self.promotion_history.append({
            "memory_key": memory_key,
            "kind": memory_data.get("kind"),
            "importance": memory_data.get("importance"),
            "access_count": memory_data.get("access_count"),
            "promoted": promotion_result.should_promote,
            "timestamp": time.time(),
        })
        
        # Keep only recent history
        if len(self.promotion_history) > 1000:
            self.promotion_history = self.promotion_history[-1000:]
    
    def get_type_promotion_rates(self) -> dict[str, float]:
        """Get promotion rates by memory type."""
        
        type_counts: dict[str, dict] = {}
        
        for record in self.promotion_history:
            kind = record.get("kind", "unknown")
            if kind not in type_counts:
                type_counts[kind] = {"total": 0, "promoted": 0}
            
            type_counts[kind]["total"] += 1
            if record.get("promoted"):
                type_counts[kind]["promoted"] += 1
        
        # Calculate rates
        rates = {}
        for kind, counts in type_counts.items():
            if counts["total"] > 0:
                rates[kind] = counts["promoted"] / counts["total"]
        
        return rates
    
    def evaluate(
        self,
        memory_key: str,
        memory_data: dict,
        current_time: Optional[float] = None,
    ) -> PromotionResult:
        """Evaluate with learned patterns."""
        
        result = super().evaluate(memory_key, memory_data, current_time)
        
        # Adjust based on learned patterns
        kind = memory_data.get("kind")
        type_rates = self.get_type_promotion_rates()
        
        if kind in type_rates:
            rate = type_rates[kind]
            
            # If this type rarely promotes, be more conservative
            if rate < 0.3 and result.should_promote:
                # Boost confidence slightly
                result = PromotionResult(
                    memory_key=result.memory_key,
                    should_promote=result.should_promote,
                    new_layer=result.new_layer,
                    promotion_reason=result.promotion_reason + f", type_rate={rate:.2f}",
                    confidence=result.confidence * 0.9,  # Reduce confidence
                )
            
            # If this type frequently promotes, be more aggressive
            elif rate > 0.7 and not result.should_promote:
                # Lower threshold slightly
                if result.confidence > 0.3:
                    result = PromotionResult(
                        memory_key=result.memory_key,
                        should_promote=True,
                        new_layer=result.new_layer,
                        promotion_reason=result.promotion_reason + f", type_rate={rate:.2f}",
                        confidence=result.confidence * 0.8,
                    )
        
        return result


def create_promotion_policy(
    policy_type: str = "default",
    **kwargs
) -> PromotionPolicy:
    """Factory function to create promotion policies."""
    
    if policy_type == "smart":
        return SmartPromotionPolicy(**kwargs)
    else:
        return PromotionPolicy(**kwargs)
