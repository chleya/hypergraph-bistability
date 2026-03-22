"""Main evaluator for agent memory performance."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
import json
import time
from pathlib import Path

from .metrics import (
    Metric,
    MetricResult,
    TaskContinuation,
    BlockerPreservation,
    DecisionContinuity,
    ProcedureContinuity,
    ConflictContinuity,
    RepeatedWorkAvoidance,
    MemoryRecallPrecision,
    MemoryRecallUsefulness,
    IrrelevantRecallRate,
    TokenUsage,
    Latency,
)
from .scenarios import Scenario, get_scenario


@dataclass
class EvalResult:
    """Result of a complete evaluation run."""
    
    scenario_name: str
    timestamp: float = field(default_factory=time.time)
    metrics: list[MetricResult] = field(default_factory=list)
    overall_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "scenario_name": self.scenario_name,
            "timestamp": self.timestamp,
            "metrics": [m.to_dict() for m in self.metrics],
            "overall_score": self.overall_score,
            "metadata": self.metadata,
        }
    
    def save(self, path: str | Path) -> None:
        """Save result to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @property
    def summary(self) -> str:
        """Get a human-readable summary."""
        lines = [f"Scenario: {self.scenario_name}", f"Overall Score: {self.overall_score:.2%}"]
        for metric in self.metrics:
            lines.append(f"  - {metric.name}: {metric.score:.2%}")
        return "\n".join(lines)


class Evaluator:
    """Main evaluator class for running agent memory evaluations."""
    
    def __init__(
        self,
        *,
        metrics: Optional[list[Metric]] = None,
        baseline_agent: Optional[Any] = None,
    ):
        # Default metrics if none provided
        self.metrics = metrics or [
            TaskContinuation(),
            BlockerPreservation(),
            DecisionContinuity(),
            ProcedureContinuity(),
            ConflictContinuity(),
            RepeatedWorkAvoidance(),
            MemoryRecallPrecision(),
            MemoryRecallUsefulness(),
            IrrelevantRecallRate(),
            TokenUsage(),
            Latency(),
        ]
        self.baseline_agent = baseline_agent
    
    def evaluate(
        self,
        scenario: Scenario,
        *,
        agent: Optional[Any] = None,
        memory_state: Optional[dict] = None,
        retrieved_memories: Optional[list[dict]] = None,
        response_times: Optional[list[float]] = None,
        **kwargs
    ) -> EvalResult:
        """Run evaluation on a scenario."""
        results = []
        
        # Extract conversation from scenario
        conversation = scenario.turns
        responses = [
            turn.get("content", "") 
            for turn in conversation 
            if turn.get("role") == "assistant"
        ]
        
        # Run each metric
        for metric in self.metrics:
            try:
                result = metric.evaluate(
                    conversation_history=conversation,
                    memory_state=memory_state,
                    retrieved_memories=retrieved_memories,
                    responses=responses,
                    response_times=response_times,
                    ground_truth=scenario.ground_truth,
                    **kwargs
                )
                results.append(result)
            except Exception as e:
                # Log error but continue
                results.append(MetricResult(
                    name=metric.name,
                    value=0.0,
                    score=0.0,
                    details={"error": str(e)}
                ))
        
        # Calculate overall score
        if results:
            weights = [m.weight for m in self.metrics]
            total_weight = sum(weights)
            overall = sum(
                r.score * m.weight 
                for r, m in zip(results, self.metrics)
            ) / total_weight if total_weight > 0 else 0.0
        else:
            overall = 0.0
        
        return EvalResult(
            scenario_name=scenario.name,
            metrics=results,
            overall_score=overall,
            metadata={
                "scenario_description": scenario.description,
                "num_turns": len(conversation),
                "expected_memories": scenario.expected_memories,
            }
        )
    
    def evaluate_scenario_by_name(
        self,
        scenario_name: str,
        **kwargs
    ) -> EvalResult:
        """Evaluate a scenario by name."""
        scenario = get_scenario(scenario_name)
        if not scenario:
            raise ValueError(f"Unknown scenario: {scenario_name}")
        return self.evaluate(scenario, **kwargs)
    
    def run_regression(
        self,
        scenarios: list[str],
        *,
        agent: Optional[Any] = None,
        output_dir: Optional[Path] = None,
    ) -> list[EvalResult]:
        """Run regression evaluation on multiple scenarios."""
        results = []
        
        for scenario_name in scenarios:
            try:
                result = self.evaluate_scenario_by_name(scenario_name, agent=agent)
                results.append(result)
                
                # Save individual result
                if output_dir:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    result.save(output_dir / f"{scenario_name}_result.json")
                    
            except Exception as e:
                print(f"Error evaluating {scenario_name}: {e}")
                results.append(EvalResult(
                    scenario_name=scenario_name,
                    overall_score=0.0,
                    metadata={"error": str(e)}
                ))
        
        return results
    
    def compare_with_baseline(
        self,
        scenario: Scenario,
        *,
        agent: Any,
        baseline_results: list[EvalResult],
    ) -> dict[str, Any]:
        """Compare agent with baseline."""
        agent_result = self.evaluate(scenario, agent=agent)
        
        comparison = {
            "scenario": scenario.name,
            "agent_score": agent_result.overall_score,
            "baseline_scores": [r.overall_score for r in baseline_results],
            "baseline_avg": sum(r.overall_score for r in baseline_results) / len(baseline_results) if baseline_results else 0,
            "improvement": agent_result.overall_score - (
                sum(r.overall_score for r in baseline_results) / len(baseline_results) 
                if baseline_results else 0
            ),
            "metric_comparison": {}
        }
        
        # Compare individual metrics
        for metric in self.metrics:
            agent_metric = next(
                (m for m in agent_result.metrics if m.name == metric.name),
                None
            )
            baseline_metrics = [
                m for m in baseline_results 
                for bm in m.metrics 
                if bm.name == metric.name
            ]
            
            if agent_metric and baseline_metrics:
                baseline_avg = sum(m.score for m in baseline_metrics) / len(baseline_metrics)
                comparison["metric_comparison"][metric.name] = {
                    "agent": agent_metric.score,
                    "baseline": baseline_avg,
                    "improvement": agent_metric.score - baseline_avg
                }
        
        return comparison


def create_evaluator(
    *,
    include_memory_metrics: bool = True,
    include_performance_metrics: bool = True,
) -> Evaluator:
    """Create an evaluator with common metric sets."""
    metrics = []
    
    if include_memory_metrics:
        metrics.extend([
            TaskContinuation(),
            BlockerPreservation(),
            DecisionContinuity(),
            ProcedureContinuity(),
            ConflictContinuity(),
            RepeatedWorkAvoidance(),
        ])
    
    if include_performance_metrics:
        metrics.extend([
            MemoryRecallPrecision(),
            MemoryRecallUsefulness(),
            IrrelevantRecallRate(),
            TokenUsage(),
            Latency(),
        ])
    
    return Evaluator(metrics=metrics)


# Convenience function for quick evaluation
def quick_eval(
    scenario_name: str,
    *,
    agent: Optional[Any] = None,
    memory_state: Optional[dict] = None,
    retrieved_memories: Optional[list[dict]] = None,
) -> EvalResult:
    """Quickly evaluate a scenario."""
    evaluator = create_evaluator()
    return evaluator.evaluate_scenario_by_name(
        scenario_name,
        agent=agent,
        memory_state=memory_state,
        retrieved_memories=retrieved_memories,
    )
