"""
Adaptive Memory Controller
==========================

Provides cognitive control for the Agent Memory:
1. AdaptiveController: Automatically adjusts λ based on conversation state
2. ConflictMitigator: Handles high-conflict situations gracefully

The controller monitors conversation patterns and adjusts physics parameters
to optimize for the current task (focused vs exploratory, etc.)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CognitiveMode(Enum):
    """Cognitive operating modes for the agent."""
    FOCUSED = "focused"       # High λ, WTA-like collapse
    EXPLORATORY = "exploratory"  # Low λ, multi-attractor
    BALANCED = "balanced"     # Moderate λ
    OSCILLATING = "oscillating"  # λ oscillation for conflict exploration


@dataclass
class ConversationState:
    """Tracks the state of a conversation."""
    turns: int = 0
    topic_changes: int = 0
    conflict_history: List[float] = field(default_factory=list)
    lambda_history: List[float] = field(default_factory=list)
    last_topic: Optional[str] = None
    consecutive_high_conflict: int = 0
    mode_switches: int = 0
    pending_decision: bool = False


@dataclass
class TriggerRule:
    """A rule that triggers mode switching based on conditions."""
    name: str
    condition: Callable[[ConversationState], bool]
    target_lambda_ratio: float
    target_mu: float = 0.0
    reasoning: str = ""


class ConflictMitigator:
    """
    Handles high-conflict situations by inducing controlled oscillation.
    
    When conflict is detected:
    1. Lower λ to allow multiple states to coexist
    2. After user clarification, raise λ to force WTA decision
    """
    
    def __init__(self, base_gamma: float = 0.01):
        self.base_gamma = base_gamma
        self.oscillation_phase: float = 0.0
        self.oscillation_amplitude: float = 0.0
        self.is_oscillating: bool = False
        self.pending_questions: List[str] = []
        self.conflict_peaks: List[Tuple[int, float]] = []
    
    def detect_conflict_peak(self, conflict_level: float, turn: int) -> bool:
        """Detect if this is a conflict peak that needs mitigation."""
        if conflict_level > 0.7:
            self.conflict_peaks.append((turn, conflict_level))
            return True
        return False
    
    def should_oscillate(self, conflict_level: float) -> bool:
        """Determine if system should enter oscillation mode."""
        return conflict_level > 0.5 and conflict_level < 0.85
    
    def compute_oscillating_lambda(self, base_lambda: float, 
                                   amplitude: float = 0.1) -> float:
        """
        Compute oscillating λ for conflict exploration.
        
        Uses sinusoidal oscillation to explore multiple attractors
        without fully collapsing or dispersing.
        """
        self.oscillation_phase += 0.2
        oscillation = amplitude * np.sin(self.oscillation_phase)
        return max(0.0, base_lambda + oscillation)
    
    def generate_clarification_question(self, conflict_level: float,
                                        context: str) -> str:
        """Generate a clarification question based on conflict type."""
        if conflict_level > 0.8:
            templates = [
                "I notice two conflicting priorities here. Which should take precedence: {context}?",
                "You've mentioned contradictory goals. Could you clarify which is more important right now?",
                "I see tension between X and Y. What's the deciding factor for your choice?"
            ]
        elif conflict_level > 0.6:
            templates = [
                "You mentioned both A and B. Are you leaning more toward one of them?",
                "I sense some uncertainty. Would you like more information before deciding?",
                "There seem to be multiple approaches here. What matters most to you?"
            ]
        else:
            templates = [
                "Would you like me to explore this further, or should I focus on a specific aspect?",
                "I can approach this from different angles. Which direction interests you?",
                "There's more to consider here. Want me to dig deeper?"
            ]
        
        import random
        question = random.choice(templates)
        self.pending_questions.append(question)
        return question.format(context=context[:50] if context else "this topic")
    
    def reset(self) -> None:
        """Reset mitigator state."""
        self.oscillation_phase = 0.0
        self.oscillation_amplitude = 0.0
        self.is_oscillating = False
        self.pending_questions = []


class AdaptiveController:
    """
    Adaptive controller that automatically adjusts λ based on conversation state.
    
    The controller monitors:
    - Topic stability (detects mode switches)
    - Conflict levels (adjusts coupling strength)
    - User intent (focused vs exploratory)
    - Decision state (forces WTA when needed)
    
    Usage:
        controller = AdaptiveController(k=4, lambda_c=0.044)
        
        # In chat loop:
        conflict = detector.detect(prompt)
        state = controller.update(prompt, conflict_level=conflict)
        
        memory.lambda_ = state.lambda_value
        memory.mu = state.mu_value
        
        if state.response_suggestion:
            response = state.response_suggestion
    """
    
    def __init__(self, k: int = 4, lambda_c: Optional[float] = None,
                 auto_mode_switch: bool = True):
        self.k = k
        self.lambda_c = lambda_c or (3 * 0.25 * 0.25 / (0.25 + 0.25 * (k - 1)))
        self.auto_mode_switch = auto_mode_switch
        
        self.conversation_state = ConversationState()
        self.mitigator = ConflictMitigator()
        
        self.current_mode = CognitiveMode.BALANCED
        self.target_lambda_ratio = 0.5
        
        self._setup_trigger_rules()
    
    def _setup_trigger_rules(self) -> None:
        """Define trigger rules for automatic mode switching."""
        self.trigger_rules: List[TriggerRule] = [
            TriggerRule(
                name="brainstorm",
                condition=lambda text: any(w in str(text or "").lower() 
                                       for w in ["brainstorm", "think about", "consider"]),
                target_lambda_ratio=0.3,
                target_mu=0.0,
                reasoning="Exploratory: brainstorming mode"
            ),
            TriggerRule(
                name="idea",
                condition=lambda text: "idea" in str(text or "").lower(),
                target_lambda_ratio=0.35,
                target_mu=0.0,
                reasoning="Exploratory: ideation mode"
            ),
            TriggerRule(
                name="decision",
                condition=lambda text: any(w in str(text or "").lower()
                                       for w in ["decide", "choose", "final", "commit"]),
                target_lambda_ratio=0.9,
                target_mu=0.0,
                reasoning="Focused: decision mode requires WTA"
            ),
            TriggerRule(
                name="technical",
                condition=lambda text: any(w in str(text or "").lower()
                                       for w in ["code", "debug", "fix", "implement", "algorithm"]),
                target_lambda_ratio=0.85,
                target_mu=0.2,
                reasoning="Focused: technical tasks need precision"
            ),
            TriggerRule(
                name="creative",
                condition=lambda text: any(w in str(text or "").lower()
                                       for w in ["write", "create", "design", "art", "story", "paint"]),
                target_lambda_ratio=0.4,
                target_mu=-0.1,
                reasoning="Creative: allow layer desynchronization"
            ),
        ]
    
    def _check_trigger_rules(self, text: str) -> Tuple[bool, Optional[CognitiveMode], float, float]:
        """Check if any trigger rule conditions are met. Returns (matched, mode, lambda_ratio, mu)."""
        if not self.auto_mode_switch:
            return False, None, 0.5, 0.0
        for rule in self.trigger_rules:
            if rule.condition(text):
                mode = CognitiveMode.EXPLORATORY if rule.target_lambda_ratio < 0.5 else CognitiveMode.FOCUSED
                return True, mode, rule.target_lambda_ratio, rule.target_mu
        return False, None, 0.5, 0.0
        """Check if any trigger rule conditions are met."""
        if not self.auto_mode_switch:
            return False
        return any(rule.condition(text) for rule in self.trigger_rules)
    
    def update(self, prompt: str, conflict_level: float = 0.0,
               topic: Optional[str] = None) -> Tuple[float, float, Optional[str]]:
        """
        Update controller state based on new input.
        
        Parameters
        ----------
        prompt : str
            User prompt
        conflict_level : float
            Detected conflict level (0.0-1.0)
        topic : str, optional
            Detected topic
            
        Returns
        -------
        Tuple[float, float, Optional[str]]
            (lambda_value, mu_value, response_suggestion)
        """
        state = self.conversation_state
        state.turns += 1
        state.conflict_history.append(conflict_level)
        
        if topic and topic != state.last_topic:
            state.topic_changes += 1
            state.last_topic = topic
        
        if conflict_level > 0.7:
            state.consecutive_high_conflict += 1
        else:
            state.consecutive_high_conflict = 0
        
        response_suggestion = None
        lambda_ratio = self.target_lambda_ratio
        mu = 0.0
        
        if self.mitigator.should_oscillate(conflict_level):
            self.current_mode = CognitiveMode.OSCILLATING
            self.mitigator.is_oscillating = True
            lambda_ratio = 0.5 + 0.2 * np.sin(state.turns * 0.3)
            response_suggestion = self.mitigator.generate_clarification_question(
                conflict_level, prompt
            )
        elif conflict_level > 0.85:
            lambda_ratio = 0.95
            self.current_mode = CognitiveMode.FOCUSED
            state.pending_decision = True
        elif conflict_level < 0.2 and state.consecutive_high_conflict > 2:
            lambda_ratio = 0.9
            self.current_mode = CognitiveMode.FOCUSED
            state.pending_decision = True
            response_suggestion = "Let me summarize and make a decision based on what you've told me."
        matched, trigger_mode, trigger_lambda, trigger_mu = self._check_trigger_rules(topic or prompt)
        if matched and trigger_mode:
            self.current_mode = trigger_mode
            lambda_ratio = trigger_lambda
            mu = trigger_mu
        elif conflict_level < 0.3:
            self.current_mode = CognitiveMode.EXPLORATORY
            lambda_ratio = 0.3
            state.pending_decision = False
        else:
            lambda_ratio = 0.5
            self.current_mode = CognitiveMode.BALANCED
        
        if self.mitigator.is_oscillating and not self.mitigator.should_oscillate(conflict_level):
            self.mitigator.reset()
            self.current_mode = CognitiveMode.BALANCED
        
        self.target_lambda_ratio = lambda_ratio
        lambda_value = lambda_ratio * self.lambda_c
        
        state.lambda_history.append(lambda_value)
        
        return lambda_value, mu, response_suggestion
    
    def force_mode(self, mode: CognitiveMode, lambda_ratio: Optional[float] = None) -> None:
        """
        Force a specific cognitive mode.
        
        Parameters
        ----------
        mode : CognitiveMode
            Target mode
        lambda_ratio : float, optional
            Override λ ratio (only used for BALANCED mode)
        """
        self.current_mode = mode
        self.conversation_state.mode_switches += 1
        
        if mode == CognitiveMode.FOCUSED:
            self.target_lambda_ratio = 0.9
        elif mode == CognitiveMode.EXPLORATORY:
            self.target_lambda_ratio = 0.3
        elif mode == CognitiveMode.BALANCED:
            self.target_lambda_ratio = lambda_ratio or 0.5
        elif mode == CognitiveMode.OSCILLATING:
            self.target_lambda_ratio = 0.5
            self.mitigator.is_oscillating = True
    
    def get_state_summary(self) -> Dict:
        """Get a summary of the current controller state."""
        state = self.conversation_state
        return {
            "mode": self.current_mode.value,
            "lambda_ratio": self.target_lambda_ratio,
            "lambda_value": self.target_lambda_ratio * self.lambda_c,
            "mu": 0.0,
            "turns": state.turns,
            "topic_changes": state.topic_changes,
            "avg_conflict": np.mean(state.conflict_history[-10:]) if state.conflict_history else 0.0,
            "pending_decision": state.pending_decision,
            "is_oscillating": self.mitigator.is_oscillating,
            "pending_questions": len(self.mitigator.pending_questions)
        }
    
    def reset(self) -> None:
        """Reset conversation state."""
        self.conversation_state = ConversationState()
        self.mitigator.reset()
        self.current_mode = CognitiveMode.BALANCED
        self.target_lambda_ratio = 0.5


def demo():
    """Demo of adaptive controller."""
    print("=" * 60)
    print("Adaptive Memory Controller Demo")
    print("=" * 60)
    
    controller = AdaptiveController(k=4, lambda_c=0.044)
    
    test_scenarios = [
        {"prompt": "I need to brainstorm some ideas for my startup", 
         "conflict": 0.2, "topic": "brainstorm"},
        {"prompt": "I'm torn between option A and option B, what do you think?", 
         "conflict": 0.6, "topic": "decision"},
        {"prompt": "Actually wait, I'm reconsidering everything", 
         "conflict": 0.75, "topic": "conflict"},
        {"prompt": "OK I've decided to go with option A, let's implement it", 
         "conflict": 0.15, "topic": "decision"},
        {"prompt": "Now let's debug this authentication issue", 
         "conflict": 0.3, "topic": "technical"},
        {"prompt": "Write me a short story about a robot", 
         "conflict": 0.1, "topic": "creative"},
    ]
    
    print("\nProcessing scenarios:")
    for i, scenario in enumerate(test_scenarios):
        lam, mu, suggestion = controller.update(
            scenario["prompt"],
            scenario["conflict"],
            scenario["topic"]
        )
        summary = controller.get_state_summary()
        
        print(f"\n[Turn {i+1}] {scenario['topic']}")
        print(f"  Prompt: {scenario['prompt'][:50]}...")
        print(f"  Conflict: {scenario['conflict']:.1%}")
        print(f"  Mode: {summary['mode']}, λ={lam:.4f}, μ={mu:.2f}")
        if suggestion:
            print(f"  Suggestion: {suggestion[:60]}...")


if __name__ == "__main__":
    demo()
