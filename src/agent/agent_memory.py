"""
Agent Memory Module - Enhanced Version
======================================

Enhanced features:
- Automatic λ adjustment based on prompt conflict detection
- Memory persistence (save/load JSON)
- Semantic conflict keywords
- Integration helpers for LLM APIs
"""

import numpy as np
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass
import json


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


class ConflictDetector:
    """
    Detect semantic conflict level from prompt text.
    
    Uses keyword matching for conflict signals.
    For production, replace with LLM-based classification.
    """
    
    CONFLICT_KEYWORDS = [
        "but", "however", "although", "despite", "instead",
        "contradict", "opposite", "different", "change",
        "unfortunately", "actually", "wait"
    ]
    
    FOCUS_KEYWORDS = [
        "exactly", "precisely", "specifically", "only",
        "must", "need", "focus", "strict"
    ]
    
    MULTI_CONTEXT_KEYWORDS = [
        "both", "and", "also", "additionally", "moreover",
        "meanwhile", "simultaneously", "together"
    ]
    
    @classmethod
    def detect_conflict_level(cls, prompt: str) -> float:
        """
        Detect conflict level from 0.0 (no conflict) to 1.0 (high conflict).
        
        Returns
        -------
        float
            Conflict level [0, 1]
        """
        prompt_lower = prompt.lower()
        words = prompt_lower.split()
        
        conflict_signals = sum(1 for kw in cls.CONFLICT_KEYWORDS if kw in prompt_lower)
        focus_signals = sum(1 for kw in cls.FOCUS_KEYWORDS if kw in prompt_lower)
        multi_signals = sum(1 for kw in cls.MULTI_CONTEXT_KEYWORDS if kw in prompt_lower)
        
        conflict_level = conflict_signals * 0.3 + focus_signals * 0.1
        
        if multi_signals > 1:
            conflict_level -= multi_signals * 0.05
        
        return max(0.0, min(1.0, conflict_level))
    
    @classmethod
    def suggest_lambda(cls, conflict_level: float) -> Tuple[float, str]:
        """
        Suggest λ based on conflict level.
        
        Parameters
        ----------
        conflict_level : float
            From 0.0 (calm) to 1.0 (highly conflicting)
        
        Returns
        -------
        Tuple[float, str]
            (lambda_value, reasoning)
        """
        if conflict_level < 0.1:
            return 0.01, "calm, allow multi-mode coexistence"
        elif conflict_level < 0.3:
            return 0.03, "mild conflict, slight focus"
        elif conflict_level < 0.5:
            return 0.06, "moderate conflict, moderate focus"
        elif conflict_level < 0.7:
            return 0.10, "high conflict, strong focus"
        else:
            return 0.15, "very high conflict, near WTA collapse"


class AgentMemory:
    """
    Agent memory module using hypergraph multistability.
    """
    
    def __init__(
        self,
        k: int = 3,
        L: int = 2,
        alpha: float = 1.0,
        a_bistable: float = 0.5,
        lambda_: float = 0.0,
        mu: float = 0.0,
        name: str = "default",
        auto_adjust: bool = True
    ):
        self.k = k
        self.L = L
        self.alpha = alpha
        self.a = a_bistable
        self.lambda_ = lambda_
        self.mu = mu
        self.name = name
        self.auto_adjust = auto_adjust
        
        self.M = np.full((k, L), 0.5)
        self.group_labels = [f"group_{i}" for i in range(k)]
        self.layer_labels = [f"layer_{l}" for l in range(L)]
        self.current_mode = "neutral"
        self.mode_history = []
        self.content_map: Dict[str, str] = {}
        self.conflict_history = []
        
        try:
            from hypergraph_control import compute_lambda_c
            self._compute_lambda_c = compute_lambda_c
        except ImportError:
            self._compute_lambda_c = None
    
    def bistable_dynamics(self, m: np.ndarray) -> np.ndarray:
        return m * (1 - m) * (2 * m - 1)
    
    def step(self, dt: float = 0.1, n_steps: int = 1) -> None:
        for _ in range(n_steps):
            dM = np.zeros_like(self.M)
            for i in range(self.k):
                for l in range(self.L):
                    m = self.M[i, l]
                    bistable = self.bistable_dynamics(m)
                    gc = self.lambda_ * (np.mean(self.M[:, l]) - m)
                    other_layers = [self.M[i, j] for j in range(self.L) if j != l]
                    other_mean = np.mean(other_layers) if other_layers else 0.0
                    lc = self.mu * (other_mean - m)
                    dM[i, l] = bistable + gc + lc
            self.M = np.clip(self.M + dM * dt, 0, 1)
    
    def read(self, query: Optional[str] = None) -> MemoryState:
        pattern = self._binarize()
        return MemoryState(
            groups=np.mean(pattern, axis=1),
            layers=np.mean(pattern, axis=0),
            pattern=pattern,
            n_active=int(np.sum(pattern)),
            mode_label=self.current_mode
        )
    
    def write(
        self,
        content: str,
        group: int,
        layer: int,
        activate: bool = True,
        n_steps: int = 50
    ) -> None:
        if group < 0 or group >= self.k:
            raise ValueError(f"group must be in [0, {self.k-1}]")
        if layer < 0 or layer >= self.L:
            raise ValueError(f"layer must be in [0, {self.L-1}]")
        
        key = f"({group},{layer})"
        self.content_map[key] = content
        
        target = HIGH_TARGET if activate else LOW_TARGET
        self.M[group, layer] = target
        self.step(dt=DEFAULT_DT, n_steps=n_steps)
    
    def process_prompt(self, prompt: str, auto_adjust: Optional[bool] = None) -> Dict:
        """
        Process a prompt and optionally adjust memory state.
        
        Parameters
        ----------
        prompt : str
            User prompt
        auto_adjust : bool, optional
            Override auto_adjust setting
        
        Returns
        -------
        dict
            Processing results including conflict level, suggested λ
        """
        if auto_adjust is None:
            auto_adjust = self.auto_adjust
        
        conflict_level = ConflictDetector.detect_conflict_level(prompt)
        suggested_lambda, reasoning = ConflictDetector.suggest_lambda(conflict_level)
        
        result = {
            "prompt": prompt,
            "conflict_level": conflict_level,
            "suggested_lambda": suggested_lambda,
            "reasoning": reasoning,
            "auto_adjusted": False
        }
        
        if auto_adjust and conflict_level > 0.1:
            old_lambda = self.lambda_
            self.lambda_ = suggested_lambda
            self.step(dt=DEFAULT_DT, n_steps=20)
            result["auto_adjusted"] = True
            result["old_lambda"] = old_lambda
            result["new_lambda"] = suggested_lambda
        
        self.conflict_history.append(conflict_level)
        return result
    
    def switch_mode(
        self,
        mode: str,
        lambda_override: Optional[float] = None,
        mu_override: Optional[float] = None
    ) -> None:
        presets = {
            "neutral": (0.0, 0.0),
            "exploratory": (0.01, 0.0),
            "focused": (0.15, 0.0),
            "sync": (0.05, 0.3),
            "creative": (0.05, -0.3),
            "professional": (0.08, 0.0),
            "casual": (0.02, 0.1),
        }
        
        if mode in presets:
            lam, mu = presets[mode]
            if lambda_override is not None:
                lam = lambda_override
            if mu_override is not None:
                mu = mu_override
            self.lambda_ = lam
            self.mu = mu
        else:
            if lambda_override is None or mu_override is None:
                raise ValueError(f"Unknown mode '{mode}'")
            self.lambda_ = lambda_override
            self.mu = mu_override
        
        self.current_mode = mode
        self.mode_history.append(mode)
        self.step(dt=0.1, n_steps=20)
    
    def get_active_pattern(self) -> str:
        state = self.read()
        lines = [f"Mode: {self.current_mode}"]
        
        for i in range(self.k):
            active = state.groups[i] > 0.5
            key = f"({i},"
            content_hints = [v for k, v in self.content_map.items() if k.startswith(key)]
            
            if active:
                active_layers = [
                    self.layer_labels[l] for l in range(self.L)
                    if state.pattern[i, l] > 0.5
                ]
                hint = f" [{content_hints[0][:30]}...]" if content_hints else ""
                lines.append(f"  {self.group_labels[i]}: ACTIVE ({', '.join(active_layers)}){hint}")
            else:
                lines.append(f"  {self.group_labels[i]}: inactive")
        
        lines.append(f"Total active: {state.n_active}/{self.k*self.L}")
        return "\n".join(lines)
    
    def get_context_for_llm(self) -> str:
        """
        Get memory state as a string for LLM context.
        """
        state = self.read()
        ctx_lines = [f"[Memory State: {self.current_mode}]"]
        
        active_groups = []
        for i in range(self.k):
            if state.groups[i] > 0.5:
                key_prefix = f"({i},"
                memories = [v for k, v in self.content_map.items() if k.startswith(key_prefix)]
                if memories:
                    active_groups.append(f"{self.group_labels[i]}: {', '.join(memories[:2])}")
                else:
                    active_groups.append(self.group_labels[i])
        
        if active_groups:
            ctx_lines.append(f"Active contexts: {', '.join(active_groups)}")
        else:
            ctx_lines.append("No specific context activated.")
        
        lc = self.get_lambda_c()
        if lc:
            ctx_lines.append(f"(Focus threshold λ_c≈{lc:.3f}, current λ={self.lambda_:.3f})")
        
        return "\n".join(ctx_lines)
    
    def _binarize(self) -> np.ndarray:
        binary = np.full((self.k, self.L), 0.5)
        binary[self.M > HIGH_THRESHOLD] = 1.0
        binary[self.M < LOW_THRESHOLD] = 0.0
        return binary
    
    def get_lambda_c(self) -> Optional[float]:
        if self._compute_lambda_c is None:
            return None
        return self._compute_lambda_c(self.k, n_high=1)
    
    def save(self, filepath: Optional[str] = None) -> str:
        """
        Save memory state to JSON file.
        """
        if filepath is None:
            filepath = f"{self.name}_memory.json"
        
        data = {
            "name": self.name,
            "k": self.k,
            "L": self.L,
            "lambda_": float(self.lambda_),
            "mu": float(self.mu),
            "current_mode": self.current_mode,
            "mode_history": self.mode_history,
            "M": self.M.tolist(),
            "content_map": self.content_map,
            "group_labels": self.group_labels,
            "layer_labels": self.layer_labels
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        return filepath
    
    @classmethod
    def load(cls, filepath: str) -> "AgentMemory":
        """
        Load memory state from JSON file.
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        mem = cls(
            k=data["k"],
            L=data["L"],
            lambda_=data["lambda_"],
            mu=data["mu"],
            name=data["name"]
        )
        mem.M = np.array(data["M"])
        mem.current_mode = data["current_mode"]
        mem.mode_history = data.get("mode_history", [])
        mem.content_map = data.get("content_map", {})
        mem.group_labels = data.get("group_labels", mem.group_labels)
        mem.layer_labels = data.get("layer_labels", mem.layer_labels)
        
        return mem
    
    def status(self) -> str:
        lc = self.get_lambda_c()
        lc_str = f"{lc:.4f}" if lc else "N/A"
        
        return (
            f"AgentMemory '{self.name}'\n"
            f"  k={self.k}, L={self.L} | λ={self.lambda_:.3f} (λ_c≈{lc_str}), μ={self.mu:.3f}\n"
            f"  Mode: {self.current_mode} | Active: {self.read().n_active}/{self.k*self.L}"
        )


def demo():
    print("=" * 60)
    print("Agent Memory Module - Enhanced Demo")
    print("=" * 60)
    
    mem = AgentMemory(k=3, L=2, name="assistant", auto_adjust=True)
    mem.group_labels = ["professional", "friendly", "technical"]
    mem.layer_labels = ["preferences", "context"]
    
    print("\n1. Initial state:")
    print(mem.status())
    
    print("\n2. Writing memories:")
    mem.write("Short responses preferred", group=0, layer=0)
    mem.write("Like analogies", group=1, layer=0)
    mem.write("Tech industry", group=2, layer=1)
    print(mem.get_active_pattern())
    
    print("\n3. Processing prompts with auto λ adjustment:")
    
    prompts = [
        "Tell me about the weather today",
        "I want to learn coding but I'm also tired",
        "Both options seem good, actually wait no, let me focus on the first one exactly",
    ]
    
    for prompt in prompts:
        result = mem.process_prompt(prompt)
        print(f"\nPrompt: '{prompt}'")
        print(f"  Conflict: {result['conflict_level']:.1%}")
        print(f"  λ adjusted: {result.get('old_lambda', 'N/A')} → {result.get('new_lambda', mem.lambda_):.3f}")
        print(f"  Reasoning: {result['reasoning']}")
    
    print("\n4. Memory context for LLM:")
    print(mem.get_context_for_llm())
    
    print("\n5. Save/Load test:")
    path = mem.save()
    print(f"Saved to: {path}")
    
    mem2 = AgentMemory.load(path)
    print(f"Loaded: {mem2.status()}")
    
    print("\n" + "=" * 60)
    print("Demo complete")


if __name__ == "__main__":
    demo()