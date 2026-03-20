"""
Enhanced Agent Memory Module with LLM-based Conflict Detection
=============================================================

Features:
- LLM-based semantic conflict detection
- Contextual memory retrieval
- Enhanced mode switching with learned patterns
- Physics-based λ_c control (CollapsController)
"""

import numpy as np
from typing import Optional, List, Tuple, Dict, Callable
from dataclasses import dataclass, field
import json
import os

from agent.agent_memory import CollapsController, _regime_name


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


class LLMConflictDetector:
    """
    LLM-based conflict detection for prompts.
    
    Uses an LLM to classify whether a prompt contains
    conflicting or competing signals.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.model = model
        self._client = None
        self._has_openai = False
        
        if api_key or os.environ.get("OPENAI_API_KEY"):
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
                self._has_openai = True
            except ImportError:
                pass
    
    def detect_conflict(self, prompt: str) -> Tuple[float, str]:
        """
        Detect conflict level in a prompt.
        
        Parameters
        ----------
        prompt : str
            User prompt
        
        Returns
        -------
        Tuple[float, str]
            (conflict_level, reasoning)
            conflict_level: 0.0 (no conflict) to 1.0 (high conflict)
            reasoning: explanation of the classification
        """
        if self._has_openai:
            return self._detect_with_llm(prompt)
        else:
            return self._detect_fallback(prompt)
    
    def _detect_with_llm(self, prompt: str) -> Tuple[float, str]:
        """Use OpenAI to classify conflict."""
        system_prompt = """You are a conflict detector for AI agent prompts.
Analyze the user prompt and classify the level of conflicting or competing signals.

Conflicting signals include:
- "I want A but I also want B"
- "On one hand... on the other hand"
- "However", "although", "despite"
- Multiple competing goals or contexts
- Hesitation ("wait", "actually", "no wait")

Rate conflict level:
- 0.0-0.2: Clear, single intent (e.g., "Help me write code")
- 0.3-0.5: Mild conflict or ambiguity (e.g., "I'd prefer A, but B could work")
- 0.6-0.8: Strong conflict (e.g., "I want to exercise but I'm too tired")
- 0.9-1.0: Very high conflict or paralysis (multiple equally weighted competing goals)

Respond ONLY with JSON:
{"level": 0.0-1.0, "reasoning": "brief explanation"}"""
        
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=100
            )
            
            result_text = response.choices[0].message.content.strip()
            
            import re
            match = re.search(r'"level":\s*([0-9.]+)', result_text)
            level_match = re.search(r'"reasoning":\s*"([^"]+)"', result_text)
            
            if match:
                level = float(match.group(1))
                reasoning = level_match.group(1) if level_match else "No reasoning provided"
                return level, reasoning
            else:
                return self._detect_fallback(prompt)
                
        except Exception as e:
            print(f"LLM detection failed: {e}, falling back to keyword method")
            return self._detect_fallback(prompt)
    
    def _detect_fallback(self, prompt: str) -> Tuple[float, str]:
        """Fallback keyword-based detection."""
        prompt_lower = prompt.lower()
        
        conflict_keywords = [
            "but", "however", "although", "despite", "instead",
            "contradict", "opposite", "different", "unfortunately",
            "actually", "wait", "no wait", "on the other hand",
            "either", "or", "neither", "vs", "versus"
        ]
        
        focus_keywords = [
            "exactly", "precisely", "specifically", "only",
            "must", "need", "focus", "strict"
        ]
        
        multi_keywords = [
            "both", "and", "also", "additionally", "moreover"
        ]
        
        conflict_count = sum(1 for kw in conflict_keywords if kw in prompt_lower)
        focus_count = sum(1 for kw in focus_keywords if kw in prompt_lower)
        multi_count = sum(1 for kw in multi_keywords if kw in prompt_lower)
        
        level = min(1.0, conflict_count * 0.15 + focus_count * 0.05)
        
        if multi_count > 1:
            level = max(0, level - multi_count * 0.05)
        
        if conflict_count >= 3:
            level = min(1.0, level + 0.3)
        
        reasoning_parts = []
        if conflict_count > 0:
            reasoning_parts.append(f"{conflict_count} conflict keywords")
        if focus_count > 0:
            reasoning_parts.append(f"{focus_count} focus keywords")
        if multi_count > 0:
            reasoning_parts.append(f"{multi_count} multi-context keywords")
        
        reasoning = "; ".join(reasoning_parts) if reasoning_parts else "No specific signals"
        
        return level, reasoning


class SemanticMemoryMapper:
    """
    Maps memory slots to semantic content using LLM.
    
    Enables flexible memory writing without hardcoding slot indices.
    """
    
    def __init__(self, k: int, L: int, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.k = k
        self.L = L
        self.group_labels = [f"group_{i}" for i in range(k)]
        self.layer_labels = [f"layer_{l}" for l in range(L)]
        self.content_map: Dict[str, str] = {}
        
        self._client = None
        self._has_openai = False
        
        if api_key or os.environ.get("OPENAI_API_KEY"):
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
                self._has_openai = True
            except ImportError:
                pass
    
    def find_slot(self, content: str, action: str = "write") -> Tuple[int, int]:
        """
        Find the best slot for content using semantic understanding.
        
        Parameters
        ----------
        content : str
            What to store/retrieve
        action : str
            "write" or "read"
        
        Returns
        -------
        Tuple[int, int]
            (group_index, layer_index)
        """
        if self._has_openai:
            return self._find_slot_llm(content, action)
        else:
            return self._find_slot_keyword(content, action)
    
    def _find_slot_llm(self, content: str, action: str) -> Tuple[int, int]:
        """Use LLM to find the best slot."""
        system_prompt = f"""You are a memory mapper for an AI agent with {self.k} groups and {self.L} layers.

Groups represent: personas, roles, or contexts (e.g., "work", "personal", "technical")
Layers represent: aspects or dimensions (e.g., "preferences", "context", "goals")

Given content to store or retrieve, output ONLY JSON:
{{"group": 0-{self.k-1}, "layer": 0-{self.L-1}, "reasoning": "brief"}}"""
        
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"{action.upper()}: {content}"}
                ],
                temperature=0.0,
                max_tokens=100
            )
            
            result_text = response.choices[0].message.content.strip()
            
            import re
            group_match = re.search(r'"group":\s*([0-9]+)', result_text)
            layer_match = re.search(r'"layer":\s*([0-9]+)', result_text)
            
            if group_match and layer_match:
                g = int(group_match.group(1))
                l = int(layer_match.group(1))
                return min(g, self.k-1), min(l, self.L-1)
            
            return self._find_slot_keyword(content, action)
            
        except Exception as e:
            print(f"LLM slot finding failed: {e}")
            return self._find_slot_keyword(content, action)
    
    def _find_slot_keyword(self, content: str, action: str) -> Tuple[int, int]:
        """Fallback keyword-based slot finding."""
        content_lower = content.lower()
        
        group_map = {
            "work": 0, "professional": 0, "job": 0, "business": 0,
            "personal": 1, "friend": 1, "casual": 1, "social": 1,
            "technical": 2, "code": 2, "programming": 2, "science": 2
        }
        
        layer_map = {
            "prefer": 0, "preference": 0, "like": 0, "want": 0,
            "context": 1, "situation": 1, "background": 1,
            "goal": 2, "aim": 2, "objective": 2
        }
        
        best_group, best_layer = 0, 0
        
        for keyword, idx in group_map.items():
            if keyword in content_lower:
                best_group = idx
                break
        
        for keyword, idx in layer_map.items():
            if keyword in content_lower:
                best_layer = idx
                break
        
        return best_group, best_layer


class AgentMemoryEnhanced:
    """
    Enhanced Agent Memory with LLM integration.
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
        use_llm_detector: bool = False,
        use_llm_mapper: bool = False,
        api_key: Optional[str] = None,
        use_physics_control: bool = True
    ):
        self.k = k
        self.L = L
        self.alpha = alpha
        self.a = a_bistable
        self.lambda_ = lambda_
        self.mu = mu
        self.name = name
        self.use_physics_control = use_physics_control
        
        self.M = np.full((k, L), 0.5)
        self.group_labels = [f"group_{i}" for i in range(k)]
        self.layer_labels = [f"layer_{l}" for l in range(L)]
        self.current_mode = "neutral"
        self.mode_history = []
        self.content_map: Dict[str, str] = {}
        self.conflict_history = []
        
        self.use_llm_detector = use_llm_detector
        self.use_llm_mapper = use_llm_mapper
        
        if use_llm_detector:
            self.detector = LLMConflictDetector(api_key=api_key)
        else:
            from agent.agent_memory import ConflictDetector
            self.detector = ConflictDetector()
        
        if use_llm_mapper:
            self.mapper = SemanticMemoryMapper(k, L, api_key=api_key)
        else:
            self.mapper = None
        
        try:
            from hypergraph_control import compute_lambda_c
            self._compute_lambda_c = compute_lambda_c
        except ImportError:
            self._compute_lambda_c = None
        
        self._has_scipy = True
        try:
            from scipy.integrate import odeint
        except ImportError:
            self._has_scipy = False
    
    def bistable_dynamics(self, m: np.ndarray) -> np.ndarray:
        return m * (1 - m) * (2 * m - 1)
    
    def _compute_dMdt(self, M_flat: np.ndarray, t: float) -> np.ndarray:
        M = M_flat.reshape((self.k, self.L))
        dM = np.zeros_like(M)
        for i in range(self.k):
            for l in range(self.L):
                m = M[i, l]
                bistable = self.bistable_dynamics(m)
                gc = self.lambda_ * (np.mean(M[:, l]) - m)
                ol = [M[i, j] for j in range(self.L) if j != l]
                om = np.mean(ol) if ol else 0.0
                dM[i, l] = bistable + gc + self.mu * (om - m)
        return dM.flatten()
    
    def step(self, dt: float = 0.1, n_steps: int = 1) -> None:
        if self._has_scipy and n_steps >= 10:
            total_t = dt * n_steps
            t = np.linspace(0, total_t, min(n_steps + 1, 50))
            M0_flat = self.M.flatten()
            sol = __import__('scipy.integrate', fromlist=['odeint']).odeint(
                self._compute_dMdt, M0_flat, t
            )
            self.M = np.clip(sol[-1].reshape((self.k, self.L)), 0, 1)
        else:
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
    
    def write_semantic(self, content: str, activate: bool = True) -> None:
        """
        Write content using semantic slot finding.
        """
        if self.mapper:
            group, layer = self.mapper.find_slot(content, "write")
        else:
            group, layer = 0, 0
        
        self.write(content, group, layer, activate)
    
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
    
    def process_prompt(
        self,
        prompt: str,
        use_llm: Optional[bool] = None,
        target_r: float = None
    ) -> Dict:
        """
        Process prompt via physics-based λ_c control.
        
        Optionally uses LLM for semantic conflict detection (if use_llm=True
        and api_key provided), but λ adjustment always uses r = λ/λ_c.
        """
        old_lambda = self.lambda_
        
        if self.use_physics_control and self._compute_lambda_c is not None:
            if target_r is None:
                complexity = min(1.0, len(prompt.split()) / 50.0)
                target_r = 0.4 + 0.5 * complexity
            else:
                target_r = min(1.0, max(0.1, target_r))
            
            suggested_lambda, reasoning = CollapsController.suggest_lambda(target_r, self.k)
            self.lambda_ = suggested_lambda
            
            regime = _regime_name(target_r)
            result = {
                "prompt": prompt,
                "conflict_level": target_r,
                "reasoning": f"Physics-based: target r={target_r:.2f} → {reasoning}",
                "old_lambda": old_lambda,
                "new_lambda": self.lambda_,
                "regime": regime
            }
        else:
            self.lambda_ = 0.05
            result = {
                "prompt": prompt,
                "conflict_level": 0.0,
                "reasoning": "Fallback: no physics control",
                "old_lambda": old_lambda,
                "new_lambda": self.lambda_
            }
        
        self.step(dt=DEFAULT_DT, n_steps=20)
        self.conflict_history.append(result["conflict_level"])
        return result
    
    def switch_mode(self, mode: str) -> None:
        proximity_presets = {
            "neutral": (0.0, 0.0),
            "exploratory": (0.3, 0.0),
            "focused": (0.85, 0.0),
            "sync": (0.5, 0.3),
            "creative": (0.5, -0.3),
            "professional": (0.65, 0.0),
            "casual": (0.4, 0.1),
        }
        
        lc = self.get_lambda_c()
        
        if mode in proximity_presets:
            r, mu_val = proximity_presets[mode]
            if lc is not None and self.use_physics_control:
                lam = r * lc
            else:
                lam = r * 0.1
            self.lambda_ = lam
            self.mu = mu_val
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        self.current_mode = mode
        self.mode_history.append(mode)
        self.step(dt=DEFAULT_DT, n_steps=20)
    
    def get_active_pattern(self) -> str:
        state = self.read()
        lines = [f"Mode: {self.current_mode}"]
        
        for i in range(self.k):
            active = state.groups[i] > 0.5
            key_prefix = f"({i},"
            memories = [v for k, v in self.content_map.items() if k.startswith(key_prefix)]
            
            if active:
                hint = f" [{memories[0][:30]}...]" if memories else ""
                lines.append(f"  {self.group_labels[i]}: ACTIVE{hint}")
            else:
                lines.append(f"  {self.group_labels[i]}: inactive")
        
        lines.append(f"Total active: {state.n_active}/{self.k*self.L}")
        return "\n".join(lines)
    
    def get_context_for_llm(self) -> str:
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
            r = CollapsController.compute_collapse_ratio(self.lambda_, self.k)
            regime = _regime_name(r)
            ctx_lines.append(
                f"(λ={self.lambda_:.3f}, λ_c≈{lc:.3f}, r={r:.2f}, regime={regime})"
            )
        
        return "\n".join(ctx_lines)
    
    def _binarize(self) -> np.ndarray:
        if self.use_physics_control and self._compute_lambda_c is not None:
            high_t, low_t = CollapsController.get_dynamic_thresholds(
                self.lambda_, self.k, HIGH_THRESHOLD, LOW_THRESHOLD
            )
        else:
            high_t, low_t = HIGH_THRESHOLD, LOW_THRESHOLD
        
        binary = np.full((self.k, self.L), 0.5)
        binary[self.M > high_t] = 1.0
        binary[self.M < low_t] = 0.0
        return binary
    
    def get_lambda_c(self) -> Optional[float]:
        if self._compute_lambda_c is None:
            return None
        return self._compute_lambda_c(self.k, n_high=1)
    
    def get_n_high_groups(self) -> int:
        state = self.read()
        return int(np.sum(state.groups > 0.5))
    
    def set_n_high(self, n_high: int, n_steps: int = 100) -> Dict:
        if not (0 <= n_high <= self.k):
            raise ValueError(f"n_high must be between 0 and {self.k}")
        
        lc = self.get_lambda_c()
        if lc is None or not self.use_physics_control:
            lc = 0.1
        
        lo, hi = 0.0, lc * 0.99
        best_lambda = 0.0
        best_nh = 0
        best_diff = self.k
        
        for _ in range(8):
            test_lambdas = np.linspace(lo, hi, 5)
            for lam_test in test_lambdas:
                old_lambda = self.lambda_
                self.lambda_ = lam_test
                self.step(dt=DEFAULT_DT, n_steps=n_steps // 4)
                nh_achieved = self.get_n_high_groups()
                diff = abs(nh_achieved - n_high)
                
                if diff < best_diff:
                    best_diff = diff
                    best_lambda = lam_test
                    best_nh = nh_achieved
                
                self.lambda_ = old_lambda
            
            if best_diff <= 0:
                break
            
            if best_nh < n_high:
                lo = best_lambda
            else:
                hi = best_lambda
        
        self.lambda_ = best_lambda
        self.step(dt=DEFAULT_DT, n_steps=n_steps)
        final_nh = self.get_n_high_groups()
        
        return {
            "lambda_used": best_lambda,
            "n_high_achieved": final_nh,
            "n_high_target": n_high,
            "converged": abs(final_nh - n_high) <= 1,
            "lambda_c": lc
        }
    
    def save(self, filepath: Optional[str] = None) -> str:
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
    def load(cls, filepath: str) -> "AgentMemoryEnhanced":
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
            f"AgentMemoryEnhanced '{self.name}'\n"
            f"  k={self.k}, L={self.L} | λ={self.lambda_:.3f} (λ_c≈{lc_str}), μ={self.mu:.3f}\n"
            f"  Mode: {self.current_mode} | Active: {self.read().n_active}/{self.k*self.L}\n"
            f"  LLM detector: {self.use_llm_detector} | LLM mapper: {self.use_llm_mapper}"
        )


def demo():
    """Demo with keyword-based detection."""
    print("=" * 60)
    print("Agent Memory Enhanced Demo (Keyword Mode)")
    print("=" * 60)
    
    mem = AgentMemoryEnhanced(
        k=3, L=2, name="assistant",
        use_llm_detector=False,
        use_llm_mapper=False
    )
    mem.group_labels = ["professional", "friendly", "technical"]
    mem.layer_labels = ["preferences", "context"]
    
    print("\n1. Initial state:")
    print(mem.status())
    
    print("\n2. Writing memories:")
    mem.write("Short responses preferred", group=0, layer=0)
    mem.write("Like analogies", group=1, layer=0)
    mem.write("Tech industry background", group=2, layer=1)
    
    print("\n3. Processing prompts:")
    prompts = [
        "Tell me about the weather",
        "I want to learn coding but I'm also tired",
        "Both options seem good, actually wait, focus on the first one",
    ]
    
    for prompt in prompts:
        result = mem.process_prompt(prompt)
        print(f"\nPrompt: '{prompt}'")
        print(f"  Conflict: {result['conflict_level']:.0%}")
        print(f"  λ: {result['old_lambda']:.3f} → {result['new_lambda']:.3f}")
    
    print("\n4. Active pattern:")
    print(mem.get_active_pattern())
    
    print("\n5. LLM context:")
    print(mem.get_context_for_llm())
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    demo()