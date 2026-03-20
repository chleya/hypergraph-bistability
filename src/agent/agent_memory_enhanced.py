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
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini",
                 base_url: Optional[str] = None):
        self.model = model
        self._client = None
        self._has_openai = False
        
        if api_key or os.environ.get("OPENAI_API_KEY"):
            try:
                from openai import OpenAI
                actual_key = api_key or os.environ.get("OPENAI_API_KEY")
                actual_base = (base_url or os.environ.get("OPENAI_API_BASE")
                              or "https://api.openai.com/v1")
                self._client = OpenAI(api_key=actual_key, base_url=actual_base)
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

Respond with JSON object containing "level" (number 0.0-1.0) and "reasoning" (string)."""
        
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=100,
                response_format={"type": "json_object"}
            )
            
            import json
            result_data = json.loads(response.choices[0].message.content)
            
            level = float(result_data.get("level", 0.0))
            reasoning = str(result_data.get("reasoning", "No reasoning"))
            return level, reasoning
                
        except Exception as e:
            print(f"LLM detection failed: {e}, falling back to keyword method")
            return self._detect_fallback(prompt)
    
    def _detect_fallback(self, prompt: str) -> Tuple[float, str]:
        """Simple keyword-based conflict detection."""
        prompt_lower = prompt.lower()
        
        conflict_keywords = [
            "but", "however", "although", "despite", "instead",
            "actually", "wait", "on the other hand",
            "or", "either", "neither"
        ]
        
        conflict_count = sum(1 for kw in conflict_keywords if kw in prompt_lower)
        word_count = len(prompt.split())
        
        level = min(1.0, conflict_count * 0.15 + word_count * 0.005)
        
        return level, f"Simple: {conflict_count} conflict keywords"


class SemanticMemoryMapper:
    """
    Maps memory slots to semantic content using LLM.
    
    Enables flexible memory writing without hardcoding slot indices.
    """
    
    def __init__(self, k: int, L: int, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.k = k
        self.L = L
        self.model = model
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

Respond with JSON object containing "group" (integer 0-{self.k-1}), "layer" (integer 0-{self.L-1}), and "reasoning" (string)."""
        
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"{action.upper()}: {content}"}
                ],
                temperature=0.0,
                max_tokens=100,
                response_format={"type": "json_object"}
            )
            
            import json
            result_data = json.loads(response.choices[0].message.content)
            
            g = int(result_data.get("group", 0))
            l = int(result_data.get("layer", 0))
            return min(g, self.k-1), min(l, self.L-1)
            
        except Exception as e:
            print(f"LLM slot finding failed: {e}")
            return self._find_slot_keyword(content, action)
    
    def _find_slot_keyword(self, content: str, action: str) -> Tuple[int, int]:
        """Simple keyword-based slot finding."""
        content_lower = content.lower()
        
        if any(w in content_lower for w in ["work", "job", "career", "project"]):
            group = 0
        elif any(w in content_lower for w in ["personal", "family", "friend", "home"]):
            group = 1
        elif any(w in content_lower for w in ["code", "technical", "bug", "program"]):
            group = 2
        else:
            group = 0
        
        layer = 0 if any(w in content_lower for w in ["want", "prefer", "like", "need"]) else 1
        
        return min(group, self.k-1), min(layer, self.L-1)


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
        use_physics_control: bool = True,
        gamma: float = 0.0
    ):
        self.k = k
        self.L = L
        self.alpha = alpha
        self.a = a_bistable
        self.lambda_ = lambda_
        self.mu = mu
        self.gamma = gamma
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
        
        self._lambda_from_mode = False
    
    def bistable_dynamics(self, m: np.ndarray) -> np.ndarray:
        return m * (1 - m) * (2 * m - 1)
    
    def _compute_dMdt(self, M_flat: np.ndarray, t: float) -> np.ndarray:
        """ODE right-hand side for scipy odeint (vectorized).
        
        Dynamics: dm/dt = bistable + lambda*(m_col_mean - m) + mu*(m_row_other - m) - gamma*m
        
        - Bistable: m(1-m)(2m-1) attracts to 0 or 1
        - Lambda: global coupling (column-wise mean)
        - Mu: local coupling (row-wise, other layers)
        - Gamma: decay/ forgetting term (pulls toward 0)
        """
        M = M_flat.reshape((self.k, self.L))
        
        # 1. Bistable term (vectorized)
        bistable = M * (1 - M) * (2 * M - 1)
        
        # 2. Lambda global coupling (per column mean)
        layer_means = np.mean(M, axis=0)
        gc = self.lambda_ * (layer_means - M)
        
        # 3. Mu local coupling (per row, other layers mean)
        if self.L > 1:
            row_sums = np.sum(M, axis=1, keepdims=True)
            other_means = (row_sums - M) / (self.L - 1)
            lc = self.mu * (other_means - M)
        else:
            lc = np.zeros_like(M)
        
        # 4. Gamma decay/ forgetting term
        decay = self.gamma * M
        
        dM = bistable + gc + lc - decay
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
                M = self.M
                
                # 1. Bistable term
                bistable = M * (1 - M) * (2 * M - 1)
                
                # 2. Lambda global coupling (per column mean)
                layer_means = np.mean(M, axis=0)
                gc = self.lambda_ * (layer_means - M)
                
                # 3. Mu local coupling (per row, other layers mean)
                if self.L > 1:
                    row_sums = np.sum(M, axis=1, keepdims=True)
                    other_means = (row_sums - M) / (self.L - 1)
                    lc = self.mu * (other_means - M)
                else:
                    lc = np.zeros_like(M)
                
                # 4. Gamma decay
                decay = self.gamma * M
                
                dM = bistable + gc + lc - decay
                self.M = np.clip(M + dM * dt, 0, 1)
    
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
        
        If lambda was just set by switch_mode (_lambda_from_mode=True),
        preserve it for one prompt before auto-adjusting again.
        """
        old_lambda = self.lambda_
        
        if self._lambda_from_mode:
            regime = _regime_name(self.lambda_ / (self.get_lambda_c() or 0.1))
            result = {
                "prompt": prompt,
                "conflict_level": self.lambda_ / (self.get_lambda_c() or 0.1),
                "reasoning": f"Mode-preserved λ={self.lambda_:.4f}",
                "old_lambda": old_lambda,
                "new_lambda": self.lambda_,
                "regime": regime
            }
        elif self.use_physics_control and self._compute_lambda_c is not None:
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
        
        self._lambda_from_mode = False
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
        else:
            r, mu_val = 0.5, 0.0
        
        if lc is not None and self.use_physics_control:
            lam = r * lc
        else:
            lam = r * 0.1
        self.lambda_ = lam
        self.mu = mu_val
        
        self.current_mode = mode
        self.mode_history.append(mode)
        self._lambda_from_mode = True
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
    
    def get_context_for_llm(self, max_per_group: int = 3) -> str:
        state = self.read()
        ctx_lines = [f"[Memory State: {self.current_mode}]"]
        
        active_groups = []
        for i in range(self.k):
            if state.groups[i] > 0.5:
                key_prefix = f"({i},"
                memories = [v for k, v in self.content_map.items() if k.startswith(key_prefix)]
                if memories:
                    truncated = memories[:max_per_group]
                    active_groups.append(f"{self.group_labels[i]}: {', '.join(truncated)}")
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
            "gamma": float(self.gamma),
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
            gamma=data.get("gamma", 0.0),
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
            f"  k={self.k}, L={self.L} | λ={self.lambda_:.3f} (λ_c≈{lc_str}), μ={self.mu:.3f}, γ={self.gamma:.4f}\n"
            f"  Mode: {self.current_mode} | Active: {self.read().n_active}/{self.k*self.L}\n"
            f"  LLM detector: {self.use_llm_detector} | LLM mapper: {self.use_llm_mapper}"
        )
    
    def set_decay(self, gamma: float) -> None:
        """
        Set the decay/forgetting rate γ.
        
        Parameters
        ----------
        gamma : float
            Decay rate. 0 = no decay (permanent memory).
            Typical values: 0.001-0.01 for slow forgetting.
        """
        self.gamma = max(0.0, float(gamma))
    
    def boost_slot(self, group: int, layer: int, boost: float = 0.1) -> None:
        """
        Boost a specific memory slot's activation.
        
        Parameters
        ----------
        group : int
            Group index
        layer : int
            Layer index
        boost : float
            Amount to add to M[group, layer]
        """
        if 0 <= group < self.k and 0 <= layer < self.L:
            self.M[group, layer] = min(1.0, self.M[group, layer] + boost)
            self.step(dt=0.1, n_steps=5)
    
    def decay_all(self, amount: float = 0.01) -> None:
        """
        Apply uniform decay to all memory slots.
        
        Parameters
        ----------
        amount : float
            Amount to subtract from all M values
        """
        self.M = np.clip(self.M - amount, 0.0, 1.0)
        self.step(dt=0.1, n_steps=5)


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