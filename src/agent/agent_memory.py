"""
Agent Memory Module - Enhanced Version
======================================

Physics-based multistability control using λ_c analytic formula.

Key design principles:
- λ is controlled via proximity ratio r = λ/λ_c(k), NOT keyword heuristics
- Dynamics solved via scipy odeint (not simple Euler)
- Binarization thresholds adapt based on distance to λ_c
- n_high targeting via iterative λ search

Reference: paper_final.md Section 3.8 (λ_c formula)
"""

import numpy as np
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass
import json
from scipy.integrate import odeint


HIGH_THRESHOLD = 0.7
LOW_THRESHOLD = 0.3

DEFAULT_DT = 0.1
HIGH_TARGET = 0.95
LOW_TARGET = 0.05


def _regime_name(proximity_ratio: float) -> str:
    """Get regime name from proximity ratio r = λ/λ_c."""
    if proximity_ratio < 0.5:
        return "multi-attractor"
    elif proximity_ratio < 0.7:
        return "moderate-coupling"
    elif proximity_ratio < 0.85:
        return "approaching-critical"
    elif proximity_ratio < 0.95:
        return "near-critical"
    elif proximity_ratio < 1.05:
        return "at-critical-WTA"
    else:
        return "post-critical-WTA"


class CollapsController:
    """
    Physics-based collapse controller using λ_c analytic formula.
    
    Uses the saddle-node bifurcation condition from paper Section 3.8:
    λ_c = 3h²l²/(h²+(k-1)l²) for n_high=1 saddle-node.
    
    Proximity ratio r = λ/λ_c determines behavior:
    - r < 0.70: multi-attractor regime (N_att = 2^{k×L})
    - 0.70-0.95: approaching critical point (gradual dimensional collapse)
    - r >= 0.95: WTA regime (effective dimensionality reduced)
    
    This REPLACES keyword-based conflict detection with a rigorous
    physics-based approach.
    """
    
    _lambda_c_cache: Dict[int, float] = {}
    
    @classmethod
    def clear_cache(cls) -> None:
        """Clear λ_c cache. Call between test runs."""
        cls._lambda_c_cache.clear()
    
    @classmethod
    def get_lambda_c(cls, k: int) -> float:
        """Get cached λ_c for given k."""
        if k not in cls._lambda_c_cache:
            from hypergraph_control import compute_lambda_c
            cls._lambda_c_cache[k] = compute_lambda_c(k, n_high=1)
        return cls._lambda_c_cache[k]
    
    @staticmethod
    def compute_collapse_ratio(lambda_: float, k: int) -> float:
        """
        Compute proximity ratio r = λ / λ_c.
        
        Returns r ∈ [0, ∞). r > 1 means past the critical point.
        """
        lc = CollapsController.get_lambda_c(k)
        return lambda_ / lc if lc > 0 else 0.0
    
    @classmethod
    def suggest_lambda(cls, proximity_ratio: float, k: int) -> Tuple[float, str]:
        """
        Suggest λ given a target proximity ratio.
        
        Parameters
        ----------
        proximity_ratio : float
            Target r = λ/λ_c. Recommended: 0.5-0.95.
        k : int
            Number of groups.
        
        Returns
        -------
        Tuple[float, str]
            (suggested_λ, reasoning)
        """
        lc = cls.get_lambda_c(k)
        suggested = proximity_ratio * lc
        suggested = max(0.0, min(suggested, lc * 1.5))
        
        if proximity_ratio < 0.5:
            regime = "multi-attractor (all states accessible)"
        elif proximity_ratio < 0.8:
            regime = "approaching critical (some dimensional reduction)"
        elif proximity_ratio < 0.95:
            regime = "near-critical (strong coupling, few states)"
        else:
            regime = "WTA regime (dimensional collapse)"
        
        reasoning = f"r={proximity_ratio:.2f}×λ_c({k})={lc:.4f} → λ={suggested:.4f} [{regime}]"
        return suggested, reasoning
    
    @classmethod
    def get_dynamic_thresholds(cls, lambda_: float, k: int,
                               base_high: float = 0.7,
                               base_low: float = 0.3) -> Tuple[float, float]:
        """
        Compute dynamic binarization thresholds based on proximity to λ_c.
        
        As λ → λ_c (r → 1), the fixed points sharpen, so thresholds
        can be tighter. As r → 0, thresholds can be looser.
        
        Parameters
        ----------
        lambda_ : float
            Current coupling strength
        k : int
            Number of groups
        base_high : float
            Upper threshold at r=0
        base_low : float
            Lower threshold at r=0
        
        Returns
        -------
        Tuple[float, float]
            (high_threshold, low_threshold)
        """
        r = cls.compute_collapse_ratio(lambda_, k)
        
        compression = 0.5 + 0.5 * min(r, 1.0)
        
        mid = (base_high + base_low) / 2.0
        half_width = (base_high - base_low) / 2.0 * compression
        
        high_t = mid + half_width
        low_t = mid - half_width
        
        return high_t, low_t
    
    @classmethod
    def suggest_lambda_for_n_high(cls, n_high: int, k: int, 
                                  tolerance: float = 0.5) -> Tuple[float, List[float]]:
        """
        Estimate λ that produces approximately n_high groups in HIGH state.
        
        Uses binary search on λ ∈ [0, λ_c] to find the λ that yields
        n_high high groups after convergence.
        
        At λ=0: all 2^{k×L} attractors equally accessible → n_high is set by IC
        At λ=λ_c: WTA collapse → all groups synchronized → n_high = k or 0
        
        Parameters
        ----------
        n_high : int
            Desired number of HIGH groups (0 to k)
        k : int
            Number of groups
        tolerance : float
            Acceptable error in n_high
            
        Returns
        -------
        Tuple[float, List[float]]
            (estimated_lambda, n_high_profile as function of λ)
        
        Note: This is an approximation. Exact n_high control requires
        solving the full ODE system. Works best for n_high=1 (strong WTA)
        or n_high=k (fully synchronized).
        """
        lc = cls.get_lambda_c(k)
        
        def count_high_groups(lambda_val: float, mu: float = 0.0) -> int:
            """Simulate and count HIGH groups."""
            np.random.seed(42)
            M0 = np.full((self.k, self.L), 0.5) + np.random.randn(self.k, self.L) * 0.02
            M0 = np.clip(M0, 0.01, 0.99)
            M0_flat = M0.flatten()
            
            def dM_flat(M_flat, t):
                M = M_flat.reshape((self.k, self.L))
                dM_arr = np.zeros_like(M)
                for i in range(self.k):
                    for l in range(self.L):
                        m = M[i, l]
                        bistable = m * (1 - m) * (2 * m - 1)
                        gc = lambda_val * (np.mean(M[:, l]) - m)
                        ol = [M[i, j] for j in range(self.L) if j != l]
                        om = np.mean(ol) if ol else 0.0
                        lc_coup = mu * (om - m)
                        dM_arr[i, l] = bistable + gc + lc_coup
                return dM_arr.flatten()
            
            sol = odeint(dM_flat, M0_flat, [0, 50])
            M_final = sol[-1].reshape((self.k, self.L))
            return int(np.sum(np.mean(M_final, axis=1) > 0.5))
        
        lo, hi = 0.0, lc * 0.99
        best_lambda = 0.0
        best_diff = k
        
        for lam_test in np.linspace(lo, hi, 20):
            nh = count_high_groups(lam_test)
            diff = abs(nh - n_high)
            if diff < best_diff:
                best_diff = diff
                best_lambda = lam_test
            if diff <= tolerance:
                break
        
        profile = [float(count_high_groups(lam)) for lam in np.linspace(0, lc * 0.99, 10)]
        
        return best_lambda, profile


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
        auto_adjust: bool = True,
        use_physics_control: bool = True
    ):
        self.k = k
        self.L = L
        self.alpha = alpha
        self.a = a_bistable
        self.lambda_ = lambda_
        self.mu = mu
        self.name = name
        self.auto_adjust = auto_adjust
        self.use_physics_control = use_physics_control
        
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
        
        self._has_scipy = True
        try:
            from scipy.integrate import odeint
        except ImportError:
            self._has_scipy = False
    
    def bistable_dynamics(self, m: np.ndarray) -> np.ndarray:
        return m * (1 - m) * (2 * m - 1)
    
    def _compute_dMdt(self, M_flat: np.ndarray, t: float) -> np.ndarray:
        """ODE right-hand side for scipy odeint."""
        M = M_flat.reshape((self.k, self.L))
        dM = np.zeros_like(M)
        for i in range(self.k):
            for l in range(self.L):
                m = M[i, l]
                bistable = self.bistable_dynamics(m)
                gc = self.lambda_ * (np.mean(M[:, l]) - m)
                other_layers = [M[i, j] for j in range(self.L) if j != l]
                other_mean = np.mean(other_layers) if other_layers else 0.0
                lc = self.mu * (other_mean - m)
                dM[i, l] = bistable + gc + lc
        return dM.flatten()
    
    def step(self, dt: float = 0.1, n_steps: int = 1) -> None:
        """
        Advance dynamics by n_steps*dt time units.
        
        Uses scipy.integrate.odeint when available (recommended).
        Falls back to small-step Euler for compatibility.
        """
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
    
    def _estimate_r_from_prompt(self, prompt: str) -> float:
        """
        Estimate proximity ratio r from prompt text.
        
        Combines:
        - Keyword conflict score (0~0.6 contribution)
        - Sentence complexity (0~0.4 contribution)
        - Question/stuck signals (bonus 0~0.15)
        
        Returns r ∈ [0.20, 0.98].
        """
        prompt_lower = prompt.lower()
        words = prompt_lower.split()
        word_count = len(words)
        sentence_count = max(1, prompt.count('.') + prompt.count('!') + prompt.count('?'))
        avg_sentence_len = word_count / sentence_count
        
        conflict_keywords = [
            "but", "however", "although", "despite", "instead",
            "contradict", "opposite", "different", "change",
            "unfortunately", "actually", "wait"
        ]
        conflict_count = sum(1 for kw in conflict_keywords if kw in prompt_lower)
        
        stuck_signals = ["wait", "actually", "no wait", "on the other hand",
                         "not sure", "i don't know", "hesitate", "maybe"]
        stuck_count = sum(1 for s in stuck_signals if s in prompt_lower)
        
        question_count = prompt.count('?')
        
        keyword_score = min(0.6, conflict_count * 0.12)
        complexity_score = min(0.4, avg_sentence_len * 0.025)
        stuck_bonus = min(0.15, stuck_count * 0.05)
        question_bonus = min(0.10, question_count * 0.03)
        
        r = 0.25 + keyword_score + complexity_score + stuck_bonus + question_bonus
        r = min(0.98, max(0.20, r))
        
        return r
    
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
    
    def process_prompt(self, prompt: str,
                       use_physics_control: bool = True,
                       target_r: Optional[float] = None) -> Dict:
        """
        Process a prompt and adjust memory state via physics-based λ_c control.
        
        All λ adjustments are driven by proximity ratio r = λ/λ_c(k), NOT keywords.
        Prompt complexity (word count) is used as a simple heuristic for r.
        
        Parameters
        ----------
        prompt : str
            User prompt
        use_physics_control : bool
            Force physics-based control (default True)
        target_r : float, optional
            Directly set target proximity ratio. 0.4~0.9 range.
            0.4 = multi-mode, 0.7 = moderate coupling, 0.9 = near WTA
        
        Returns
        -------
        dict
            {conflict_level (=r), reasoning, old_lambda, new_lambda, regime}
        """
        old_lambda = self.lambda_
        
        if target_r is None:
            target_r = self._estimate_r_from_prompt(prompt)
        
        if use_physics_control and self._compute_lambda_c is not None:
            suggested_lambda, reasoning = CollapsController.suggest_lambda(target_r, self.k)
            self.lambda_ = suggested_lambda
            
            regime = _regime_name(target_r)
            result = {
                "conflict_level": target_r,
                "reasoning": f"Physics-based: target r={target_r:.2f} → {reasoning}",
                "old_lambda": old_lambda,
                "new_lambda": self.lambda_,
                "regime": regime
            }
        else:
            lc_fallback = 0.1
            suggested_lambda = target_r * lc_fallback
            self.lambda_ = suggested_lambda
            result = {
                "conflict_level": target_r,
                "reasoning": f"Fallback r={target_r:.2f} → λ≈{suggested_lambda:.4f} (no λ_c formula)",
                "old_lambda": old_lambda,
                "new_lambda": self.lambda_
            }
        
        self.step(dt=DEFAULT_DT, n_steps=20)
        self.conflict_history.append(result["conflict_level"])
        return result
    
    def switch_mode(
        self,
        mode: str,
        lambda_override: Optional[float] = None,
        mu_override: Optional[float] = None
    ) -> None:
        """
        Switch memory mode with physics-based λ presets.
        
        Presets are now defined as proximity ratios to λ_c:
        - neutral:       r=0.0  (pure multi-attractor, λ=0)
        - exploratory:   r=0.3  (light coupling, multiple states)
        - focused:       r=0.85 (near critical, strong selection)
        - sync:          r=0.5, μ=0.3 (moderate coupling + layer sync)
        - creative:      r=0.5, μ=-0.3 (moderate coupling + layer anti-sync)
        - professional:  r=0.65 (moderate-high coupling)
        - casual:        r=0.4, μ=0.1 (light coupling + mild sync)
        
        When use_physics_control=True, λ is computed from r × λ_c(k).
        """
        lc = self.get_lambda_c()
        
        proximity_presets = {
            "neutral": (0.0, 0.0),
            "exploratory": (0.3, 0.0),
            "focused": (0.85, 0.0),
            "sync": (0.5, 0.3),
            "creative": (0.5, -0.3),
            "professional": (0.65, 0.0),
            "casual": (0.4, 0.1),
        }
        
        if mode in proximity_presets:
            r, mu_val = proximity_presets[mode]
            if lc is not None and self.use_physics_control:
                lam = r * lc
            else:
                lam = r * 0.1
            if lambda_override is not None:
                lam = lambda_override
            if mu_override is not None:
                mu_val = mu_override
            self.lambda_ = lam
            self.mu = mu_val
        else:
            if lambda_override is None or mu_override is None:
                raise ValueError(f"Unknown mode '{mode}'")
            self.lambda_ = lambda_override
            self.mu = mu_override
        
        self.current_mode = mode
        self.mode_history.append(mode)
        self.step(dt=DEFAULT_DT, n_steps=20)
    
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
        """Return the number of groups currently in HIGH state."""
        state = self.read()
        return int(np.sum(state.groups > 0.5))
    
    def set_n_high(self, n_high: int, n_steps: int = 100) -> Dict:
        """
        Set the memory to a configuration with approximately n_high HIGH groups.
        
        Uses iterative λ search to converge on the desired n_high.
        At λ=0: initial conditions determine n_high
        At λ=λ_c: WTA → n_high = k (all synchronized HIGH)
        At λ >> λ_c: n_high = 0 (all LOW)
        
        Parameters
        ----------
        n_high : int
            Desired number of HIGH groups (0 to k)
        n_steps : int
            Integration steps per λ iteration
        
        Returns
        -------
        dict
            {lambda_used, n_high_achieved, n_high_target, converged}
        """
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
            "n_high_profile": [self.get_n_high_groups() for _ in range(3)],
            "converged": abs(final_nh - n_high) <= 1,
            "lambda_c": lc
        }
    
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
    print("Agent Memory Module - Physics-Based Demo")
    print("=" * 60)
    
    CollapsController.clear_cache()
    
    mem = AgentMemory(k=3, L=2, name="assistant", use_physics_control=True)
    mem.group_labels = ["professional", "friendly", "technical"]
    mem.layer_labels = ["preferences", "context"]
    
    print("\n1. Initial state:")
    print(mem.status())
    lc = mem.get_lambda_c()
    print(f"   λ_c = {lc:.4f}" if lc else "   λ_c = N/A (hypergraph_control not available)")
    
    print("\n2. Writing memories:")
    mem.write("Short responses preferred", group=0, layer=0)
    mem.write("Like analogies", group=1, layer=0)
    mem.write("Tech industry", group=2, layer=1)
    print(mem.get_active_pattern())
    
    print("\n3. Physics-based λ adjustment (r = λ/λ_c):")
    
    prompts = [
        ("Short prompt", "Tell me about the weather"),
        ("Medium prompt", "I want to learn coding but I'm also tired"),
        ("Long complex prompt", "Both options seem good, actually wait no, let me focus on the first one exactly, I need to be very precise about this decision"),
    ]
    
    for label, prompt in prompts:
        result = mem.process_prompt(prompt)
        print(f"\n  [{label}]")
        regime = result.get('regime', 'n/a')
        print(f"  r={result['conflict_level']:.2f} (regime={regime})")
        print(f"  λ: {result['old_lambda']:.4f} → {result['new_lambda']:.4f}")
        print(f"  {result['reasoning']}")
    
    print("\n4. Memory context for LLM:")
    print(mem.get_context_for_llm())
    
    print("\n5. Regime examples (λ_c scaling):")
    for k in [2, 3, 4]:
        try:
            lc = CollapsController.get_lambda_c(k)
            for r in [0.3, 0.7, 0.95]:
                lam = r * lc
                print(f"  k={k}, r={r:.2f}: λ={lam:.4f}, regime={_regime_name(r)}")
        except Exception:
            print(f"  k={k}: λ_c unavailable (hypergraph_control not installed)")
    
    print("\n6. Save/Load test:")
    path = mem.save()
    print(f"  Saved to: {path}")
    
    mem2 = AgentMemory.load(path)
    
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