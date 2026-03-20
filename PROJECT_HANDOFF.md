# Hypergraph Multistability Framework - Project Handoff Document

## For: Next LLM Model

Date: 2026-03-20
Status: Ready for continuation

---

## 1. Project Overview

This is a research project on **hypergraph competitive dynamics** and its application to AI Agent memory systems. The core question is: **How can we build a memory system for AI Agents that supports multiple persistent contexts, controllable switching, and interpretable state collapse?**

The key insight is that the brain's attention system (biased competition, WTA) can be mathematically modeled using a **group-layer tensor product structure** with bistable units and competitive coupling.

---

## 2. Core Mathematical Framework

### 2.1 Model Structure

**State**: A `k × L` matrix `M[i,l]` where:
- `k` = number of groups (e.g., personas/roles)
- `L` = number of layers (e.g., memory dimensions)
- `M[i,l] ∈ [0,1]` = activation of group `i`'s layer `l`

**Bistable Dynamics**: Each unit is an explicit bistable switch:

```
f(m) = m(1-m)(2m-1)
```

Stable fixed points at `m=0` (LOW) and `m=1` (HIGH). Unstable fixed point at `m=0.5`.

### 2.2 Full Dynamics

```
dM[i,l]/dt = f(M[i,l]) 
            + λ * Σⱼ (mean(M[j,:]) - M[i,l])   [inter-group competition]
            + μ * Σₗ (M[i,l2] - M[i,l])          [inter-layer cooperation]
```

Where:
- `λ` (lambda) = inter-group coupling strength
- `μ` (mu) = inter-layer coupling strength

### 2.3 Key Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| α | 1.0 | Bistable amplitude |
| a | 0.5 | Bistable midpoint |
| k | 3 | Number of groups |
| L | 2 | Number of layers |
| λ | variable | Inter-group coupling |
| μ | variable | Inter-layer coupling |

---

## 3. Core Theoretical Results

### 3.1 Attractor Count

At `λ→0` (no coupling): **N_att = 2^{k×L}**

This is EXACT, verified for k=2,3,4 and L=1,2.

### 3.2 Critical Coupling λ_c

The **Winner-Take-All (WTA) collapse** occurs at critical coupling `λ_c`.

**Semi-analytic formula** (for n_high=1):

```
λ_c = 3h²l² / [h² + (k-1)l²]

where:
  h = H* - 0.5  (deviation of HIGH fixed point from midpoint)
  l = L* - 0.5  (deviation of LOW fixed point from midpoint)
```

**Verified values** (standard bistable form `f(m)=m(1-m)(2m-1)`):

| k | λ_c (n_high=1) | Power-law ≈ 0.70/k² |
|---|-----------------|---------------------|
| 2 | 0.1667 | 0.175 (5% error) |
| 3 | 0.0675 | 0.078 (15% error) |
| 4 | 0.0437 | 0.044 (0.1% error) |

**Power-law fit**: `λ_c(k) ≈ 0.70 / k²`

### 3.3 (λ, μ) Phase Diagram (k=3, L=2)

| Regime | μ | Behavior |
|--------|---|---------|
| Anti-sync | μ < -0.1 | WTA suppressed, N_att ≈ 10 |
| Standard | μ ≈ 0 | Normal bistable cascade |
| Sync | μ > 0.1 | Dimensional collapse to effective k |

### 3.4 L→∞ Limit

**Critical dimension L_c = 2**:
- L=1: Discrete phase transition (sharp 8→2 collapse)
- L≥2: Gradual transition (no sharp bifurcation)

### 3.5 Noise Escape

Critical noise for escape: **σ_c ≈ 0.2-0.3**

Below this, noise irrelevant; above, transitions common.

---

## 4. Control Results

### 4.1 What Works

| Strategy | Success |
|----------|---------|
| Global Pull | 100% |
| Global Boost | 100% |

### 4.2 What Doesn't Work

| Strategy | Success |
|----------|---------|
| Local Boost (standard dynamics) | **0%** |
| PPO Learned (standard dynamics) | **0%** |

**Key insight**: Local control cannot overcome global coupling under standard bistable dynamics. The 62% reported in prior work came from **non-standard dynamics with asymmetric capacities** (Kc=[0.32, 0.40, 0.48]).

---

## 5. Agent Memory Module

### 5.1 Purpose

Build a **discrete, real-time, controllable memory module** for AI Agents using the hypergraph multistability framework.

### 5.2 Architecture

```
AgentMemory (k=3, L=2)
├── M[i,l] ∈ [0,1]     State matrix
├── group_labels        ["professional", "friendly", "technical"]
├── layer_labels        ["preferences", "context"]
└── modes              ["neutral", "exploratory", "focused", "creative"]
```

### 5.3 Core Interface

```python
from agent.agent_memory import AgentMemory

# Initialize
mem = AgentMemory(k=3, L=2, auto_adjust=True)
mem.group_labels = ["professional", "personal", "technical"]
mem.layer_labels = ["preferences", "context"]

# Write memory
mem.write("User prefers concise answers", group=0, layer=0)

# Auto-adjust λ based on prompt conflict
result = mem.process_prompt("I want to learn coding but I'm also tired")
# → Detects conflict level, adjusts λ automatically

# Get context for LLM
context = mem.get_context_for_llm()
# → "[Memory State: neutral]\nActive contexts: professional: User prefers..."

# Switch mode
mem.switch_mode("focused")

# Save/Load
mem.save("my_memory.json")
mem2 = AgentMemory.load("my_memory.json")
```

### 5.4 Conflict Detection

```python
conflict_level = ConflictDetector.detect_conflict_level(prompt)
# Uses keywords: "but", "however", "although" → high conflict
# Output: 0.0 (calm) to 1.0 (highly conflicting)
```

### 5.5 λ Auto-Adjustment

| Conflict | λ | Mode |
|----------|---|------|
| < 0.1 | 0.01 | exploratory |
| 0.1-0.3 | 0.03-0.06 | neutral |
| 0.3-0.5 | 0.06-0.10 | focused |
| > 0.5 | 0.10-0.15 | near-WTA |

---

## 6. Files Reference

### 6.1 Core Source Files

| File | Purpose |
|------|---------|
| `src/agent/agent_memory.py` | Main Agent memory module (basic) |
| `src/agent/agent_memory_enhanced.py` | Enhanced with LLM-based conflict detection |
| `src/agent/llm_integration.py` | LLM integration examples |
| `src/agent/llm_integration_test.py` | Test suite for LLM integration |

### 6.2 Verification Scripts

| File | Purpose |
|------|---------|
| `src/verify_layered_cascade.py` | Path B verification |
| `src/PPO_standard_dynamics.py` | Local control verification |
| `src/L_infinity_study.py` | L→∞ limit study |
| `src/Q1_lambda_c_scaling.py` | λ_c(L) scaling study |
| `src/Q2_lambda_mu_phase.py` | (λ,μ) phase diagram study |
| `src/Q3_noise_escape.py` | Noise escape rate study |
| `src/Q4_asymmetric_k.py` | Asymmetric k study |

### 6.3 Results

| Directory | Contents |
|-----------|----------|
| `results/Q1_lambda_c_scaling/` | L→∞ scaling data |
| `results/Q2_lambda_mu_phase/` | (λ,μ) phase diagram |
| `results/Q3_noise_escape/` | Noise escape rates |
| `results/Q4_asymmetric_k/` | Asymmetric k data |
| `results/L_infinity/` | L→∞ limit study |
| `results/COMPLETE_RESEARCH_LOG.md` | Full research log |

### 6.4 Paper

| File | Status |
|------|--------|
| `paper/paper_final.md` | Complete paper with Sections 3.6-3.8 (L→∞, asymmetry, noise, topology, control, λ_c formula) |

---

## 7. Important Notes

### 7.1 Driving Function Note

Two equivalent forms exist in the codebase:

1. **Standard form** (used in papers): `f(m) = m(1-m)(2m-1)`
2. **Parameteric form** (used in code): `f(m) = α*m*(1-m)*(m-a)`

They differ by a constant factor of 1/2, absorbed into the time scale. **Use `m(1-m)(2m-1)` for all future work** as it's the standard form in neuroscience/dynamical systems literature.

### 7.2 Critical k Values

The **old** λ_c values (wrong form) were:
- k=2: 0.12469
- k=3: 0.08305
- k=4: 0.02185

The **correct** λ_c values (standard form `m(1-m)(2m-1)`) are:
- k=2: **0.16667**
- k=3: **0.06753**
- k=4: **0.04369**

### 7.3 Local Control Finding

**Do NOT use** the 62% success rate from prior work as evidence that local control works. Our verification shows:

- Local control with fixed parameters: **0% success**
- Local control with PPO (standard dynamics): **0% success**
- Local control with PPO (non-standard dynamics): 35-45% success

The 62% is an artifact of asymmetric capacities and non-standard cubic dynamics.

---

## 8. Suggested Next Steps

### Priority 1: Paper Completion
- Write the full paper using existing results
- Add proper citations for Hopfield/high-order Hopfield
- Include the semi-analytic λ_c formula (Section 3.8)

### Priority 2: Enhanced Agent Memory
- Replace keyword-based conflict detection with LLM-based classification
- Implement semantic memory mapping (not just slot-based)
- Add memory decay/forgetting mechanisms

### Priority 3: Experimental Validation
- Run actual LLM integration tests with OpenAI API
- Compare with classic Hopfield on memory tasks
- Test on real Agent scenarios (multi-persona, context switching)

---

## 9. Key Insights for Future Work

### 9.1 Why This Framework is Different from Hopfield

1. **Explicit bistability** (designed) vs Hebbian emergence
2. **Parameter-controlled collapse** vs temperature-driven phase transition
3. **Saddle-node bifurcation** vs spin-glass transitions
4. **Dimensionality control** via μ parameter

### 9.2 Biological Inspiration

The framework maps naturally to brain attention mechanisms:
- Groups → Competing concept/persona clusters
- Layers → Abstraction levels (sensory → semantic → goals)
- λ → Competition strength (attention focus)
- μ → Binding/synchronization between layers

### 9.3 Engineering Trade-offs

| Aspect | Classic RAG | Hopfield | Our Framework |
|--------|-------------|----------|---------------|
| Multi-context | Poor | Good | Excellent |
| Controllability | Low | Medium | High |
| Interpretability | Medium | Low | High |
| Computational cost | Low | Medium | Low |

---

## 10. Quick Start for Next LLM

```python
# 1. Load the memory module
import sys
sys.path.insert(0, 'F:/hypergraph_bistability/src')
from agent.agent_memory import AgentMemory

# 2. Create memory
mem = AgentMemory(k=3, L=2, auto_adjust=True)
mem.group_labels = ["professional", "friendly", "technical"]
mem.layer_labels = ["preferences", "context"]

# 3. Write memories
mem.write("User likes code examples", group=2, layer=0)

# 4. Process prompts
result = mem.process_prompt("I need help but I'm tired")
print(f"λ adjusted to: {mem.lambda_}")

# 5. Get LLM context
print(mem.get_context_for_llm())

# 6. Save state
mem.save("my_agent_memory.json")
```

---

## 11. Contact / Context

This project was developed through iterative research between human and LLM, verifying theoretical claims against numerical simulations.

**Key validation approach**:
1. Derive mathematical predictions
2. Implement numerical simulations (scipy.integrate, Sobol sampling)
3. Compare theory vs simulation
4. Fix bugs and iterate

**Validation standard**: Every theoretical claim must be verified numerically before being added to the paper.

---

*End of Handoff Document*