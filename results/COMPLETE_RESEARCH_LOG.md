# Complete Research Findings: Hypergraph Bistability Project

## Project Summary

This document records all verified findings, corrections, and new research directions from the hypergraph bistability project. Last updated: 2026-03-20.

---

## Part I: Corrections to Prior Work

### Bug Fixes

| File | Bug | Fix |
|------|-----|-----|
| `core/model.py` | Growth rule always created binary edges (`p_pair` effectively =1.0), locking model in fragmented phase | Added `p_pair=0.3` parameter; growth rule now creates ternary edges based on `p_pair` |
| `verify_layered_cascade.py` | Wrong closed-form λ_c formula: `λ_c = (k-1)/(16·n_high)` | Replaced with numerical binary search computation |
| `verify_layered_cascade.py` | Step 4 imported from wrong path | Fixed to `core.model.MultiGroupHypergraph` |

### Paper Corrections

| Section | Original | Corrected |
|---------|----------|-----------|
| Abstract | "62% local switching success" vague | Corrected: 62% only under non-standard dynamics |
| 3.7 | Local Boost 62% unexplained | Fixed: standard dynamics = 0%, non-standard = 35-62% |
| 3.8 | (new section) | Added semi-analytic λ_c formula |
| 4.2 | "λ_c ≈ 0.35" | "λ_c ≈ 0.03-0.08" |
| 4.3 | "no closed form" | Semi-analytic formula: λ_c = 3h²l²/[h²+(k-1)l²] |

---

## Part II: Core Verified Results

### 1. Attractor Count at λ→0

- **N_att = 2^{k×L}** — exact, verified for k=2,3,4 and L=1,2
- Previous "missing attractors" (135/256) were sampling artifacts

### 2. λ_c Values (Standard Bistable Form)

Using standard bistable drive f(m) = m(1-m)(2m-1):

| k | n_high | λ_c (numerical) | Power-law ≈ 0.70/k² |
|---|--------|------------------|---------------------|
| 2 | 1 | 0.1667 | 0.1750 (5% error) |
| 3 | 1 | 0.0675 | 0.0778 (15% error) |
| 4 | 1 | 0.0437 | 0.0438 (0.1% error) |
| 5 | 1 | 0.0280 | 0.0280 (exact) |
| 6 | 1 | 0.0194 | 0.0194 (exact) |

### 3. Semi-Analytic Formula (n_high=1)

At the saddle-node bifurcation:

$$\lambda_c = \frac{3h^2 l^2}{h^2 + (k-1)l^2}$$

where h = H* − 1/2, l = L* − 1/2 are deviations of the coexisting fixed points from the unstable midpoint.

**Derivation**: Set det(J) = 0 and eliminate λ from the fixed-point equations.

**Error**: < 0.3% for k = 2, 3, 4

### 4. Control Strategies (k=3, L=2)

| Strategy | Setting | λ=0.05 | λ=0.1 | λ=0.2 |
|----------|---------|--------|-------|-------|
| Global Pull | Standard | 100% | 100% | 100% |
| Global Boost | Standard | 100% | 100% | 100% |
| Local Boost | Standard dynamics | 0% | 0% | 0% |
| Local Boost | Non-standard | 35-45% | 35-45% | 35-45% |

---

## Part III: Q1-Q5 Research Findings

### Q1: λ_c(L) Scaling Law

| L | λ=0.001 | λ=0.03 | λ=0.05 | λ=0.15 | λ_c(WTA) |
|---|---------|---------|---------|---------|----------|
| 1 | 8 | 8 | 2 | 2 | 0.0338 |
| 2 | 64 | 51 | 7 | 2 | N/A (gradual) |
| 3 | 135* | 81 | 6 | 2 | N/A |
| 4 | 130* | 96 | 6 | 2 | N/A |

*L=3,4: N_att < 2^{k×L} due to insufficient Sobol sampling

**Key findings**:
1. **L=1 has discrete phase transition**: 8→2 at λ_c ≈ 0.034
2. **L≥2 have gradual transitions**: No sharp bifurcation
3. **Critical dimension L_c = 2**: Phase transition nature changes between L=1 and L=2

### Q2: (λ, μ) Phase Diagram (k=3, L=2)

| μ\λ | 0.001 | 0.01 | 0.03 | 0.05 | 0.08 | 0.15 |
|-----|-------|------|------|------|------|------|
| **-0.4** | 13 | 13 | 13 | 10 | 10 | 10 |
| **-0.3** | 13 | 13 | 13 | 10 | 10 | 10 |
| **-0.2** | 21 | 16 | 14 | 10 | 10 | 12 |
| **-0.1** | 51 | 50 | 38 | 26 | 23 | 10 |
| **0.0** | 64 | 63 | 50 | 8 | 4 | 2 |
| **0.1** | 8 | 8 | 8 | 2 | 2 | 3 |
| **0.2** | 8 | 9 | 7 | 3 | 3 | 3 |
| **0.3** | 9 | 9 | 9 | 3 | 3 | 3 |

**Three regimes**:
- **μ < -0.1 (Anti-sync)**: WTA suppressed, N_att ≈ 10
- **μ ≈ 0 (Standard)**: Normal bistable cascade
- **μ > 0.1 (Sync)**: Dimensional collapse to effective k-dim

### Q3: Noise Escape Rate (k=3, L=1)

| λ\σ | 0.00 | 0.10 | 0.20 | 0.30 | 0.50 |
|-----|------|------|------|------|------|
| 0.05 | 0% | 0% | 0% | 40% | 13% |
| 0.10 | 0% | 0% | 0% | 40% | 27% |
| 0.20 | 0% | 0% | 0% | 40% | 33% |

**Critical noise σ_c ≈ 0.2-0.3**

### Q4: Asymmetric k (k=3, L=1)

| λ | Symmetric | Asym1 | Asym2 |
|---|-----------|-------|-------|
| 0.001 | 8 | 8 | 8 |
| 0.030 | 8 | 6 | 5 |
| 0.050 | 2 | 4 | 4 |
| 0.080+ | 2 | 2 | 2 |

**Asymmetry affects intermediate states but not final WTA**

### Q5: Local Control Verification

| Setting | Success |
|---------|---------|
| Fixed local boost (standard) | **0%** |
| PPO learned (standard) | **0%** |
| Non-standard dynamics (PPO) | 35-45% |

**Conclusion**: 62% is an artifact of non-standard dynamics. Local control cannot overcome global coupling under standard bistable dynamics.

---

## Part IV: Agent Memory Module (Direction B)

### Architecture

```
AgentMemory (k=3, L=2)
├── M[i,l] ∈ [0,1] - State matrix
├── group_labels = ["professional", "friendly", "technical"]
├── layer_labels = ["preferences", "context"]
└── modes = ["neutral", "exploratory", "focused", "creative", ...]
```

### Core Interface

| Function | Description |
|----------|-------------|
| `read()` | Get current memory state (binarized) |
| `write(content, group, layer)` | Write memory to specific slot |
| `switch_mode(mode)` | Change coupling parameters |
| `process_prompt(prompt)` | Auto-detect conflict, adjust λ |
| `get_context_for_llm()` | Generate context string for LLM |
| `save()` / `load()` | JSON persistence |

### Conflict Detection

```python
conflict_level = ConflictDetector.detect_conflict_level(prompt)
# Uses keyword matching: "but", "however", "although" → high conflict
# Output: 0.0 (calm) to 1.0 (highly conflicting)
```

### λ Auto-Adjustment

| Conflict | λ | Mode |
|----------|---|------|
| < 0.1 | 0.01 | exploratory |
| 0.1-0.3 | 0.03-0.06 | neutral |
| 0.3-0.5 | 0.06-0.10 | focused |
| > 0.5 | 0.10-0.15 | near-WTA |

### Files

| File | Description |
|------|-------------|
| `src/hypergraph_control.py` | λ_c computation |
| `src/agent/agent_memory.py` | Agent memory module |
| `src/agent/llm_integration.py` | LLM integration examples |

### Usage Example

```python
from agent.agent_memory import AgentMemory

mem = AgentMemory(k=3, L=2, auto_adjust=True)
mem.group_labels = ["professional", "personal", "technical"]

# Write memories
mem.write("User prefers concise answers", group=0, layer=0)
mem.write("Likes code examples", group=2, layer=0)

# Process prompt with auto λ adjustment
result = mem.process_prompt("I want to learn coding but I'm also tired")
# → λ adjusted from 0.0 to 0.06 based on conflict detection

# Get context for LLM
context = mem.get_context_for_llm()
# → "[Memory State: neutral]\nActive contexts: professional: User prefers concise..."
```

---

## Part V: Theoretical Summary

### What Works
- **Global Pull/Boost**: 100% success
- **N_att = 2^{k×L}**: Exact
- **λ_c semi-analytic formula**: < 0.3% error

### What Doesn't Work
- **Local control (standard)**: 0% success
- **PPO under standard dynamics**: 0% success

### New Understanding
- **μ is a dimensionality switch**
- **L≥2: gradual transition** (not discrete)
- **Critical noise σ_c ≈ 0.2-0.3**
- **Asymmetry: affects intermediate, not final WTA**

---

## Part VI: Files Created/Modified

### Core Files
| File | Changes |
|------|---------|
| `src/core/model.py` | Added p_pair parameter |
| `src/verify_layered_cascade.py` | Numerical λ_c, import fix |
| `src/hypergraph_control.py` | **NEW**: λ_c computation |
| `src/agent/agent_memory.py` | **NEW**: Agent memory module |
| `src/agent/llm_integration.py` | **NEW**: LLM integration |

### Results
| File | Description |
|------|-------------|
| `results/Q1_lambda_c_scaling/` | L→∞ scaling data |
| `results/Q2_lambda_mu_phase/` | (λ,μ) phase diagram |
| `results/Q3_noise_escape/` | Noise escape rates |
| `results/Q4_asymmetric_k/` | Asymmetric k data |
| `results/L_infinity/` | L→∞ limit study |
| `results/COMPLETE_RESEARCH_LOG.md` | This document |

### Paper
| File | Changes |
|------|---------|
| `paper/paper_final.md` | Section 3.7 (local control), 3.8 (λ_c formula), 4.3 (corrections) |

---

## Appendix: Key Parameter Values

| Parameter | Value | Meaning |
|-----------|-------|---------|
| α | 1.0 | Bistable amplitude |
| a | 0.5 | Bistable midpoint |
| f(m) | m(1-m)(2m-1) | Standard bistable drive |
| k | 3 | Number of groups |
| L | 1-4 | Number of layers |
| λ_c | ~0.03-0.17 | WTA threshold (k-dependent) |
| σ_c | ~0.2-0.3 | Critical noise |

---

*Last updated: 2026-03-20*