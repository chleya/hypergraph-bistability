# Theoretical Framework Summary

## Updated: 2026-03-19

## Core Distinction

The theory now makes a clear distinction between two regimes:

### Regime 1: Complexity Reconstruction (MICROSCOPIC)
- **Mechanism**: Local rules (growth/split/fusion/delete) on hypergraph
- **Result**: 96.4% ± 8.6% of core structures are NEWLY GENERATED
- **Transport**: Negligible (fusion ≈ 0, blocking has no effect)
- **Type**: STOCHASTIC FRAGMENTATION
- **M values**: Continuous spectrum [0, ~0.56]
- **Status**: VERIFIED EMERGENT PHENOMENON

### Regime 2: Multistability (DESIGNED)
- **Mechanism**: Explicit bistable drift function F(M) = α·M·(1-M)·(M-a)
- **Result**: Hard bistability at M=0 and M=1
- **Attractor count**: N_att = 2^{k×L} (tensor product)
- **Coupling effects**: λ induces winner-take-all collapse
- **Type**: ENGINEERED (not emergent)
- **Status**: THEORETICAL FRAMEWORK

## Key Equations

### Designed Bistable Drift
```
dM/dt = α · M · (1 - M) · (M - a) + λ · Σ(M_j - M_i) + μ · Σ(M_i,l' - M_i,l)
```

### Stability Analysis
- Fixed points: M = 0, M = a, M = 1
- Stable: M = 0 and M = 1
- Unstable: M = a

### Jacobian at λ=0, μ=0
- Block diagonal with k×L independent 1×1 blocks
- Each block eigenvalue: α · M* · (1 - M*) · (M* - a)
- D_eff = k × L

## Phase Transitions

| λ | Effect | N_att (k=3, L=2) |
|---|--------|-------------------|
| 0.0 | Independent bistable elements | 35-47 |
| 0.3 | Winner-take-all begins | 12-20 |
| 0.5 | Near-complete collapse | 2-4 |

## Limitations

1. **Hard bistability is designed, not emergent**: The tensor-product multistability arises from the explicitly designed drift function, not from microscopic rules
2. **Microscopic rules produce stochastic fragmentation**: M values are continuous, not discrete
3. **Two regimes are complementary**: Complexity reconstruction (microscopic) and multistability (designed) describe different aspects of the system

## Revised Theoretical Claims

### Claims That HOLD (Verified):
- Phase transition at k_c ≈ 2.35
- Order-complexity separation (r = ±0.97)
- Transport is negligible (fusion ≈ 0)
- 96.4% new structures (rigorous parent chain tracing)
- Blocking has no effect on core complexity

### Claims That Must Be QUALIFIED:
- "Dimension controls multistability" → ONLY in the DESIGNED model
- "2^{k×L} attractors" → ONLY in the DESIGNED model with explicit bistable drift
- "Natural emergence of multistability" → NOT SUPPORTED; it's engineered

## Updated Paper Narrative

The paper now tells a coherent story:

1. **Part 1 (Complexity Reconstruction)**: Microscopic hypergraph rules produce robust complexity reconstruction through in-situ generation, NOT transport. This is verified through rigorous experiments.

2. **Part 2 (Multistability Framework)**: Separately, we introduce a designed model with explicit bistable drift that enables dimension-controlled multistability. This is a THEORETICAL FRAMEWORK, not an emergent phenomenon.

The distinction is made explicit in the paper:
- Section 3.4 is titled "Designed Multistability"
- Section 4.3 (Limitations) explicitly notes the distinction
- Abstract and conclusion reflect the engineered nature of multistability

## Value of the Framework

Even though multistability is "engineered" rather than "emergent", the framework has value:

1. **Predictive power**: Once parameters (k, L, λ, μ) are set, attractor behavior is deterministic
2. **Control**: Coupling provides a knob to tune multistability
3. **Design principles**: Shows how to build systems with specific multistability properties
4. **Connection to real systems**: Similar coupling mechanisms appear in neural and social systems

## References to Updated Files

- Paper: `paper/paper_final.md` (updated)
- Key findings: `results/theoretical_framework_update.json`
- Microscopic verification: `results/verification_b/key_finding.json`
- Designed model analysis: `results/jacobian_analysis/full_analysis.json`
