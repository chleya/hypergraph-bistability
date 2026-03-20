# Final Project Report
## Hypergraph Bistability and Complexity Reconstruction
**Date**: 2026-03-19

---

## Executive Summary

This project investigated two interrelated phenomena in hypergraph dynamics:

1. **Complexity Reconstruction**: How ordered structures emerge without transport
2. **Multistability and Dimension Ascension**: How dimensionality controls attractor count

**Key Finding**: The two phenomena have fundamentally different origins:
- Complexity reconstruction is a **verified emergent phenomenon** (96.4% new structures)
- Multistability in the high-dimensional regime is an **engineered design**, not emergence

---

## Part I: Complexity Reconstruction

### Verified Results

| Metric | Value | Status |
|--------|-------|--------|
| Fusion events per 80 steps | 0.7 ± 0.8 | Negligible |
| Peri→core fusion | **0.0** (never) | Negligible |
| Block experiment p-value | 1.0 | No effect |
| H_core growth (8 configs) | All positive | Invariant |
| **New structures (rigorous)** | **96.4% ± 8.6%** | Verified |

### Conclusion

**Complexity Reconstruction is REAL**: The overwhelming majority of core structures are newly generated in place, not transported from periphery.

---

## Part II: Multistability and Dimension Ascension

### Two Regimes Discovered

| Regime | Mechanism | Result |
|--------|-----------|--------|
| **Microscopic** | Growth/split/fusion rules | Stochastic fragmentation (M ∈ [0, ~0.56]) |
| **Designed** | Explicit drift F(M) = α·M·(1-M)·(M-a) | Hard bistability (M=0, M=1) |

### Designed Model Results

| System | λ=0 Attractors | 2^{k×L} | λ_c |
|--------|---------------|---------|-----|
| k=1, L=1 | 2 | 2 | - |
| k=2, L=1 | 4 | 4 | ≈0.09 |
| k=3, L=1 | 8 | 8 | ≈0.04 |
| k=2, L=2 | 16 | 16 | ≈0.18 |

### D_eff Definition

**D_eff = k × L** (number of independent bistable elements)

At λ=0, μ=0:
- Jacobian is block diagonal (k×L independent 1×1 blocks)
- Each block eigenvalue: α·M*·(1-M*)·(M*-a)
- All negative → D_eff = k×L confirmed

### Coupling Effects

| Coupling | Effect | Mechanism |
|----------|--------|----------|
| λ > 0 | Winner-take-all | Forces inter-group synchronization |
| μ > 0 | Synchronization | Forces same-group layers to sync |
| μ < 0 | Competition | Maintains layer independence |

---

## Paper Updates (2026-03-19)

### Changes Made

1. **Abstract**: Changed to "engineered through explicit bistable drift function"
2. **Section 3.4**: Renamed to "Designed Multistability: Dimension as a Control Parameter"
3. **Section 3.4**: Added note that multistability arises from designed drift, not microscopic rules
4. **Section 4.1**: Added distinction between microscopic (stochastic) vs designed (hard bistability)
5. **Section 4.3**: Added limitation noting hard bistability is designed, not emergent
6. **Conclusion**: Changed to "dimension and coupling can be engineered"

---

## Theoretical Framework

### The Two-Layer Theory

```
┌─────────────────────────────────────────────────────────────┐
│                    THEORETICAL FRAMEWORK                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Layer 1: Microscopic Rules (VERIFIED)                      │
│  ─────────────────────────────────────                       │
│  • Growth, Split, Fusion, Deletion                           │
│  • Produce: Stochastic fragmentation                         │
│  • M values: Continuous [0, ~0.56]                           │
│  • Key Result: 96.4% new structures (Complexity Reconstruction)│
│                                                              │
│  Layer 2: Designed Drift (THEORETICAL)                       │
│  ─────────────────────────────────                          │
│  • F(M) = α·M·(1-M)·(M-a) + λ coupling                    │
│  • Produce: Hard bistability (M=0, M=1)                      │
│  • N_att = 2^{k×L} (tensor product)                         │
│  • λ induces winner-take-all collapse                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Claims Assessment

| Claim | Evidence | Status |
|-------|----------|--------|
| Phase transition at k_c ≈ 2.35 | Experimental | ✓ VERIFIED |
| Order-complexity separation (r = ±0.97) | Experimental | ✓ VERIFIED |
| Transport is negligible | Experimental | ✓ VERIFIED |
| 96.4% new structures | Rigorous tracking | ✓ VERIFIED |
| N_att = 2^{k×L} | Theoretical model | ⚠️ DESIGNED |
| Multistability emerges | Microscopic verification | ✗ NOT SUPPORTED |

---

## Files Updated

### Paper
- `paper/paper_final.md` - Updated with honest "engineered" framing

### Results
- `results/theoretical_framework_update.json` - Summary of changes
- `results/THEORETICAL_FRAMEWORK_SUMMARY.md` - This document

### Key Verification Files
- `results/verification_b/key_finding.json` - Microscopic model findings
- `results/jacobian_analysis/full_analysis.json` - D_eff and λ_c analysis
- `results/continuous_spectrum/key_finding.json` - Soft vs hard bistability

---

## Conclusion

The theory is now built on solid foundations:

1. **Complexity Reconstruction** is a genuine emergent phenomenon, rigorously verified through experiments showing 96.4% new structures and negligible transport.

2. **Multistability** is presented as an engineered framework, not an emergent property. This honesty strengthens the paper by avoiding overclaiming.

3. **The tensor-product structure N_att = 2^{k×L}** describes the designed model, providing a framework for understanding how coupling controls dimensionality.

The revised paper tells a coherent story: hypergraph dynamics produce complexity reconstruction (verified), and separately, we provide a theoretical framework for dimension-controlled multistability (engineered).

---

## References

- Original paper: `paper/paper_final.md`
- Theoretical summary: `results/THEORETICAL_FRAMEWORK_SUMMARY.md`
- Experimental results: `results/` directory
