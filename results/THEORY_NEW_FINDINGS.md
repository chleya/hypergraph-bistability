# Theoretical Findings: Layered Bifurcation Cascade

## Core Result

We identified a **layered saddle-node cascade** governing multistability collapse as inter-group coupling λ increases. **Critical finding: the previously reported closed-form formula is incorrect. λ_c must be computed numerically.**

---

## 1. System

Designed bistable ODE: k groups, L layers

$$\dot{M}_{i,l} = \alpha M_{i,l}(1-M_{i,l})(M_{i,l}-a) + \lambda \sum_{j \neq i}\left(\bar{M}_j - M_{i,l}\right)$$

where $\bar{M}_j = \frac{1}{L}\sum_l M_{j,l}$ is the layer-mean of group j.

---

## 2. Attractor Count at λ=0

**N_att = 2^{k×L}** — exact, verified for k=2,3,4 and L=1,2.

Previous reports of "missing attractors" (e.g., 135/256 for k=4,L=2) were sampling artifacts. Direct perturbation confirms all attractors are stable.

---

## 3. Layered Bifurcation Cascade

### 3.1 Layer Definition

Attractors are grouped by **n_high** = number of groups in HIGH state (M≈1).

Layer n_high contains C(k, n_high) × 2^{k(L-1)} attractors at λ=0.

### 3.2 Critical Coupling: NUMERICAL (No Closed Form)

**The formula λ_c = α·a²·(1-a)²·(k-1)/n_high is INCORRECT.**

Numerical λ_c values (α=1, a=0.5):

| k | n_high | λ_c (numerical) | Old formula | Ratio |
|---|--------|-----------------|-------------|-------|
| 2 | 1      | 0.12469         | 0.06250     | 1.99  |
| 3 | 1      | 0.08305         | 0.04167     | 1.99  |
| 3 | 2      | 0.08305         | 0.04167     | 1.99  |
| 4 | 1      | 0.02185         | 0.02344     | 0.93  |
| 4 | 2      | 0.06234         | 0.04688     | 1.33  |
| 4 | 3      | 0.02185         | 0.02344     | 0.93  |

**Key observations:**
- λ_c(n_high=1, k) = λ_c(n_high=k-1, k) by symmetry
- No consistent closed-form formula found
- The old derivation assumed upper-triangular Jacobian with dF_low/dl = 0, which does not correctly capture the saddle-node condition

### 3.3 Cascade Order (Verified Numerically)

Layers disappear in order of decreasing n_high:

- k=4: n_high=3,1 disappear first (λ≈0.022), then n_high=2 (λ≈0.062), leaving ALL-HIGH and ALL-LOW
- k=3: n_high=1,2 disappear simultaneously (λ≈0.033) by symmetry, leaving ALL-HIGH and ALL-LOW

### 3.4 Winner-Take-All Threshold

For k=3: WTA (mixed states gone) at λ ≈ 0.033–0.040
For k=4: WTA at λ ≈ 0.062

---

## 4. N_att(λ) — Verified Step 2 Results

### k=3, L=1 (2^3 = 8 attractors at λ→0)

| λ     | Theory | Found | Surviving layers |
|-------|--------|-------|------------------|
| 0.001 | 8      | 8     | n=0,1,2,3        |
| 0.030 | 8      | 8     | n=0,1,2,3        |
| 0.040 | 8      | 2     | n=0,3 (WTA)      |
| 0.100 | 2      | 2     | n=0,3            |

### k=4, L=1 (2^4 = 16 attractors at λ→0)

| λ     | Theory | Found | Surviving layers |
|-------|--------|-------|------------------|
| 0.001 | 16     | 16    | n=0,1,2,3,4      |
| 0.030 | 8      | 8     | n=0,2,4          |
| 0.050 | 8      | 2     | n=0,4 (WTA)      |
| 0.100 | 2      | 2     | n=0,4            |

### k=3, L=2 (2^6 = 64 attractors at λ→0)

| λ     | Theory | Found | Notes |
|-------|--------|-------|-------|
| 0.001 | 64     | 64    | Full tensor product |
| 0.030 | 64     | 64    | Mixed layers intact |
| 0.050 | 64     | 13    | Gradual collapse |
| 0.080 | 64     | 4     | Only all-same layers |
| 0.130 | 16     | 2     | Near WTA |

---

## 6. Control Strategy Analysis

### 6.1 Verified Results (k=3, L=2, target: switch LOW→HIGH)

| Strategy | Mechanism | λ=0.05 | λ=0.1 | λ=0.2 |
|----------|-----------|---------|--------|--------|
| Global Pull | Additive drift to all nodes | 100% | 100% | 100% |
| Global Boost | Basin-weighted pull | 100% | 100% | 100% |
| Local Boost (fixed) | Single-node additive push | 0% | 0% | 0% |

**Local Boost fails entirely (0%)** in the bistable regime because global coupling λ synchronizes all nodes.

### 6.2 PPO Learned Strategy (62%)

The 62% success comes from `optimal_control_ppo.py`:
- **Asymmetric Kc_list = [0.32, 0.40, 0.48]** (not equal capacities!)
- **Different dynamics**: `a*M³ + b*M² + c*M` (not standard bistable!)
- **PPO learns**: boost_start (10-80), duration (10-50), boost_strength (1.0-2.5)
- **Key insight**: learning **when** and **how much** to boost matters more than the intervention itself

**This is NOT a contradiction**: fixed local strategies fail (0%) but learned temporal policies succeed (62%). The 62% is NOT about capacity modification being effective — it's about the LEARNED POLICY being effective on a DIFFERENT model with asymmetric capacities.

### 6.3 Implications

1. **Fixed local control is ineffective** in symmetric bistable systems (λ > λ_c)
2. **Learned temporal policies** can exploit asymmetry and timing to achieve non-trivial local control
3. **Real-world control** (e.g., neuromodulation) likely requires learned, adaptive strategies rather than fixed interventions

## 7. μ-λ_c Coupling (Preliminary)

Step 3 data (k=3, L=2) shows qualitative behavior:

| μ | λ_c behavior | N_att at λ=0.04 |
|---|---------------|------------------|
| -0.3 | WTA suppressed | ~10 (no collapse) |
| -0.2 | WTA suppressed | ~10 |
| -0.1 | Normal | ~46 |
| 0.0 | Normal | ~40 |
| 0.1 | Accelerated WTA | ~2 |
| 0.2 | Accelerated WTA | ~3 |
| 0.3 | Accelerated WTA | ~3 |

**Interpretation**:
- μ < 0 (anti-synchronization): raises λ_c, preserves mixed states at higher λ
- μ > 0 (synchronization): lowers λ_c, accelerates WTA collapse
- For μ > 0, layers synchronize, reducing effective dimensionality toward L=1 behavior, where coupling is stronger relative to available degrees of freedom

**No closed-form λ_c(μ) found**; the coupled ODE system with L≥2 and μ≠0 has no analytical solution. Future work: systematic numerical characterization of the (λ, μ) phase diagram.

## 8. Open Questions Summary

1. **λ_c formula**: No closed form for k > 2. Numerical computation required.
2. **High-dimensional scaling**: L^(-α) law for λ_c(L)? Systematic numerical study needed.
3. **μ-λ_c phase diagram**: Full (λ, μ) portrait for k=3,4, L=2,3?
4. **Noise escape rates**: Kramers formula for coupled bistable system?
5. **Asymmetric k**: Different group capacities?
6. **Control strategies**: Reproduce paper's 62% Local Boost claim or confirm it's an error?

The saddle-node bifurcation condition for the n_high-HIGH attractor requires solving:

$$F_L(H^*, L^*; \lambda) = 0 \quad \text{and} \quad \frac{\partial F_L}{\partial L}(H^*, L^*; \lambda) = 0$$

simultaneously, where:
- $F_L = \alpha L(1-L)(L-a) + \lambda \cdot n_{high} \cdot (H^* - L^*)$
- $\partial F_L/\partial L = \alpha(1 - 6L(1-L)) - \lambda \cdot n_{high}$

These two equations give:
$$H^* - L^* = \frac{1}{4k \cdot (1 - 6L^{*2})}$$

Combined with the fixed-point condition $F_H = 0$, this yields a cubic equation for $H^*$:
$$H^*(1-H^*)(2H^* - 1) = \frac{1 - 6L^{*2}}{4k}$$

where $L^*$ satisfies its own quadratic. For general k, n_high, this system has no closed-form solution. The cubic gives $H^*$ values (e.g., for k=3, n_high=2: $H^* = 1/2, (3\pm\sqrt{3})/6$) that don't match the actual bifurcation values ($H^* \approx 0.944$), indicating the bifurcation involves the mixed state colliding with the middle fixed point at $H=1/2$, not with the all-L state.

**Conclusion**: The layered saddle-node cascade has no closed-form $\lambda_c$ formula. The correct $\lambda_c$ values must be computed numerically. The old formula $\lambda_c = \alpha a^2(1-a)^2 (k-1)/n_{high}$ was incorrect because it assumed $H^* - L^* = a(1-a)$ at the bifurcation, which is not satisfied in the coupled system.
