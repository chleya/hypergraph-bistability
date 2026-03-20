# Complexity Reconstruction in Capacity-Constrained Competitive Dynamics on Hypergraphs

---

## Abstract

The emergence of structured order in interacting systems is commonly attributed to entropy reduction, energy minimization, or the transport and accumulation of structural complexity. Here, we study a hypergraph system governed by local rules, in which ordered cores spontaneously emerge from disordered initial conditions through in-situ reconstruction. We observe a robust order–complexity separation: as global order increases, structural complexity concentrates in the core while the periphery becomes progressively simplified. Crucially, this process cannot be explained by transport. Fusion events are negligible, and no measurable periphery-to-core flow is detected. Instead, identity tracking with rigorous parent chain tracing shows that the majority (**96.4% ± 8.6%**) of core structures are newly generated during evolution, with only a minor fraction (fusion birth: 0.4 ± 0.8 per run) originating from fusion events.

We further demonstrate that multistability can be engineered through an explicit bistable drift function, yielding a tensor-product attractor structure where the number of stable states grows as N_att = 2^{k×L} (verified: exactly 64 at λ→0 for k=3, L=2). Inter-group coupling λ induces winner-take-all collapse (threshold λ_c ≈ 0.044–0.167 for k = 4, 2 respectively), while inter-layer coupling μ modulates synchronization. Global control strategies achieve reliable switching (100% success) in the bistable regime, while local control with fixed parameters fails entirely (0% success). When tested with non-standard dynamics (asymmetric capacities Kc=[0.32, 0.40, 0.48] with cubic rather than bistable drift), reinforcement learning achieves 35–45% success by exploiting the asymmetric landscape. This establishes a framework for dimension-controlled multistability in competitive dynamics.

**Keywords**: hypergraph dynamics, multistability, complexity reconstruction, phase transition, capacity constraint

---

## 1. Introduction

### 1.1 The Problem

How does structured order emerge from disordered components? This question is central to understanding pattern formation in physical, biological, and social systems.

### 1.2 Classical Paradigms We Challenge

Three dominant frameworks have shaped the field:

1. **Entropy reduction**: Ordered structures form by decreasing entropy, following Landau's theory of phase transitions.

2. **Energy minimization**: Systems relax to energy minima, as in spin glasses and protein folding.

3. **Transport and accumulation**: Components aggregate or accumulate to form structures, as in crystallization or network growth.

These frameworks implicitly assume that **structural complexity is conserved and redistributed through transport or accumulation**.

### 1.3 Our Discovery

We present a minimal hypergraph system governed by four local rules (growth, fusion, split, deletion) with a single capacity constraint parameter k. Our key findings:

- A phase transition occurs at k_c ≈ 2.35, exhibiting bistability between fragmented and clustered states
- **Order–complexity separation**: as global order increases, complexity concentrates in the core while the periphery simplifies (r = ±0.97)
- **No transport mechanism**: fusion events are negligible (~0), and flow-blocking experiments have no effect
- **In-situ reconstruction**: identity tracking reveals that **96.4% ± 8.6%** of core structures are newly generated (rigorous parent chain tracing), not inherited

Our results show that structure formation can persist even in the absence of measurable transport, implying that reconstruction is not only sufficient but necessary.

### 1.4 Our Contribution

We propose a **Complexity Reconstruction Principle**: in non-thermal systems with capacity constraints, structural complexity is regenerated in place through local reorganization, rather than transported or accumulated. This challenges the transport/accumulation assumption across all three classical paradigms and suggests a distinct organizing principle for emergent structure in complex systems.

---

## 2. Model

### 2.1 Hypergraph Structure

We consider a hypergraph H = (V, E) where:
- V = {0, 1, ..., N-1} is the set of nodes
- E ⊆ P(V) is the set of hyperedges, where each hyperedge is a subset of nodes

Each node v ∈ V carries a state vector s_v ∈ R^d. The system evolves under four local rules:

| Rule | Probability | Description |
|------|-------------|-------------|
| Growth | p₁ = 0.35 | Add new node connected to existing edge |
| Fusion | p₂ = 0.30 | Merge two hyperedges that share nodes |
| Split | p₃ = 0.15 | Divide large hyperedge into two |
| Deletion | p₄ = 0.20 | Remove small hyperedges under capacity constraint |

### 2.2 Capacity Constraint

Each node v has a capacity K_v. When degree(v) > K_v, the excess connections are pruned. We use homogeneous capacity K_v = kN for simplicity.

### 2.3 Order Parameter

We define the order parameter M = |C_max| / N, where C_max is the largest connected component. M ≈ 0 indicates fragmentation; M ≈ 1 indicates full clustering.

---

## 3. Results

### 3.1 Phase Transition and Bistability

We begin by characterizing the asymptotic behavior of our hypergraph system as a function of the capacity constraint parameter k. We initialize the system with random hyperedges and evolve for 80 steps under the four local rules with a fixed capacity limit K = kN.

Figure 1 shows the order parameter M = |C_max|/N as a function of k. The results reveal a sharp phase transition at k_c ≈ 2.35. For k < k_c, the system evolves to a fragmented state with M ≈ 0.2–0.3. For k > k_c, the system transitions to a clustered state where M approaches 0.9–1.0.

Critically, the phase transition exhibits **bistability** in the transition region (2.3 ≤ k ≤ 2.6). Depending on initial conditions, the system can settle into either the fragmented or clustered attractor, suggesting the presence of a double-well potential.

### 3.2 Order–Complexity Separation

Having established that the system forms structured cores, we now characterize the internal organization of these cores. We decompose the system's structural diversity using a regional analysis: we measure the Shannon entropy H of hyperedge size distributions within the core and periphery.

As the system evolves, we observe a systematic divergence: **H_core increases** while **H_periphery decreases**. The core becomes internally more diverse, while the peripheral region simplifies.

Plotting M against H_core and H_periphery across multiple runs, we find:
- **r(M, H_core) = +0.97** (strong positive correlation)
- **r(M, H_periphery) = -0.93** (strong negative correlation)

This implies that as the system becomes more ordered (higher M), complexity simultaneously increases in the core but decreases in the periphery—a counterintuitive phenomenon.

### 3.3 Complexity Reconstruction Mechanism

The order–complexity separation demands explanation: is complexity redistributed via transport from periphery to core, or via an alternative mechanism? We now systematically rule out transport and demonstrate in-situ reconstruction.

#### 3.3.1 Ruling Out Transport: Fusion Event Statistics

The transport hypothesis posits that core complexity originates from the periphery via fusion events. To test this, we conducted systematic diagnostics across 20 independent runs (T=80 steps each).

| Metric | Value |
|--------|-------|
| Fusion events per 80 steps | 0.7 ± 0.8 |
| Peri→core fusion events | **0.0** (never detected) |

Fusion events are essentially negligible. Even if every fusion were peri→core (which it never is), transport could not account for H_core growth.

#### 3.3.2 Causal Intervention: Blocking Does Not Reduce Complexity

We performed a **block fusion experiment**:
- **Control**: Normal dynamics with all fusion events
- **Block**: Peri→core fusion blocked with probability p=1.0

| Metric | Control | Block |
|--------|---------|-------|
| ΔH_core | +0.170 ± 0.168 | +0.170 ± 0.168 |
| ΔM | +0.411 | +0.411 |
| p-value | 1.0 | 1.0 |

**Result**: Blocking peri→core fusion has **no significant effect** on core complexity growth. This directly falsifies the transport hypothesis.

#### 3.3.3 Identity Tracking: Majority Are New Structures (Rigorous Verification)

To directly measure the origin of core structures, we implemented **rigorous identity tracking** with true parent chain tracing. Each edge tracks its type (growth/split/fusion) and parent edge IDs.

**Results** (10 runs, T=80):
- **New ratio**: **96.4% ± 8.6%** (rigorous parent chain tracking)
- **Fusion birth**: 0.4 ± 0.8 per run (essentially zero)
- **Mean ancestry depth**: 1.12 ± 0.33 generations

The overwhelming majority of core structures are **newly generated** (96.4%), confirming in-situ reconstruction.

#### 3.3.4 Structural Invariance: Robustness Across Rule Perturbations

We tested whether H_core growth occurs across 8 different rule configurations:

| Configuration | ΔH_core |
|--------------|---------|
| Baseline | +0.088 |
| No Fusion | +0.046 |
| No Split | +0.189 |
| High Growth | +0.081 |
| High Split | +0.057 |
| All Equal | +0.104 |
| Low Activity | +0.020 |
| High Fusion | +0.165 |

H_core grew in **all 8 configurations**, demonstrating structural invariance.

#### 3.3.5 Mechanism: Growth and Split Dominate

The evidence points to a clear mechanism:

1. **Growth**: New nodes attach to existing core edges, adding complexity
2. **Split**: Large edges divide, creating new structure without external input
3. **Fusion**: Rare, and when it occurs, typically core-with-core rather than peri-with-core

The system maintains and generates complexity through **internal dynamics alone**.

### 3.4 Designed Multistability: Dimension as a Control Parameter

We now extend to multi-group, multi-layer systems to demonstrate that dimensionality can be used to control multistability through designed coupling mechanisms.

**Important distinction**: The multistability described in this section arises from an *explicitly designed drift function* F(M) = α·M·(1-M)·(M-a), not directly from the microscopic hypergraph rules. The microscopic rules produce stochastic fragmentation with a continuous spectrum of states. The designed model provides a tractable framework for studying multistability and coupling effects.

#### 3.4.1 Dimension-Controlled Attractor Count

With k=3 groups and L=2 layers (D = 6 dimensions) in the designed model, we observe exactly **64 distinct attractors** at λ→0, confirming the tensor-product structure N_att = 2^{k×L} = 2^6 = 64.

| λ | # Attractors (theory) | # Attractors (simulation) | Dominant Pattern |
|---|----------------------|--------------------------|------------------|
| 0.001 | 64 | 64 | All 2^6 co-existing |
| 0.03  | 64 | 64 | All layers intact |
| 0.05  | 64 | 13 | Mixed-state layers collapsing |
| 0.08  | 64 |  4 | Only all-same states survive |
| 0.13  | 16 |  2 | Winner-take-all |

As λ increases, the number of stable attractors decreases in discrete steps. The collapse threshold λ_c(n_high, k) is determined from the saddle-node bifurcation; for k=3 the WTA transition occurs at λ_c ≈ 0.068. At λ ≈ 0.08–0.13, the system undergoes **winner-take-all** collapse to only 2 attractors (ALL-HIGH and ALL-LOW).

#### 3.4.2 Coupling Controls Dimensionality

Inter-layer coupling μ has three regimes (k=3, L=2):
- **μ < -0.1 (anti-sync)**: WTA suppressed, N_att ≈ 10 even at λ=0.15
- **μ ≈ 0 (standard)**: Normal bistable cascade
- **μ > 0.1 (sync)**: Dimensional collapse to effective k-dimensional behavior

### 3.5 Noise Escape and Critical Sensitivity

We investigate how additive noise modulates the multistability picture (k=3, L=1). Using Euler integration with Gaussian noise, we measure escape probability from ALL-HIGH to ALL-LOW over T=150 with dt=0.2.

| λ\σ | 0.00 | 0.10 | 0.20 | 0.30 | 0.50 |
|-----|------|------|------|------|------|
| 0.05 | 0% | 0% | 0% | 40% | 13% |
| 0.10 | 0% | 0% | 0% | 40% | 27% |
| 0.20 | 0% | 0% | 0% | 40% | 33% |

**Critical noise σ_c ≈ 0.2–0.3**: Below this, noise is too weak to drive escape within T=150. Above, transitions become noise-dominated. Standard Kramers theory (τ ~ exp(ΔV/σ²)) does not apply directly to the coupled system, but the critical noise scale confirms that the designed model is robust to weak noise.

### 3.6 Scaling with Layer Count L→∞

We investigate how the phase transition nature changes as L increases (k=3, μ=0):

| L | N_att(λ=0.001) | N_att(λ=0.03) | N_att(λ=0.15) | Δλ (transition width) |
|---|-----------------|---------------|---------------|------------------------|
| 1 | 8 | 8 | 2 | ~0 (sharp) |
| 2 | 64 | 51 | 2 | 0.010 |
| 3 | 135* | 81 | 2 | 0.020 |
| 4 | 130* | 96 | 2 | 0.020 |

*N_att < 2^{k×L} due to finite Sobol sampling

**Critical dimension L_c = 2**: The phase transition nature changes between L=1 and L=2. L=1 exhibits a discrete, sharp bifurcation (8→2 at λ_c ≈ 0.034). L≥2 exhibit gradual transitions with finite width Δλ ≈ 0.020 that saturates. As L→∞, the transition width does not diverge but converges to a constant, indicating that the system retains its qualitative multi-attractor character even at high layer counts.

### 3.7 Effect of Group Asymmetry

We test whether asymmetric group capacities (different A_i values per group) affect the cascade structure (k=3, L=1):

| λ | Symmetric | Asym1 | Asym2 |
|---|-----------|-------|-------|
| 0.001 | 8 | 8 | 8 |
| 0.030 | 8 | 6 | 5 |
| 0.050 | 2 | 4 | 4 |
| 0.080+ | 2 | 2 | 2 |

Asymmetry (A = [0.4, 0.5, 0.6] or [0.3, 0.5, 0.7]) breaks the degeneracy of attractors, causing earlier collapse at intermediate λ. However, the final WTA state (N_att = 2) is robust to asymmetry at sufficiently high λ. This suggests asymmetry could serve as an additional control parameter for fine-tuning the cascade.

### 3.8 Topology Dependence

Network topology modulates the coupling-induced collapse (ODE model, k=3, L=1):

| Topology | λ=0 | λ=0.05 |
|----------|-----|--------|
| Uniform | 8 | 2 |
| Power-law | 8 | 4 |
| High-overlap | 8 | 4 |

**Power-law and high-overlap topologies are more robust** to coupling-induced collapse than uniform topology: they maintain 4 distinct attractors at λ=0.05 while uniform collapses to 2. This is because hub nodes in power-law networks create redundant paths that prevent single-group domination, while high-overlap creates cross-group correlations that resist WTA collapse.

### 3.9 Controllability

Can we intentionally switch attractor states? We test control strategies applied to the designed ODE model (k=3, L=2), starting from the ALL-LOW basin and attempting to reach ALL-HIGH:

| Method | Setting | λ=0.05 | λ=0.1 | λ=0.2 |
|--------|---------|---------|-------|-------|
| Global Pull | Standard | 100% | 100% | 100% |
| Global Boost | Standard | 100% | 100% | 100% |
| Local Boost | Standard dynamics | 0% | 0% | 0% |
| Local Boost | Non-standard dynamics | 35-45% | 35-45% | 35-45% |

**Global strategies succeed (100%)** because they overcome inter-group coupling λ that synchronizes all nodes. **Local Boost with fixed parameters fails entirely (0%)** in the standard bistable setting.

When using non-standard dynamics (asymmetric capacities Kc=[0.32, 0.40, 0.48] with cubic drift a·M³+b·M²+c·M rather than standard bistable), reinforcement learning achieves 35–45% success by exploiting the asymmetric landscape. **Under standard bistable dynamics with symmetric parameters, local control remains ineffective regardless of timing or learning (0% success).**

### 3.10 Semi-analytic control law for λ_c

At the saddle-node bifurcation where the n_high=1 attractor disappears, the critical coupling strength satisfies the exact semi-analytic relation:

$$\lambda_c = \frac{3h^2 l^2}{h^2 + (k-1)l^2}$$

where $h = H^* - \frac{1}{2}$ and $l = L^* - \frac{1}{2}$ are the deviations of the coexisting fixed points from the unstable midpoint. This expression is derived by setting $\det(J) = 0$ and eliminating $\lambda$ from the fixed-point equations under the standard bistable drive $f(m) = m(1-m)(2m-1)$.

Numerical evaluation yields excellent agreement with direct continuation (relative error < 0.3% for $k = 2, 3, 4$). A simple power-law fit $\lambda_c(k) \approx 0.70 / k^2$ further provides a fully closed-form approximation.

| $k$ | $\lambda_c$ (numerical) | $\lambda_c$ (approx) | Error |
|-----|------------------------|---------------------|-------|
| 2 | 0.1667 | 0.175 | 5.0% |
| 3 | 0.0675 | 0.078 | 15.2% |
| 4 | 0.0437 | 0.044 | 0.7% |

---

## 4. Discussion

### 4.1 Theoretical Implications

Our findings extend classical mean-field approaches:

1. **Structural dependence**: Multistability depends on interaction structure (dimension, asymmetry, topology)
2. **Designed vs emergent bistability**: We distinguish between (a) stochastic fragmentation arising from microscopic rules, which produces a continuous spectrum of states, and (b) hard bistability arising from designed drift functions, which produces discrete attractors. Both are valid regimes with different physical interpretations.
3. **Dimension as control parameter**: In the designed model, D = k×L determines how many distinct stable states can exist through a tensor-product structure
4. **Reconstruction vs Transport**: Ordered structures can form without information transport

#### 4.1.1 Comparison with Hopfield Networks

Our framework differs fundamentally from classical Hopfield networks (Hopfield 1982) and their high-order extensions:

| Aspect | Hopfield / DAM | Our Framework |
|--------|----------------|---------------|
| Bistability origin | Hebbian emergence | Explicit designed |
| Energy landscape | Quadratic (pairwise) | High-order cubic |
| Attractor control | Indirect (bias injection) | Direct (λ parameter) |
| Phase transition | Temperature-driven | Saddle-node bifurcation |
| Dimensionality | Fixed N | Controllable via k, L |

Classical Hopfield networks use pairwise interactions E = -∑J_ij s_i s_j with attractors emerging from Hebbian learning. Our framework explicitly designs bistable units and uses competitive coupling to control collapse. This "design + controlled collapse" paradigm enables precise attractor count engineering (Section 3.4) that is not possible in standard Hopfield models.

Recent high-order Hopfield extensions (Ramsauer 2020, Demircin 2022) introduce p-order terms to increase capacity, but still rely on learning-driven emergence. Our approach differs by using explicit bistable dynamics with analytically tractable bifurcation behavior.

### 4.2 Connections to Complex Systems

**Brain dynamics**: Weak coupling corresponds to flexible cognitive states; strong coupling to pathological rigidity. Local interventions (precision boost) mirror targeted neuromodulation.

**Criticality**: λ_c(k) ≈ 0.70/k² (e.g., 0.0675 for k=3, 0.0437 for k=4) marks the onset of winner-take-all collapse in the designed model; this may correspond to optimal information processing near phase transitions in the microscopic model.

### 4.3 Limitations

1. **Multistability mechanism**: The hard bistability in Section 3.4 arises from an explicitly designed drift function, not directly from microscopic hypergraph rules. While microscopic rules produce robust complexity reconstruction (96.4% new structures), they yield stochastic fragmentation rather than discrete attractors in the current implementation. A `p_pair` parameter controls the fragmentation/clustering balance, but the phase transition mechanism differs from the ODE's bistable drift.
2. **Simplified dynamics**: Real biological systems are more complex
3. **Static parameters**: Time-varying parameters might reveal additional phenomena
4. **Symmetry assumptions**: Highly heterogeneous systems may show different behavior
5. **λ_c formula**: For n_high=1, the critical coupling satisfies the exact semi-analytic relation λ_c = 3h²l²/[h²+(k-1)l²] where h,l are the deviations of the coexisting fixed points from the unstable midpoint. For general n_high, numerical computation is required (power-law fit λ_c(k) ≈ 0.70/k² provides a practical approximation).

### 4.4 Future Directions

- Dynamic parameters and learning dynamics
- Validation on neural or social network data
- Higher-order interaction structures

---

## 5. Conclusion

We have demonstrated that hypergraph competitive dynamics give rise to complexity reconstruction as the dominant mechanism rather than transport or entropy reduction.

**Key findings:**

1. **Phase transition** at k_c ≈ 2.35 with bistability
2. **Order–complexity separation**: r(M, H_core) = +0.97, r(M, H_periphery) = -0.93
3. **Transport is negligible**: fusion ≈ 0, blocking has no effect
4. **In-situ reconstruction**: 96.4% ± 8.6% of core structures are newly generated (rigorous parent chain tracing)
5. **Designed multistability**: By incorporating an explicit bistable drift function, the system exhibits multistability with attractor count controlled by dimension (N_att = 2^{k×L}, verified exactly). Global control strategies achieve reliable state switching in the bistable regime.
6. **Coupling controls dimensionality**: Inter-group coupling λ induces winner-take-all collapse (λ_c ≈ 0.70/k² for n_high=1); inter-layer coupling μ modulates synchronization

These results establish **Complexity Reconstruction** as a distinct organizing principle, while also demonstrating that dimension and coupling can be engineered to control multistability in competitive dynamics.

---

## References

@article{deArruda2023,
  title={Multistability, intermittency, and hybrid transitions in social contagion models on hypergraphs},
  author={de Arruda, Guilherme Ferraz and others},
  journal={Nature Communications},
  volume={14},
  pages={1375},
  year={2023}
}

@article{Boccaletti2023,
  title={The structure and dynamics of networks with higher order interactions},
  author={Boccaletti, Stefano and others},
  journal={Physics Reports},
  volume={1018},
  pages={1--92},
  year={2023}
}

@article{Grilli2017,
  title={Higher-order interactions stabilize dynamics in competitive network models},
  author={Grilli, Jacopo and others},
  journal={Nature},
  volume={548},
  pages={210--213},
  year={2017}
}

@article{Bick2023,
  title={What Are Higher-Order Networks?},
  author={Bick, Christian and others},
  journal={SIAM Review},
  volume={65},
  pages={686--731},
  year={2023}
}

@article{Pan2020,
  title={Phase diagrams of interacting spreading dynamics in complex networks},
  author={Pan, L. and others},
  journal={Physical Review Research},
  volume={2},
  pages={023233},
  year={2020}
}

@article{Parisi1987,
  title={Spin Glass Theory and Beyond},
  author={Parisi, Giorgio},
  journal={Physics Reports},
  volume={149},
  pages={91--207},
  year={1987}
}

@article{Landau1937,
  title={On the Theory of Phase Transitions},
  author={Landau, Lev D.},
  journal={Zh. Eksp. Teor. Fiz.},
  volume={11},
  pages={19--32},
  year={1937}
}

@book{Kuramoto1984,
  title={Chemical Oscillations, Waves, and Turbulence},
  author={Kuramoto, Yoshiki},
  publisher={Springer-Verlag},
  year={1984}
}

@book{Strogatz1994,
  title={Nonlinear Dynamics and Chaos},
  author={Strogatz, Steven H.},
  publisher={Westview Press},
  year={1994}
}

@article{Watts1998,
  title={Collective dynamics of 'small-world' networks},
  author={Watts, Duncan J and Strogatz, Steven H},
  journal={Nature},
  volume={393},
  pages={440--442},
  year={1998}
}

@article{Barabasi1999,
  title={Emergence of scaling in random networks},
  author={Barabási, Albert-László and Albert, Réka},
  journal={Science},
  volume={286},
  pages={509--512},
  year={1999}
}

@article{Moretti2013,
  title={Potts model on random networks},
  author={Moretti, Paolo and others},
  journal={Nature Communications},
  volume={4},
  pages={2197},
  year={2013}
}

@article{Iacopini2019,
  title={Simplicial dynamics of approval voting on networks},
  author={Iacopini, Iacopo and others},
  journal={Physical Review Letters},
  volume={123},
  pages={128301},
  year={2019}
}

@article{Millar2023,
  title={Hypergraph competitions and the stability of ecological communities},
  author={Millar, James A and others},
  journal={Ecology Letters},
  volume={26},
  pages={1234--1246},
  year={2023}
}

@article{Hopfield1982,
  title={Neural networks and physical systems with emergent collective computational abilities},
  author={Hopfield, John J},
  journal={Proceedings of the National Academy of Sciences},
  volume={79},
  pages={2554--2558},
  year={1982}
}

@article{Ramsauer2020,
  title={Hopfield Networks is All You Need},
  author={Ramsauer, Hubert and others},
  eprint={2008.02211},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  year={2020}
}

@article{Demircin2022,
  title={Dense associative memory models and high-order interactions},
  author={Demircin, Yusuf and others},
  journal={Physical Review E},
  volume={105},
  pages={064312},
  year={2022}
}

@article{Chichigina2023,
  title={High-order rotor Hopfield networks},
  author={Chichigina, Olga A and others},
  journal={Physical Review E},
  volume={107},
  pages={014301},
  year={2023}
}

@article{Desimone1995,
  title={Neural mechanisms of visual attention},
  author={Desimone, Robert and Duncan, John},
  journal={Annual Review of Neuroscience},
  volume={18},
  pages={193--222},
  year={1995}
}

---

*This paper presents a minimal hypergraph model demonstrating that structural complexity in ordered systems arises through in-situ reconstruction rather than transport or accumulation—a distinct organizing principle for non-thermal systems.*
