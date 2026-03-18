# Introduction

## Background and Motivation

Complex systems composed of multiple interacting entities frequently exhibit **multistability** — the coexistence of multiple stable attractor states. This phenomenon is fundamental to understanding decision-making in social networks, pattern formation in biological systems, and state transitions in neural networks. Recently, hypergraphs have emerged as a powerful framework for modeling higher-order interactions, where edges can connect more than two nodes simultaneously. This higher-order structure introduces richer dynamical behaviors than traditional graph-based models.

Our previous work established a **capacity-constrained competitive dynamics** framework on hypergraphs, demonstrating that a simple one-dimensional order parameter $M$ exhibits **bistability** with two stable fixed points at $M_1^* \approx 0.45$ and $M_2^* \approx 1.0$. This bistability arises from the competition between fusion (grouping) and split (fragmentation) processes, controlled by a critical capacity parameter $K_c \approx 2.35$. The system exhibits a phase transition at $K_c/N \approx 0.35$, where the maximum faction size jumps from $\sim 54\%$ to near $100\%$, indicating a symmetry-breaking transition.

However, real-world systems often involve **multiple competing groups** (e.g., different social communities) and **multiple layers** (e.g., different interaction modes). This raises a fundamental question: **How does the 1D bistable structure extend to multi-dimensional multistability when we introduce group and layer structures?**

## Problem Statement

The 1D mean-field theory naturally supports at most **two stable states** — a fundamental limitation arising from the topology of the order parameter space. To achieve true multistability (3+ stable states), we need to either:

1. **Increase dimensionality** — Add independent order parameters for each group/layer
2. **Introduce asymmetry** — Different groups with different capacity constraints
3. **Add coupling** — Inter-group and inter-layer interactions

This extension is not merely quantitative; it fundamentally changes the system's behavior. The competition dynamics become a **high-dimensional stochastic system** where the number of attractors, their stability, and the noise sensitivity all depend on the coupling structure.

## Key Discoveries

Our investigation reveals several unexpected findings:

### 1. Dimensionality Drives Multistability
When extending to $D$ independent groups, the system naturally supports $D+1$ stable states. At weak coupling ($\lambda \approx 0$), we observe $10$-$40$ attractors depending on system size. This follows a simple principle: **dimension = stable states - 1**.

### 2. Coupling Induces Winner-Take-All Collapse
As inter-group coupling $\lambda$ increases, the system undergoes a **phase transition** from multistability to unistability. At $\lambda \approx 0.4$, most systems collapse to just $1$-$4$ attractors. This "winner-take-all" dynamics is driven by competition overwhelming the capacity constraints of individual groups.

### 3. Noise Sensitivity is Coupling-Dependent
Counter-intuitively, **noise has minimal effect at weak coupling** (deep basins are stable) and **maximal effect at critical coupling** ($\lambda \approx 0.3$). This is opposite to classical Kramer's theory where noise typically enhances transitions. Here, **coupling itself creates the metastable landscape**, and noise merely perturbs it.

### 4. Topology Modulates Collapse
Different network topologies show dramatically different behaviors:
- **Power-law (heterogeneous)**: Delays collapse, maintains more attractors at high coupling
- **High-overlap**: Accelerates collapse, fewer stable states
- **Uniform**: Intermediate behavior

This reveals a **heterogeneity-protection mechanism**: degree heterogeneity acts as a buffer against coupling-induced collapse.

## Contributions

This paper makes three core contributions:

1. **Theoretical Framework**: We extend the 1D capacity-constrained dynamics to multi-group, multi-layer structures, establishing a mean-field theory that predicts $D+1$ stable states from $D$ dimensions.

2. **Mechanism Discovery**: We identify **coupling-induced collapse** as the dominant mechanism, where inter-group competition drives the system from multistability to unistability. This is fundamentally different from noise-driven transitions in traditional bistable systems.

3. **Topology Dependence**: We demonstrate that network heterogeneity (power-law degree distribution) delays coupling-induced collapse, while hyperedge overlap accelerates it — with implications for real-world network design.

## Related Work

Previous work on multistability in hypergraphs has primarily focused on **epidemic dynamics** (SIS-like models on hypergraphs), where the spreading rate $\beta$ and recovery rate $\gamma$ create multiple stable epidemic states. Notable works include:

- Ferraz de Arruda et al. (2023, Nat Commun): Social contagion multistability on hypergraphs
- Malizia et al. (2026): Nested hyperedges quench bistability  
- Burgio et al. (2023): Adaptive hypergraphs and bistability
- Adhikari et al. (2022/2023): Synchronization bistability on hypergraphs

These works differ fundamentally from our approach: they study **transmission dynamics** (how information/disease spreads), while we study **competition dynamics** (how entities compete for resources/attention). Our mechanism — capacity-constrained competition with group-specific thresholds — has no direct analogue in the contagion literature.

Furthermore, most previous work studies noise as a **perturbation** that enables transitions. We discover the opposite: **coupling creates the structure, noise merely modulates it**. This "coupling-induced sensitivity" represents a new paradigm in multistability research.

## Paper Structure

The paper is organized as follows: Section 2 introduces the model and mean-field reduction. Section 3 presents deterministic multistability results. Section 4 analyzes noise robustness and critical sensitivity. Section 5 discusses topology dependence. Section 6 concludes with discussion and future directions.
