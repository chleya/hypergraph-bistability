# Results

## Section 1: Phase Transition and Bistability

We begin by characterizing the asymptotic behavior of our hypergraph system as a function of the capacity constraint parameter k. We initialize the system with random hyperedges and evolve for 80 steps under the four local rules (growth, fusion, split, deletion) with a fixed capacity limit K = kN.

Figure 1 shows the order parameter M = |C_max|/N (the size of the largest connected component normalized by system size) as a function of k. The results reveal a sharp phase transition at k_c ≈ 2.35. For k < k_c, the system evolves to a fragmented state with M ≈ 0.2–0.3, where no single large cluster dominates. For k > k_c, the system transitions to a clustered state where M approaches 0.9–1.0, indicating the emergence of a dominant structural core.

Critically, the phase transition exhibits **bistability** in the transition region (2.3 ≤ k ≤ 2.6). Depending on initial conditions, the system can settle into either the fragmented or clustered attractor, suggesting the presence of a double-well potential in the asymptotic dynamics. This bistability is a signature of first-order phase transitions and indicates that the system's asymptotic state is not uniquely determined by k, but also depends on the basin of attraction.

These results establish that the minimal hypergraph dynamics with capacity constraints naturally produces structured order—ruling out the trivial explanation that the system remains forever disordered.

---

## Section 2: Order–Complexity Separation

Having established that the system forms structured cores, we now characterize the internal organization of these cores. We decompose the system's structural diversity using a regional analysis: we measure the Shannon entropy H of hyperedge size distributions within the core (nodes belonging to the largest cluster) and periphery (all other nodes).

Figure 3 shows the temporal evolution of H over 100 time steps. As the system evolves, we observe a systematic divergence: **H_core increases** while **H_periphery decreases**. The core becomes internally more diverse, while the peripheral region simplifies.

Figure 4 presents the M–H phase diagram, revealing a striking **order–complexity separation**. Plotting M against H_core and H_periphery across multiple runs, we find:
- **r(M, H_core) = +0.97** (strong positive correlation)
- **r(M, H_periphery) = -0.93** (strong negative correlation)

This implies that as the system becomes more ordered (higher M), complexity simultaneously increases in the core but decreases in the periphery—a counterintuitive phenomenon that cannot be explained by simple entropy minimization.

The phase diagram also reveals the evolutionary trajectory: starting from low-M, high-H_periphery states, the system follows a characteristic path toward high-M, high-H_core, low-H_periphery states. This trajectory represents a systematic reorganization in which complexity is not eliminated but redistributed.

**This phenomenon demands explanation.** The strong anti-correlation between core and peripheral complexity suggests a mechanism of complexity redistribution—but the question remains: is this redistribution driven by transport (complexity flowing from periphery to core), or by an alternative mechanism?
