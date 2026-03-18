# Results

## 3 Deterministic Multistability: Dimensionality, Coupling, and Phase Transitions

In this section, we present the core deterministic results of our extended model. We demonstrate that the multi-group, multi-layer extension naturally generates multistability, and that the inter-group coupling $\lambda$ serves as the primary control parameter governing the number of stable attractors.

### 3.1 Dimensionality Drives Attractor Count

We first investigate the baseline case with zero inter-group coupling ($\lambda = 0$) and zero inter-layer coupling ($\mu = 0$). Under these conditions, each group-layer pair acts as an independent bistable subsystem with its own capacity constraint $\hat{K}_{c,i}$. With $k=3$ groups and $L=2$ layers, we have $D = kL = 6$ independent dimensions, theoretically supporting up to $2^6 = 64$ attractors.

**Experimental setup.** We use $N=80$ nodes, $k=3$ groups, $L=2$ layers, with capacity constraints $\hat{K}_c = [0.32, 0.40, 0.48]$. The system is initialized with 1,000 random initial conditions uniformly distributed in $[0,1]^N$. Trajectories are integrated for $T=500$ steps, and final states are clustered using Euclidean distance with threshold $\epsilon = 0.2$.

**Results.** Figure 1 shows the distribution of final states in the reduced order parameter space. We find **35–45 distinct attractors** at $\lambda = 0$, confirming that the high-dimensional system naturally generates multistability. Each attractor corresponds to a unique configuration of group-layer affiliations. Notably, the attractors are not uniformly distributed; they cluster into several meta-stable regions corresponding to different "dominance patterns" (e.g., group 1 dominant across both layers, groups 1 and 2 jointly dominant, etc.).

This result establishes the first key principle: **dimension = stable states - 1** (approximately, due to basin overlaps).

### 3.2 Coupling-Induced Winner-Take-All Collapse

We now introduce inter-group coupling $\lambda$ while maintaining $\mu = 0$. The coupling term $-\lambda M_i M_j$ penalizes simultaneous activation of competing groups, favoring configurations where one or few groups dominate.

**Experimental setup.** We scan $\lambda$ in the range $[0, 0.9]$ with step $0.1$, keeping all other parameters fixed. For each $\lambda$, we run 50 random initial conditions and count distinct attractors using the same clustering method.

**Results.** Figure 2 and Table 1 present the main finding.

| $\lambda$ | # Attractors | Dominant Pattern |
|-----------|---------------|------------------|
| 0.0 | 35–47 | Multiple co-existing |
| 0.1 | 30–40 | Slight preference for dominance |
| 0.2 | 20–30 | Clear winner emerges |
| 0.3 | 12–20 | Winner-take-all begins |
| 0.4 | 4–8 | Strong dominance |
| 0.5 | 2–4 | Near-complete collapse |
| 0.7–0.9 | 1–2 | Winner-take-all |

**Key observation.** As $\lambda$ increases, the number of stable attractors monotonically decreases. At $\lambda \approx 0.4$, the system undergoes a **phase transition** from multistability to near-unistability. This is the **winner-take-all** regime where competition forces the system into a state where one group dominates.

The critical coupling $\lambda_c \approx 0.3$ marks the onset of this collapse. Below $\lambda_c$, the system retains multiple stable configurations; above $\lambda_c$, competition overwhelms the capacity constraints and forces convergence to few states.

### 3.3 Phase Diagram: $\lambda$ vs. $\mu$

To fully characterize the deterministic dynamics, we construct a two-dimensional phase diagram scanning both inter-group coupling $\lambda$ and inter-layer coupling $\mu$.

**Results.** Figure 3 shows the $\lambda$-$\mu$ phase diagram. We observe:

1. **$\lambda$ dominates**: The number of attractors is primarily controlled by $\lambda$; $\mu$ has secondary effects.
2. **Positive $\mu$ promotes synchronization**: When $\mu > 0$, layers tend to align, reducing the effective dimensionality.
3. **Negative $\mu$ inhibits synchronization**: When $\mu < 0$, layers compete, slightly increasing attractor count.

The phase diagram reveals that $\lambda$ is the dominant control parameter for multistability, while $\mu$ serves as a fine-tuning knob.

### 3.4 Theoretical Interpretation: Potential Landscape

The observed behavior can be understood through the lens of potential landscape theory. At $\lambda = 0$, each group-layer has its own bistable potential $V_i(M_i)$, creating a high-dimensional potential with multiple wells. As $\lambda$ increases, the cross-term $-\lambda M_i M_j$ introduces negative curvature that flattens secondary wells, causing them to merge into the dominant well.

The effective potential for the total order $M = \sum_i M_i$ can be approximated as:
$$V_{eff}(M) \approx \sum_i V_i(M_i) - \lambda \sum_{i \neq j} M_i M_j$$

As $\lambda$ grows, the second term dominates, transforming the multi-well landscape into a single deep well — the mathematical signature of winner-take-all collapse.

### 3.5 Summary

We have demonstrated that:

1. **High dimensionality naturally produces multistability**: With $k$ groups and $L$ layers, the system supports $O(2^{kL})$ attractors in principle, with 35–47 observed in practice.
2. **Inter-group coupling $\lambda$ controls attractor count**: Weak coupling preserves multistability; strong coupling induces winner-take-all collapse.
3. **Critical coupling $\lambda_c \approx 0.3$** marks the phase transition between multistable and unistable regimes.
4. **Inter-layer coupling $\mu$ has secondary effects**: Positive $\mu$ promotes synchronization; negative $\mu$ slightly increases attractor count.

These results establish the deterministic backbone of our theory. In the next section, we investigate how noise modulates this picture.
