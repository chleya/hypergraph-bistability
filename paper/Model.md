# Model

## 2.1 Original Bistable Model on Single-Layer Hypergraphs

The foundation of our work is the capacity-constrained competitive dynamics on hypergraphs introduced in the baseline project. Consider a uniform hypergraph with node set V (|V| = N) and hyperedge set E. Each node v ∈ V belongs to one of two competing factions (or "camps") with state m_v(t) ∈ [0,1], where m_v ≈ 1 indicates strong affiliation to camp 1.

The macroscopic order parameter is defined as 
$$M(t) = |C_{max}(t)| / |E|,$$ 
where $C_{max}(t)$ is the largest faction occupying hyperedges at time t.

Under mean-field approximation (valid for large, dense hypergraphs), the dynamics reduce to a one-dimensional stochastic differential equation: 
$$dM/dt = F(M) + \eta(t),$$ 
where $\eta(t)$ is Gaussian white noise with $\langle\eta(t)\rangle = 0$, $\langle\eta(t)\eta(t')\rangle = 2D \delta(t-t')$, and the drift function takes a cubic form: 
$$F(M) = \omega (a M^3 + b M^2 + c M),$$ 
with coefficients a, b, c tuned by the capacity constraint $K_c$ (effective resource per hyperedge). For $K_c/N \approx 0.35$, the potential $V(M) = -\int F(M) dM$ yields an asymmetric double-well landscape with attractors at 
$$M_1^* \approx 0.45 \text{ (deep, robust well)},$$ 
$$M_2^* \approx 1.0 \text{ (shallow, metastable well)},$$ 
and an unstable saddle at $M_0 \approx 0.6$ (basin boundary). 

This bistability arises from the competition for limited hyperedge capacity: nodes "pay" a cost to occupy edges, leading to spontaneous symmetry breaking when capacity is intermediate.

## 2.2 Extension to Multistability: Multi-Group and Multi-Layer Order Parameters

To break the "at most two attractors" limitation of 1D systems, we introduce multi-group and multi-layer structures, naturally present in real hypergraphs (e.g., users belonging to multiple interest groups; communication networks with core-access layers).

- **Multi-group competition** ($k \geq 3$ groups): Nodes are partitioned into $k$ groups, each with group-specific capacity constraint $\hat{K}_{c,i}$ ($i=1,...,k$). The order parameter becomes a vector $\mathbf{M} = (M_1, ..., M_k)$, where $M_i$ is the average affiliation strength of group $i$.

- **Multi-layer hypergraphs** ($L \geq 2$ layers): The hypergraph is decomposed into $L$ layers, with cross-layer coupling. The full order parameter is $M_{l,i}$ ($l=1,...,L$; $i=1,...,k$), dimension $D = kL$.

The drift function generalizes to: 
$$F(M_{l,i}) = \omega_i (a_i M_{l,i}^3 + b_i M_{l,i}^2 + c_i M_{l,i}) - \lambda \sum_{j \neq i} M_{l,i} M_{l,j} + \mu \sum_{l' \neq l} M_{l',i},$$ 

where:
- $a_i, b_i, c_i$ depend on $\hat{K}_{c,i}$ (e.g., $a_i = -3/\hat{K}_{c,i}$, etc., reproducing bistability per group when $\lambda=\mu=0$),
- $-\lambda$ term: inter-group suppression (winner-take-all tendency),
- $\mu$ term: cross-layer interaction ($\mu > 0$ promotes synchronization, $\mu < 0$ inhibits).

At $\lambda \approx 0$ and $\mu \approx 0$, the system decouples into $kL$ independent bistable subsystems → up to $2^{kL}$ attractors in principle (though sampling yields 8–40 due to basin overlap and noise).

Increasing $\lambda$ drives global competition, collapsing the high-dimensional landscape toward fewer deep wells (winner-take-all).

## 2.3 Microscopic Realization: Group-Mean Driven Dynamics

To go beyond mean-field, we implement a microscopic node-level dynamics that reproduces the collective basins while respecting hypergraph topology.

Each node $v$ has state $m_v(t) \in [0,1]$. Let $\text{group\_assign}(v) = i$, $\text{layer\_assign}(v) = l$. The evolution follows: 
$$\frac{dm_v}{dt} = F(M_{l,i}(t)) + \text{local\_adjustment} + \eta_v(t),$$ 

where $M_{l,i}(t)$ is the instantaneous group-layer mean: 
$$M_{l,i}(t) = \frac{1}{|G_{l,i}|} \sum_{u \in G_{l,i}} m_u(t),$$ 
$G_{l,i} = \{u \mid \text{group\_assign}(u)=i, \text{layer\_assign}(u)=l\}$.

The local\_adjustment term (small coefficient, e.g. 0.05) incorporates hyperedge-level synchronization: 
$$\text{local\_adjustment} = \alpha (\langle m \rangle_{\text{incident edges}} - m_v),$$ 

ensuring nodes in the same hyperedge tend to align weakly, while the dominant drive remains the group mean $F(M_{l,i})$.

This "Group-Mean Driven" scheme guarantees that microscopic trajectories converge to the same attractors as the mean-field vector $\mathbf{M}$, but topology modulates basin volumes (e.g., power-law heterogeneity buffers collapse).

## 2.4 Stochastic Formulation with Noise

For robustness analysis, we add additive white noise to the microscopic level: 
$$\frac{dm_v}{dt} = \text{deterministic drive} + \sigma \eta_v(t),$$ 
with $\eta_v(t)$ independent Gaussian processes.

Long trajectories are simulated using solve_ivp (RK45 method), and attractors are identified via clustering of sampled states (threshold 0.05 in Euclidean distance on normalized states).

(Figure X: Schematic of mean-field reduction and microscopic embedding.)
