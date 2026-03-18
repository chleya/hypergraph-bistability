"""
完整相图生成 - 顶会级可视化
===========================
"""
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import os

os.makedirs('F:/hypergraph_bistability/figures/multistable_phase', exist_ok=True)

# ==================== 核心动力学 ====================
def F(M_flat, t, omega, a_list, b_list, c_list, lambda_, mu, L, k):
    M = M_flat.reshape((L, k))
    dM = np.zeros((L, k))
    for l in range(L):
        for i in range(k):
            self_term = omega[i] * (a_list[i] * M[l,i]**3 + b_list[i] * M[l,i]**2 + c_list[i] * M[l,i])
            cross_group = -lambda_ * M[l,i] * np.sum(M[l, np.arange(k) != i])
            cross_layer = mu * np.sum(M[np.arange(L) != l, i])
            dM[l,i] = self_term + cross_group + cross_layer
    return dM.flatten()

def simple_cluster(finals, threshold=0.05):
    unique = []
    for s in finals:
        is_new = True
        for u in unique:
            dist = np.sqrt(np.mean((s - u)**2))
            if dist < threshold:
                is_new = False
                break
        if is_new:
            unique.append(s)
    return len(unique), unique

def count_stable_states(Kc_list, lambda_, mu, L=2, k=3, N_init=100, T=100, threshold=0.05):
    omega = np.ones(k)
    a_list = [-3.0 / kc for kc in Kc_list]
    b_list = [4.5 / kc for kc in Kc_list]
    c_list = [-1.5 / kc for kc in Kc_list]
    
    params = (omega, a_list, b_list, c_list, lambda_, mu, L, k)
    finals = []
    t = np.linspace(0, T, 1000)
    for _ in range(N_init):
        init = np.random.uniform(0.01, 0.99, L * k)
        sol = odeint(F, init, t, args=params, atol=1e-8, rtol=1e-8)
        finals.append(sol[-1])
    finals = np.array(finals)
    return simple_cluster(finals, threshold)

# ==================== 相图1: lambda vs 吸引子数 ====================
print("Generating phase diagram 1: lambda vs attractors...")

Kc_asym = [0.32, 0.40, 0.48]
Kc_sym = [0.4, 0.4, 0.4]

lambdas = np.linspace(0, 1.0, 21)
n_asym = []
n_sym = []

for lam in lambdas:
    n, _ = count_stable_states(Kc_asym, lam, 0.0, N_init=100)
    n_asym.append(n)
    n, _ = count_stable_states(Kc_sym, lam, 0.0, N_init=100)
    n_sym.append(n)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(lambdas, n_asym, 'o-', label='Asymmetric Kc=[0.32,0.40,0.48]', linewidth=2, markersize=6)
ax.plot(lambdas, n_sym, 's--', label='Symmetric Kc=[0.4,0.4,0.4]', linewidth=2, markersize=6)
ax.set_xlabel('λ (Inter-group Competition)', fontsize=12)
ax.set_ylabel('Number of Stable States', fontsize=12)
ax.set_title('Phase Diagram: λ vs Attractor Count\n(k=3, L=2, 6D state space)', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 55)
plt.tight_layout()
plt.savefig('F:/hypergraph_bistability/figures/multistable_phase/lambda_vs_attractors.png', dpi=200)
plt.close()
print("  Saved: lambda_vs_attractors.png")

# ==================== 相图2: 2D heatmap (lambda vs mu) ====================
print("Generating phase diagram 2: 2D heatmap...")

Kc = [0.32, 0.40, 0.48]
lambdas = np.linspace(0, 0.8, 9)
mus = np.linspace(-0.3, 0.3, 7)

phase_matrix = np.zeros((len(lambdas), len(mus)))

for i, lam in enumerate(lambdas):
    for j, mu in enumerate(mus):
        n, _ = count_stable_states(Kc, lam, mu, N_init=80)
        phase_matrix[i, j] = n

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(phase_matrix, origin='lower', aspect='auto',
               extent=[mus[0], mus[-1], lambdas[0], lambdas[-1]],
               cmap='RdYlBu_r', vmin=0, vmax=50)
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Number of Attractors', fontsize=12)
ax.set_xlabel('μ (Cross-layer Coupling)', fontsize=12)
ax.set_ylabel('λ (Inter-group Competition)', fontsize=12)
ax.set_title('2D Phase Diagram: Attractors vs λ and μ\n(Kc asymmetric)', fontsize=14)
plt.tight_layout()
plt.savefig('F:/hypergraph_bistability/figures/multistable_phase/phase_2d_heatmap.png', dpi=200)
plt.close()
print("  Saved: phase_2d_heatmap.png")

# ==================== 相图3: Kc非对称程度 ====================
print("Generating phase diagram 3: Kc asymmetry...")

base = 0.4
asym_degrees = [0, 0.05, 0.10, 0.15, 0.20]
n_by_asym = []

for deg in asym_degrees:
    Kc = [base - deg, base, base + deg]
    n, _ = count_stable_states(Kc, 0.0, 0.0, N_init=100)
    n_by_asym.append(n)

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(range(len(asym_degrees)), n_by_asym, color='steelblue', alpha=0.7)
ax.set_xticks(range(len(asym_degrees)))
ax.set_xticklabels([f'ΔKc={d:.2f}' for d in asym_degrees])
ax.set_xlabel('Kc Asymmetry (ΔKc)', fontsize=12)
ax.set_ylabel('Number of Stable States', fontsize=12)
ax.set_title('Effect of Kc Asymmetry on Multistability\n(λ=0, weak coupling)', fontsize=14)
ax.set_ylim(0, 60)
plt.tight_layout()
plt.savefig('F:/hypergraph_bistability/figures/multistable_phase/kc_asymmetry.png', dpi=200)
plt.close()
print("  Saved: kc_asymmetry.png")

# ==================== 相图4: 维度效应 ====================
print("Generating phase diagram 4: dimension effect...")

dims = [(2,2), (2,3), (2,4), (3,3)]  # (L, k)
n_by_dim = []

for L, k in dims:
    Kc = [0.4] * k
    n, _ = count_stable_states(Kc, 0.0, 0.0, L=L, k=k, N_init=80)
    n_by_dim.append(n)
    print(f"  L={L}, k={k}: {n} attractors")

fig, ax = plt.subplots(figsize=(8, 5))
labels = [f'L={L},k={k}\n({L*k}D)' for L, k in dims]
bars = ax.bar(range(len(dims)), n_by_dim, color='coral', alpha=0.7)
ax.set_xticks(range(len(dims)))
ax.set_xticklabels(labels)
ax.set_xlabel('State Space Dimension (L × k)', fontsize=12)
ax.set_ylabel('Number of Stable States', fontsize=12)
ax.set_title('Effect of State Space Dimension on Multistability\n(λ=0, symmetric Kc)', fontsize=14)
plt.tight_layout()
plt.savefig('F:/hypergraph_bistability/figures/multistable_phase/dimension_effect.png', dpi=200)
plt.close()
print("  Saved: dimension_effect.png")

print("\n" + "=" * 60)
print("All phase diagrams generated!")
print("=" * 60)
