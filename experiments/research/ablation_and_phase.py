"""
Kc差异 vs Lambda 相图 + 消融实验
=================================
"""
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import os

os.makedirs('F:/hypergraph_bistability/figures/multistable_phase', exist_ok=True)

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

def count_states(Kc_list, lambda_, mu, L=2, k=3, N_init=60, T=80):
    omega = np.ones(k)
    a_list = [-3.0 / kc for kc in Kc_list]
    b_list = [4.5 / kc for kc in Kc_list]
    c_list = [-1.5 / kc for kc in Kc_list]
    
    params = (omega, a_list, b_list, c_list, lambda_, mu, L, k)
    finals = []
    t = np.linspace(0, T, 800)
    for _ in range(N_init):
        init = np.random.uniform(0.01, 0.99, L * k)
        sol = odeint(F, init, t, args=params, atol=1e-6, rtol=1e-6)
        finals.append(sol[-1])
    finals = np.array(finals)
    return simple_cluster(finals, 0.05)

# ==================== 实验1: Kc差异 vs Lambda ====================
print("=" * 60)
print("Experiment 1: Kc Delta vs Lambda")
print("=" * 60)

lambdas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
delta_Kcs = [0.0, 0.05, 0.10, 0.15, 0.20]
base_Kc = 0.40

matrix = np.zeros((len(delta_Kcs), len(lambdas)))

for i, dK in enumerate(delta_Kcs):
    Kc_list = [base_Kc - dK, base_Kc, base_Kc + dK]
    for j, lam in enumerate(lambdas):
        n, _ = count_states(Kc_list, lam, 0.0, N_init=50)
        matrix[i, j] = n

# 绘图
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(matrix, origin='lower', aspect='auto',
               extent=[lambdas[0], lambdas[-1], delta_Kcs[0], delta_Kcs[-1]],
               cmap='RdYlBu_r', vmin=0, vmax=50)
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Number of Attractors', fontsize=12)
ax.set_xlabel('λ (Inter-group Competition)', fontsize=12)
ax.set_ylabel('ΔKc (Kc Asymmetry)', fontsize=12)
ax.set_title('Phase Diagram: ΔKc vs λ\n(Multistability Region)', fontsize=14)

# 添加数值标注
for i, dK in enumerate(delta_Kcs):
    for j, lam in enumerate(lambdas):
        text = ax.text(lam, dK, int(matrix[i,j]), ha="center", va="center", 
                      color="white" if matrix[i,j] > 25 else "black", fontsize=8)

plt.tight_layout()
plt.savefig('F:/hypergraph_bistability/figures/multistable_phase/fig4_KcDelta_vs_lambda.png', dpi=150)
plt.close()
print("  Saved: fig4_KcDelta_vs_lambda.png")

# ==================== 实验2: 消融实验 ====================
print("\n" + "=" * 60)
print("Experiment 2: Ablation Studies")
print("=" * 60)

# 2A: 仅多群体 (L=1, k=3)
print("\n2A: Multi-group only (L=1, k=3):")
n_group_only = []
for lam in lambdas:
    Kc = [0.32, 0.40, 0.48]
    n, _ = count_states(Kc, lam, 0.0, L=1, k=3, N_init=50)
    n_group_only.append(n)
    print(f"  lam={lam:.1f}: {n}")

# 2B: 仅多层 (L=2, k=1)  
print("\n2B: Multi-layer only (L=2, k=1):")
n_layer_only = []
for lam in lambdas:
    Kc = [0.40]
    n, _ = count_states(Kc, lam, 0.0, L=2, k=1, N_init=50)
    n_layer_only.append(n)
    print(f"  lam={lam:.1f}: {n}")

# 2C: 完整模型 (L=2, k=3)
print("\n2C: Full model (L=2, k=3):")
n_full = []
for lam in lambdas:
    Kc = [0.32, 0.40, 0.48]
    n, _ = count_states(Kc, lam, 0.0, L=2, k=3, N_init=50)
    n_full.append(n)
    print(f"  lam={lam:.1f}: {n}")

# 绘图：消融对比
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(lambdas, n_group_only, 'o-', label='Multi-group only (L=1, k=3)', linewidth=2, markersize=6)
ax.plot(lambdas, n_layer_only, 's-', label='Multi-layer only (L=2, k=1)', linewidth=2, markersize=6)
ax.plot(lambdas, n_full, '^-', label='Full (L=2, k=3)', linewidth=2, markersize=6)
ax.set_xlabel('λ (Inter-group Competition)', fontsize=12)
ax.set_ylabel('Number of Attractors', fontsize=12)
ax.set_title('Ablation Study: Multi-group vs Multi-layer vs Full', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 55)
plt.tight_layout()
plt.savefig('F:/hypergraph_bistability/figures/multistable_phase/fig5_ablation.png', dpi=150)
plt.close()
print("  Saved: fig5_ablation.png")

# ==================== 实验3: 维度缩放 ====================
print("\n" + "=" * 60)
print("Experiment 3: Dimension Scaling")
print("=" * 60)

configs = [
    (1, 1, '1D'),
    (1, 2, '2D'),
    (1, 3, '3D'),
    (2, 2, '4D'),
    (2, 3, '6D'),
    (3, 3, '9D'),
]

dim_results = []
for L, k, label in configs:
    Kc = [0.40] * k
    n, _ = count_states(Kc, 0.0, 0.0, L=L, k=k, N_init=50)
    dim_results.append((label, n))
    print(f"  {label} (L={L},k={k}): {n}")

# 绘图
fig, ax = plt.subplots(figsize=(8, 5))
labels = [d[0] for d in dim_results]
values = [d[1] for d in dim_results]
bars = ax.bar(range(len(labels)), values, color='steelblue', alpha=0.7)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels)
ax.set_xlabel('State Space Dimension', fontsize=12)
ax.set_ylabel('Number of Attractors', fontsize=12)
ax.set_title('Dimension Scaling: Attractors vs Dimension\n(λ=0, symmetric Kc)', fontsize=14)
ax.set_ylim(0, 60)
for i, v in enumerate(values):
    ax.text(i, v + 1, str(v), ha='center', fontsize=10)
plt.tight_layout()
plt.savefig('F:/hypergraph_bistability/figures/multistable_phase/fig6_dim_scaling.png', dpi=150)
plt.close()
print("  Saved: fig6_dim_scaling.png")

print("\n" + "=" * 60)
print("All experiments completed!")
print("=" * 60)
