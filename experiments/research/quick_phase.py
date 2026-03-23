"""
快速相图生成
============
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

print("Generating phase diagrams...")

# 1. Lambda vs Attractors
Kc_asym = [0.32, 0.40, 0.48]
Kc_sym = [0.4, 0.4, 0.4]
lambdas = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]

n_asym = [count_states(Kc_asym, lam, 0.0)[0] for lam in lambdas]
n_sym = [count_states(Kc_sym, lam, 0.0)[0] for lam in lambdas]

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(lambdas, n_asym, 'o-', label='Asymmetric', linewidth=2, markersize=8)
ax.plot(lambdas, n_sym, 's--', label='Symmetric', linewidth=2, markersize=8)
ax.set_xlabel('Lambda (coupling)', fontsize=12)
ax.set_ylabel('Attractors', fontsize=12)
ax.set_title('Multistability: Lambda vs Attractor Count')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('F:/hypergraph_bistability/figures/multistable_phase/fig1_lambda.png', dpi=150)
plt.close()
print("  fig1_lambda.png saved")

# 2. 2D Phase (lambda vs mu) - smaller grid
print("  Generating 2D phase...")
lambdas = [0.0, 0.2, 0.4, 0.6, 0.8]
mus = [-0.2, 0.0, 0.2]
matrix = np.zeros((len(lambdas), len(mus)))

for i, lam in enumerate(lambdas):
    for j, mu in enumerate(mus):
        n, _ = count_states(Kc_asym, lam, mu, N_init=50)
        matrix[i, j] = n

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(matrix, origin='lower', aspect='auto',
               extent=[mus[0], mus[-1], lambdas[0], lambdas[-1]],
               cmap='RdYlBu_r', vmin=0, vmax=50)
plt.colorbar(im, ax=ax, label='Attractors')
ax.set_xlabel('Mu (cross-layer)')
ax.set_ylabel('Lambda')
ax.set_title('Phase Diagram')
plt.tight_layout()
plt.savefig('F:/hypergraph_bistability/figures/multistable_phase/fig2_phase.png', dpi=150)
plt.close()
print("  fig2_phase.png saved")

# 3. Dimension effect
print("  Generating dimension effect...")
dims = [(2,2), (2,3), (3,3)]
n_dims = []
for L, k in dims:
    n, _ = count_states([0.4]*k, 0.0, 0.0, L=L, k=k, N_init=50)
    n_dims.append(n)

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(range(len(dims)), n_dims, color='steelblue')
ax.set_xticks(range(len(dims)))
ax.set_xticklabels([f'L={L},k={k}' for L,k in dims])
ax.set_ylabel('Attractors')
ax.set_title('Dimension Effect')
plt.tight_layout()
plt.savefig('F:/hypergraph_bistability/figures/multistable_phase/fig3_dim.png', dpi=150)
plt.close()
print("  fig3_dim.png saved")

print("\nDone! All figures saved.")
