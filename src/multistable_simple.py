"""
超图多稳态相图 - 简化版
=========================
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from collections import defaultdict
import os

os.makedirs('F:/hypergraph_bistability/figures/multistable_phase', exist_ok=True)


def hypergraph_dynamics(M, t, k, L, Kc_list, lam, mu):
    """超图动力学"""
    dMdt = np.zeros(k * L)
    M = np.array(M).reshape(k, L)
    
    for i in range(k):
        for l in range(L):
            Mc = M[i, l]
            Kc_i = Kc_list[i]
            
            # 自身非线性项
            term1 = Mc * (1 - Mc / Kc_i)
            
            # 群体间竞争
            competition = 0
            for j in range(k):
                if j != i:
                    Mj_mean = np.mean(M[j, :])
                    competition += lam * (Mj_mean - Mc)
            
            # 层间耦合
            layer_coup = 0
            for l2 in range(L):
                if l2 != l:
                    layer_coup += mu * (M[i, l2] - Mc)
            
            dMdt[i * L + l] = term1 + competition + layer_coup
    
    return dMdt


def simulate_one(M0, t, k, L, Kc_list, lam, mu):
    """单次模拟"""
    M0 = np.array(M0).flatten()
    sol = odeint(hypergraph_dynamics, M0, t, args=(k, L, Kc_list, lam, mu))
    return sol[-1]


def cluster_states(finals, threshold=0.15):
    """聚类"""
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
    return unique


def count_attractors(k, L, Kc_list, lam, mu, n_init=100, threshold=0.15):
    """计算吸引子数量"""
    t = np.linspace(0, 20, 200)
    
    finals = []
    for _ in range(n_init):
        M0 = np.random.uniform(0.1, 0.9, k * L)
        try:
            final = simulate_one(M0, t, k, L, Kc_list, lam, mu)
            finals.append(final)
        except:
            pass
    
    unique = cluster_states(finals, threshold)
    return len(unique), unique, finals


# 实验
print("=" * 60)
print("Multistable Phase Diagram")
print("=" * 60)

k, L = 3, 2
n_init = 100

# 对称 vs 非对称
print("\n1. Symmetry test:")

Kc_sym = [0.4, 0.4, 0.4]
n1, _, _ = count_attractors(k, L, Kc_sym, 0.5, 0.0, n_init)
print(f"   Symmetric Kc={Kc_sym}: {n1} attractors")

Kc_asym = [0.32, 0.40, 0.48]
n2, _, _ = count_attractors(k, L, Kc_asym, 0.5, 0.0, n_init)
print(f"   Asymmetric Kc={Kc_asym}: {n2} attractors")

Kc_strong = [0.20, 0.50, 0.80]
n3, _, _ = count_attractors(k, L, Kc_strong, 0.5, 0.0, n_init)
print(f"   Strong Kc={Kc_strong}: {n3} attractors")

# Lambda 扫描
print("\n2. Lambda scan:")
lam_results = []
for lam in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    n, _, _ = count_attractors(k, L, Kc_asym, lam, 0.0, n_init)
    lam_results.append((lam, n))
    print(f"   lam={lam:.1f}: {n} attractors")

# 相图
print("\n3. Phase diagram:")
lam_vals = np.linspace(0, 0.8, 5)
delta_vals = np.linspace(0, 0.3, 4)

phase = np.zeros((len(delta_vals), len(lam_vals)))

for i, dK in enumerate(delta_vals):
    for j, lam in enumerate(lam_vals):
        Kc = [0.4 - dK, 0.4, 0.4 + dK]
        n, _, _ = count_attractors(k, L, Kc, lam, 0.0, n_init)
        phase[i, j] = n

# 绘图
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(phase, origin='lower', aspect='auto',
               extent=[lam_vals[0], lam_vals[-1], delta_vals[0], delta_vals[-1]],
               cmap='RdYlGn_r', vmin=0, vmax=10)
plt.colorbar(im, ax=ax, label='Attractors')
ax.set_xlabel('lambda')
ax.set_ylabel('delta_Kc')
ax.set_title('Phase: Attractors vs lambda and delta_Kc')

plt.tight_layout()
plt.savefig('F:/hypergraph_bistability/figures/multistable_phase/phase_test.png', dpi=150)
plt.close()

print("\nDone! Figure saved.")
