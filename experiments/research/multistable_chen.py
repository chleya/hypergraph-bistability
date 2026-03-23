"""
超图多稳态相图 - 陈雷阳版本
=============================
"""
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import os

os.makedirs('F:/hypergraph_bistability/figures/multistable_phase', exist_ok=True)

# ==================== 1. 核心动力学 ====================
def F(M_flat, t, omega, a_list, b_list, c_list, lambda_, mu, L, k):
    M = M_flat.reshape((L, k))
    dM = np.zeros((L, k))
    for l in range(L):
        for i in range(k):
            # 自交互（三次项，容量约束决定双稳）
            self_term = omega[i] * (a_list[i] * M[l,i]**3 + b_list[i] * M[l,i]**2 + c_list[i] * M[l,i])
            # 跨群体竞争（抑制项）
            cross_group = -lambda_ * M[l,i] * np.sum(M[l, np.arange(k) != i])
            # 跨层交互
            cross_layer = mu * np.sum(M[np.arange(L) != l, i])
            dM[l,i] = self_term + cross_group + cross_layer
    return dM.flatten()

# ==================== 2. 聚类 ====================
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

def count_stable_states(Kc_list, lambda_, mu, L=2, k=3, N_init=80, T=100, threshold=0.05):
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

# ==================== 3. 测试 ====================
print("=" * 60)
print("Testing Chen's version")
print("=" * 60)

# 非对称 Kc
Kc_list = [0.32, 0.40, 0.48]

# Lambda 扫描
print("\nLambda scan (Kc asymmetric):")
results = []
for lam in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]:
    n, _ = count_stable_states(Kc_list, lam, 0.0, L=2, k=3, N_init=80)
    results.append((lam, n))
    print(f"  lambda={lam:.1f}: {n} attractors")

# 对称 Kc
Kc_sym = [0.4, 0.4, 0.4]
print("\nLambda scan (Kc symmetric):")
for lam in [0.0, 0.1, 0.3, 0.5]:
    n, _ = count_stable_states(Kc_sym, lam, 0.0, L=2, k=3, N_init=80)
    print(f"  lambda={lam:.1f}: {n} attractors")

print("\nDone!")
