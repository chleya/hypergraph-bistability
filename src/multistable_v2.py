"""
超图多稳态 - 激进版
===================

更强的非线性竞争，产生真正的多稳态
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import os

os.makedirs('F:/hypergraph_bistability/figures/multistable_phase', exist_ok=True)


def hypergraph_dynamics_v2(M, t, k, L, Kc_list, lam, mu):
    """
    更激进的动力学 - 更容易产生多稳态
    """
    dMdt = np.zeros(k * L)
    M_arr = np.array(M).reshape(k, L)
    
    for i in range(k):
        for l in range(L):
            Mc = M_arr[i, l]
            Kc_i = Kc_list[i]
            
            # 更强的非线性项 - 使用三次项来产生双稳态
            # dM/dt = M*(1-M/Kc)*(M - a) 形式
            term1 = Mc * (1 - Mc / Kc_i) * (Mc - 0.3)  # 0.3是"转换点"
            
            # 群体间竞争 - winner-take-all 倾向
            competition = 0
            for j in range(k):
                if j != i:
                    Mj_mean = np.mean(M_arr[j, :])
                    # 差分竞争
                    competition += lam * (Mj_mean - Mc) * Mc
            
            # 层间耦合
            layer_coup = 0
            for l2 in range(L):
                if l2 != l:
                    layer_coup += mu * (M_arr[i, l2] - Mc)
            
            dMdt[i * L + l] = term1 + competition + layer_coup
    
    return dMdt


def run_simulation(k, L, Kc_list, lam, mu, n_init=50):
    """运行多次模拟"""
    t = np.linspace(0, 25, 250)
    finals = []
    
    for _ in range(n_init):
        # 不同类型的初始条件
        M0 = np.random.uniform(0.05, 0.95, k * L)
        try:
            sol = odeint(hypergraph_dynamics_v2, M0, t, args=(k, L, Kc_list, lam, mu))
            finals.append(sol[-1])
        except:
            pass
    
    return finals


def cluster(finals, threshold=0.2):
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


# 测试
print("=" * 60)
print("Testing v2 dynamics")
print("=" * 60)

k, L = 3, 2
n_init = 50

# 测试不同配置
tests = [
    ([0.4, 0.4, 0.4], 0.3, 0.0, "Symmetric"),
    ([0.25, 0.40, 0.55], 0.3, 0.0, "Slight asym"),
    ([0.20, 0.40, 0.60], 0.3, 0.0, "Strong asym"),
    ([0.15, 0.40, 0.75], 0.5, 0.0, "Very strong"),
]

for Kc_list, lam, mu, name in tests:
    finals = run_simulation(k, L, Kc_list, lam, mu, n_init)
    unique = cluster(finals, 0.2)
    print(f"\n{name}: Kc={Kc_list}, lam={lam}")
    print(f"  Unique attractors: {len(unique)}")
    for i, u in enumerate(unique):
        print(f"    {i}: {np.round(u, 3)}")

# Lambda 扫描
print("\n" + "=" * 60)
print("Lambda scan with strong asymmetry")
print("=" * 60)

Kc_list = [0.15, 0.40, 0.75]
results = []
for lam in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
    finals = run_simulation(k, L, Kc_list, lam, 0.0, n_init)
    unique = cluster(finals, 0.2)
    results.append((lam, len(unique)))
    print(f"lam={lam:.1f}: {len(unique)} attractors")

print("\nDone!")
