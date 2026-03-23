"""
真实超图微观动力学 - 简化版
=============================
"""
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import os

os.makedirs('F:/hypergraph_bistability/figures/microscopic/', exist_ok=True)

# ==================== 超图生成 ====================
def generate_hypergraph(N_nodes, N_edges, min_size=2, max_size=5):
    """生成随机超图"""
    H = np.zeros((N_nodes, N_edges))
    for e in range(N_edges):
        size = np.random.randint(min_size, max_size + 1)
        nodes = np.random.choice(N_nodes, size, replace=False)
        H[nodes, e] = 1
    return H

# ==================== 简化微观动力学 ====================
def simple_micro_dynamics(m, t, H, Kc, lambda抑制):
    """简化版微观动力学"""
    N = len(m)
    dm = np.zeros(N)
    
    # 每个节点的局部场
    for v in range(N):
        # 找到包含v的所有超边
        edges_with_v = np.where(H[v, :] > 0)[0]
        
        # 这些超边内的平均激活
        local_field = 0
        for e in edges_with_v:
            nodes_in_e = np.where(H[:, e] > 0)[0]
            local_field += np.mean(m[nodes_in_e])
        
        if len(edges_with_v) > 0:
            local_field /= len(edges_with_v)
        
        # 动力学：向局部场收敛，受Kc约束
        dm[v] = lambda抑制 * (local_field - m[v]) * (1 - m[v]/Kc[v % len(Kc)])
    
    return dm

# ==================== Mean-Field对比 ====================
def mean_field(M, t, Kc_list, lambda_):
    """简化Mean-Field"""
    L, k = 2, 3
    M = M.reshape(L, k)
    dM = np.zeros_like(M)
    
    for l in range(L):
        for i in range(k):
            # 自我 + 竞争
            self_term = M[l,i] * (1 - M[l,i]/Kc_list[i])
            cross = -lambda_ * M[l,i] * np.mean(M[l, :])
            dM[l,i] = self_term + cross
    
    return dM.flatten()

def cluster_states(finals, threshold=0.1):
    unique = []
    for s in finals:
        is_new = True
        for u in unique:
            if np.sqrt(np.mean((s - u)**2)) < threshold:
                is_new = False
                break
        if is_new:
            unique.append(s)
    return len(unique)

# ==================== 实验 ====================
print("=" * 60)
print("实验: 微观超图 vs Mean-Field")
print("=" * 60)

# 参数
N = 30
E = 50
Kc = [0.4, 0.5, 0.6]

# 生成超图
np.random.seed(42)
H = generate_hypergraph(N, E, 2, 4)
print(f"超图: N={N}, E={E}")

# 微观动力学扫描
print("\n微观动力学:")
micro_results = []
for lam in [0.0, 0.1, 0.3, 0.5]:
    finals = []
    t = np.linspace(0, 40, 400)
    for _ in range(40):
        init = np.random.uniform(0.01, 0.8, N)
        try:
            sol = odeint(simple_micro_dynamics, init, t, 
                        args=(H, Kc, lam), atol=1e-4, rtol=1e-4)
            finals.append(sol[-1])
        except:
            pass
    
    n = cluster_states(np.array(finals))
    micro_results.append((lam, n))
    print(f"  λ={lam}: {n}个吸引子")

# Mean-Field扫描
print("\nMean-Field:")
mf_results = []
for lam in [0.0, 0.1, 0.3, 0.5]:
    finals = []
    t = np.linspace(0, 40, 400)
    for _ in range(40):
        init = np.random.uniform(0.01, 0.8, 6)
        try:
            sol = odeint(mean_field, init, t, args=(Kc, lam), atol=1e-4, rtol=1e-4)
            finals.append(sol[-1])
        except:
            pass
    
    n = cluster_states(np.array(finals))
    mf_results.append((lam, n))
    print(f"  λ={lam}: {n}个吸引子")

# 绘图
fig, ax = plt.subplots(figsize=(10, 6))
lam_m = [r[0] for r in micro_results]
n_m = [r[1] for r in micro_results]
lam_f = [r[0] for r in mf_results]
n_f = [r[1] for r in mf_results]

ax.plot(lam_m, n_m, 'o-', label='Microscopic (Hypergraph)', linewidth=2, markersize=8)
ax.plot(lam_f, n_f, 's--', label='Mean-Field', linewidth=2, markersize=8)
ax.set_xlabel('λ (Coupling)', fontsize=12)
ax.set_ylabel('Attractors', fontsize=12)
ax.set_title('Microscopic vs Mean-Field', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('F:/hypergraph_bistability/figures/microscopic/micro_vs_mf.png', dpi=150)
plt.close()
print("\nSaved: micro_vs_mf.png")

print("\n完成!")
