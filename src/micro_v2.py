"""
微观超图动力学 - 改进版
========================
使用更稳定的动力学：每个节点向其超边内平均收敛
"""
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('F:/hypergraph_bistability/figures/microscopic/', exist_ok=True)
np.random.seed(42)

# ==================== 超图 ====================
def gen_hypergraph(N, E, min_s=2, max_s=5):
    H = np.zeros((N, E))
    for e in range(E):
        s = np.random.randint(min_s, max_s+1)
        nodes = np.random.choice(N, size=s, replace=False)
        H[nodes, e] = 1
    return H

# ==================== 改进动力学 ====================
def micro_dynamics(m, H, Kc_list, lam):
    """改进版：节点受超边内平均的吸引/排斥"""
    N = len(m)
    dm = np.zeros(N)
    
    # 每条超边计算平均激活
    edge_means = []
    for e in range(H.shape[1]):
        nodes = np.where(H[:, e] > 0)[0]
        if len(nodes) > 0:
            edge_means.append(np.mean(m[nodes]))
        else:
            edge_means.append(0)
    edge_means = np.array(edge_means)
    
    for v in range(N):
        edges = np.where(H[v, :] > 0)[0]
        if len(edges) == 0:
            dm[v] = 0
            continue
        
        # 该节点所在超边的平均
        local_mean = np.mean(edge_means[edges])
        Ki = Kc_list[v % len(Kc_list)]
        
        # 动力学：向local_mean收敛，受Ki约束
        # 双稳项：m(1-m)(m-0.5)
        dm[v] = 0.1 * (local_mean - lam * m[v]) * m[v] * (1 - m[v]) * (m[v] - 0.4)
    
    return dm

def run_dynamics(N, E, Kc, lam, n_init=50, steps=80):
    """运行并统计吸引子"""
    H = gen_hypergraph(N, E, 2, 4)
    finals = []
    
    for _ in range(n_init):
        m = np.random.uniform(0.05, 0.95, N)
        for _ in range(steps):
            dm = micro_dynamics(m, H, Kc, lam)
            m = np.clip(m + dm, 0.01, 0.99)
        finals.append(m.copy())
    
    return cluster(np.array(finals))

def cluster(finals, thresh=0.2):
    unique = []
    for s in finals:
        is_new = True
        for u in unique:
            if np.sqrt(np.mean((s-u)**2)) < thresh:
                is_new = False
                break
        if is_new:
            unique.append(s)
    return len(unique)

# ==================== 实验 ====================
print("=" * 50)
print("改进版微观超图 vs Mean-Field")
print("=" * 50)

Kc = [0.4, 0.5, 0.6]

# 微观
print("\n微观超图:")
micro = []
for lam in [0.0, 0.2, 0.4, 0.6]:
    n = run_dynamics(30, 50, Kc, lam)
    micro.append((lam, n))
    print(f"  λ={lam}: {n}")

# MF
print("\nMean-Field:")
mf = []
for lam in [0.0, 0.2, 0.4, 0.6]:
    # 简化MF
    finals = []
    for _ in range(50):
        M = np.random.uniform(0.05, 0.95, 6).reshape(2, 3)
        for _ in range(80):
            for l in range(2):
                for i in range(3):
                    m = M[l,i]
                    m_m = np.mean(M[l,:])
                    dm = 0.1 * (m_m - lam * m) * m * (1-m) * (m-0.4)
                    M[l,i] = np.clip(m + dm, 0.01, 0.99)
        finals.append(M.flatten())
    n = cluster(np.array(finals))
    mf.append((lam, n))
    print(f"  λ={lam}: {n}")

# 绘图
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot([x[0] for x in micro], [x[1] for x in micro], 'o-', label='Microscopic', lw=2, ms=7)
ax.plot([x[0] for x in mf], [x[1] for x in mf], 's--', label='Mean-Field', lw=2, ms=7)
ax.set_xlabel('λ')
ax.set_ylabel('Attractors')
ax.set_title('Microscopic Hypergraph vs Mean-Field')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('F:/hypergraph_bistability/figures/microscopic/v2.png', dpi=150)
plt.close()
print("\nSaved v2.png")
