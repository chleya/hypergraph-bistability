"""
真实超图微观动力学 - 离散版
=============================
使用离散时间步迭代，避免ODE stiffness问题
"""
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('F:/hypergraph_bistability/figures/microscopic/', exist_ok=True)
np.random.seed(42)

# ==================== 超图生成 ====================
def generate_hypergraph(N, E, min_size=2, max_size=5):
    """生成随机超图"""
    H = np.zeros((N, E))
    for e in range(E):
        size = np.random.randint(min_size, max_size + 1)
        nodes = np.random.choice(N, size=size, replace=False)
        H[nodes, e] = 1
    return H

# ==================== 离散动力学 ====================
def micro_update(m, H, Kc, lambda_c):
    """微观超图更新（离散时间步）"""
    N = len(m)
    m_new = np.zeros(N)
    
    for v in range(N):
        # 找到v所在的所有超边
        edges = np.where(H[v, :] > 0)[0]
        
        if len(edges) == 0:
            m_new[v] = m[v]
            continue
        
        # 计算局部场：超边内平均
        local_sum = 0
        for e in edges:
            nodes_in_e = np.where(H[:, e] > 0)[0]
            local_sum += np.mean(m[nodes_in_e])
        local_field = local_sum / len(edges)
        
        # 三次项动力学（类似mean-field的双稳）
        K = Kc[v % len(Kc)]
        dm = m[v] * (1 - m[v]) * (local_field - lambda_c * m[v]) * (m[v] - 0.5)
        m_new[v] = np.clip(m[v] + 0.1 * dm, 0, 1)
    
    return m_new

def mean_field_update(M, Kc, lambda_c):
    """Mean-Field更新"""
    L, k = 2, 3
    M_new = np.zeros((L, k))
    
    for l in range(L):
        for i in range(k):
            Mi = M[l, i]
            # 全局平均
            M_mean = np.mean(M[l, :])
            K = Kc[i]
            dm = Mi * (1 - Mi) * (M_mean - lambda_c * Mi) * (Mi - 0.5)
            M_new[l, i] = np.clip(Mi + 0.1 * dm, 0, 1)
    
    return M_new

def run_micro(N, E, Kc, lambda_c, n_init=50, steps=100):
    """运行微观动力学"""
    H = generate_hypergraph(N, E, 2, 4)
    finals = []
    
    for _ in range(n_init):
        m = np.random.uniform(0.1, 0.9, N)
        for _ in range(steps):
            m = micro_update(m, H, Kc, lambda_c)
        finals.append(m)
    
    # 聚类
    return cluster(finals)

def run_mf(Kc, lambda_c, n_init=50, steps=100):
    """运行Mean-Field"""
    finals = []
    
    for _ in range(n_init):
        M = np.random.uniform(0.1, 0.9, (2, 3))
        for _ in range(steps):
            M = mean_field_update(M, Kc, lambda_c)
        finals.append(M.flatten())
    
    return cluster(finals)

def cluster(finals, threshold=0.15):
    """吸引子聚类"""
    finals = np.array(finals)
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
print("离散版: 微观超图 vs Mean-Field")
print("=" * 60)

Kc = [0.4, 0.5, 0.6]

# 微观扫描
print("\n微观超图:")
micro_results = []
for lam in [0.0, 0.2, 0.4, 0.6, 0.8]:
    n = run_micro(30, 50, Kc, lam)
    micro_results.append((lam, n))
    print(f"  λ={lam}: {n}个吸引子")

# Mean-Field扫描
print("\nMean-Field:")
mf_results = []
for lam in [0.0, 0.2, 0.4, 0.6, 0.8]:
    n = run_mf(Kc, lam)
    mf_results.append((lam, n))
    print(f"  λ={lam}: {n}个吸引子")

# 绘图
fig, ax = plt.subplots(figsize=(10, 6))
lam_m = [r[0] for r in micro_results]
n_m = [r[1] for r in micro_results]
lam_f = [r[0] for r in mf_results]
n_f = [r[1] for r in mf_results]

ax.plot(lam_m, n_m, 'o-', label='Microscopic (Hypergraph)', linewidth=2, markersize=8, color='steelblue')
ax.plot(lam_f, n_f, 's--', label='Mean-Field', linewidth=2, markersize=8, color='coral')
ax.set_xlabel('λ (Coupling)', fontsize=12)
ax.set_ylabel('Number of Attractors', fontsize=12)
ax.set_title('Microscopic Hypergraph vs Mean-Field', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('F:/hypergraph_bistability/figures/microscopic/compare.png', dpi=150)
plt.close()
print("\nSaved: compare.png")

# ==================== 拓扑效应 ====================
print("\n" + "=" * 60)
print("拓扑效应:")
print("=" * 60)

topos = {
    'Sparse (E=N)': (30, 30),
    'Medium (E=1.5N)': (30, 45),
    'Dense (E=2N)': (30, 60),
}

topo_results = []
for name, (N, E) in topos.items():
    n = run_micro(N, E, Kc, 0.4)
    topo_results.append((name, n))
    print(f"  {name}: {n}个吸引子")

fig, ax = plt.subplots(figsize=(8, 5))
names = [t[0] for t in topo_results]
vals = [t[1] for t in topo_results]
ax.bar(names, vals, color=['lightblue', 'steelblue', 'darkblue'])
ax.set_ylabel('Attractors')
ax.set_title('Topology Density Effect')
for i, v in enumerate(vals):
    ax.text(i, v + 0.2, str(v), ha='center')
plt.tight_layout()
plt.savefig('F:/hypergraph_bistability/figures/microscopic/topo.png', dpi=150)
plt.close()
print("Saved: topo.png")

print("\n完成!")
