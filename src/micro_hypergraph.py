"""
真实超图微观动力学 vs Mean-Field 对比实验
============================================
目标：验证拓扑是否"放大"或"抑制"多稳态
"""
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import os

os.makedirs('F:/hypergraph_bistability/figures/microscopic/', exist_ok=True)

# ==================== 超图生成 ====================
def generate_hypergraph(N_nodes, N_edges, min_size=2, max_size=5):
    """生成随机超图（关联矩阵H）"""
    H = np.zeros((N_nodes, N_edges))
    for e in range(N_edges):
        size = np.random.randint(min_size, max_size + 1)
        nodes = np.random.choice(N_nodes, size, replace=False)
        H[nodes, e] = 1
    return H

def generate_config_model_hypergraph(N_nodes, N_edges, degree_dist):
    """生成配置模型超图（给定度分布）"""
    # 简化版本：使用幂律度分布
    H = np.zeros((N_nodes, N_edges))
    degrees = np.random.choice(N_nodes, size=N_edges, p=degree_dist/degree_dist.sum())
    degrees = np.clip(degrees, 2, 10)
    for e in range(N_edges):
        nodes = np.random.choice(N_nodes, size=degrees[e], replace=False)
        H[nodes, e] = 1
    return H

# ==================== 微观动力学 ====================
def microscopic_dynamics(m, t, H, Kc_list, lambda_local, lambda_global, mu):
    """
    微观超图动力学
    m: N_nodes 向量（每个节点的序参量）
    H: N_nodes x N_edges 关联矩阵
    """
    N_nodes = len(m)
    dm = np.zeros(N_nodes)
    
    # 每条超边的局部竞争
    for e in range(H.shape[1]):
        nodes_in_e = np.where(H[:, e] > 0)[0]
        if len(nodes_in_e) < 2:
            continue
            
        # 超边内 winner-take-all
        m_e = m[nodes_in_e]
        m_max = np.max(m_e)
        winner_idx = nodes_in_e[np.argmax(m_e)]
        
        # 对胜者增强，对败者抑制
        for v in nodes_in_e:
            if v == winner_idx:
                # 胜者：向1增长，受Kc约束
                dm[v] += m[v] * (1 - m[v]) * (1 - m[v]/Kc_list[v % len(Kc_list)])
            else:
                # 败者：被压制
                dm[v] -= lambda_local * m[v] * m_max
    
    # 全局跨群体抑制
    for i in range(len(Kc_list)):
        group_nodes = np.arange(i, N_nodes, len(Kc_list))
        m_group = m[group_nodes]
        mean_m = np.mean(m_group)
        for v in group_nodes:
            dm[v] -= lambda_global * m[v] * mean_m
    
    # 跨层耦合（简化：全局）
    # dm += mu * (np.mean(m) - m)  # 可选
    
    return dm

# ==================== Mean-Field版本（对比） ====================
def mean_field_dynamics(M_flat, t, omega, a_list, b_list, c_list, lambda_, mu, L, k):
    """原mean-field版本（来自multistable_chen.py）"""
    M = M_flat.reshape((L, k))
    dM = np.zeros((L, k))
    for l in range(L):
        for i in range(k):
            self_term = omega[i] * (a_list[i] * M[l,i]**3 + b_list[i] * M[l,i]**2 + c_list[i] * M[l,i])
            cross_group = -lambda_ * M[l,i] * np.sum(M[l, np.arange(k) != i])
            cross_layer = mu * np.sum(M[np.arange(L) != l, i])
            dM[l,i] = self_term + cross_group + cross_layer
    return dM.flatten()

def cluster_states(finals, threshold=0.05):
    """聚类吸引子"""
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

# ==================== 实验1: 基本对比 ====================
print("=" * 60)
print("实验1: 微观超图 vs Mean-Field 对比")
print("=" * 60)

# 参数
N_nodes = 50
N_edges = 80
Kc_list = [0.32, 0.40, 0.48]
lambda_local = 0.3
lambda_global = 0.2

# 生成超图
H = generate_hypergraph(N_nodes, N_edges, min_size=2, max_size=5)
print(f"超图: {N_nodes}节点, {N_edges}边, 平均度={H.sum()/N_nodes:.2f}")

# 微观动力学扫描
print("\n微观动力学 (λ_local扫描):")
micro_results = []
for lam_loc in [0.0, 0.1, 0.3, 0.5, 0.7]:
    finals = []
    t = np.linspace(0, 60, 600)
    for _ in range(60):
        init = np.random.uniform(0.01, 0.99, N_nodes)
        sol = odeint(microscopic_dynamics, init, t, 
                    args=(H, Kc_list, lam_loc, lambda_global, 0.0),
                    atol=1e-6, rtol=1e-6)
        finals.append(sol[-1])
    
    n_states, _ = cluster_states(np.array(finals), threshold=0.08)
    micro_results.append((lam_loc, n_states))
    print(f"  λ_local={lam_loc:.1f}: {n_states}个吸引子")

# Mean-Field对比
print("\nMean-Field (λ扫描):")
mf_results = []
omega = np.ones(3)
a_list = [-3.0/kc for kc in Kc_list]
b_list = [4.5/kc for kc in Kc_list]
c_list = [-1.5/kc for kc in Kc_list]

for lam in [0.0, 0.1, 0.3, 0.5, 0.7]:
    finals = []
    t = np.linspace(0, 60, 600)
    params = (omega, a_list, b_list, c_list, lam, 0.0, 2, 3)
    for _ in range(60):
        init = np.random.uniform(0.01, 0.99, 6)
        sol = odeint(mean_field_dynamics, init, t, args=params, atol=1e-6, rtol=1e-6)
        finals.append(sol[-1])
    
    n_states, _ = cluster_states(np.array(finals), threshold=0.05)
    mf_results.append((lam, n_states))
    print(f"  λ={lam:.1f}: {n_states}个吸引子")

# ==================== 绘图 ====================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 左图：对比
lam_micro = [r[0] for r in micro_results]
n_micro = [r[1] for r in micro_results]
lam_mf = [r[0] for r in mf_results]
n_mf = [r[1] for r in mf_results]

ax1.plot(lam_micro, n_micro, 'o-', label='Microscopic (Hypergraph)', linewidth=2, markersize=8)
ax1.plot(lam_mf, n_mf, 's--', label='Mean-Field', linewidth=2, markersize=8)
ax1.set_xlabel('λ (Coupling Strength)', fontsize=12)
ax1.set_ylabel('Number of Attractors', fontsize=12)
ax1.set_title('Microscopic vs Mean-Field', fontsize=14)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 50)

# 右图：超图结构
ax2.spy(H, markersize=1, aspect='auto')
ax2.set_xlabel('Hyperedge Index', fontsize=12)
ax2.set_ylabel('Node Index', fontsize=12)
ax2.set_title(f'Hypergraph Structure\n(N={N_nodes}, E={N_edges})', fontsize=14)

plt.tight_layout()
plt.savefig('F:/hypergraph_bistability/figures/microscopic/fig1_micro_vs_mf.png', dpi=150)
plt.close()
print("\nSaved: fig1_micro_vs_mf.png")

# ==================== 实验2: 拓扑效应 ====================
print("\n" + "=" * 60)
print("实验2: 拓扑效应（不同超图结构）")
print("=" * 60)

topologies = {
    'Uniform': lambda: generate_hypergraph(50, 80, 2, 5),
    'Small-World': lambda: generate_hypergraph(50, 80, 3, 4),  # 简化
    'Scale-Free': lambda: generate_config_model_hypergraph(50, 80, 
                  np.array([10, 8, 6, 5, 4, 3, 2, 1, 1, 1], dtype=float)),
}

topo_results = {}
for name, gen_fn in topologies.items():
    H_topo = gen_fn()
    finals = []
    t = np.linspace(0, 60, 600)
    for _ in range(60):
        init = np.random.uniform(0.01, 0.99, 50)
        sol = odeint(microscopic_dynamics, init, t,
                    args=(H_topo, Kc_list, 0.3, 0.2, 0.0),
                    atol=1e-6, rtol=1e-6)
        finals.append(sol[-1])
    
    n_states, _ = cluster_states(np.array(finals), threshold=0.08)
    topo_results[name] = n_states
    print(f"  {name}: {n_states}个吸引子")

# 绘图
fig, ax = plt.subplots(figsize=(8, 5))
names = list(topo_results.keys())
values = list(topo_results.values())
bars = ax.bar(names, values, color=['steelblue', 'coral', 'green'], alpha=0.7)
ax.set_xlabel('Topology Type', fontsize=12)
ax.set_ylabel('Number of Attractors', fontsize=12)
ax.set_title('Topology Effect on Multistability\n(λ_local=0.3)', fontsize=14)
for i, v in enumerate(values):
    ax.text(i, v + 0.5, str(v), ha='center', fontsize=12)
plt.tight_layout()
plt.savefig('F:/hypergraph_bistability/figures/microscopic/fig2_topology.png', dpi=150)
plt.close()
print("Saved: fig2_topology.png")

# ==================== 实验3: N_nodes缩放 ====================
print("\n" + "=" * 60)
print("实验3: 节点数缩放效应")
print("=" * 60)

scale_results = []
for N in [20, 50, 100, 200]:
    H_scale = generate_hypergraph(N, int(N*1.6), 2, 5)
    finals = []
    t = np.linspace(0, 60, 600)
    Kc = [0.32, 0.40, 0.48] * (N // 3 + 1)
    for _ in range(40):
        init = np.random.uniform(0.01, 0.99, N)
        sol = odeint(microscopic_dynamics, init, t,
                    args=(H_scale, Kc[:N], 0.3, 0.2, 0.0),
                    atol=1e-6, rtol=1e-6)
        finals.append(sol[-1])
    
    n_states, _ = cluster_states(np.array(finals), threshold=0.08)
    scale_results.append((N, n_states))
    print(f"  N={N}: {n_states}个吸引子")

# 绘图
fig, ax = plt.subplots(figsize=(8, 5))
Ns = [r[0] for r in scale_results]
n_vals = [r[1] for r in scale_results]
ax.plot(Ns, n_vals, 'o-', linewidth=2, markersize=8)
ax.set_xlabel('Number of Nodes', fontsize=12)
ax.set_ylabel('Number of Attractors', fontsize=12)
ax.set_title('Scaling: Attractors vs System Size', fontsize=14)
ax.grid(True, alpha=0.3)
for i, (N, n) in enumerate(scale_results):
    ax.annotate(str(n), (N, n), textcoords="offset points", xytext=(0,10), ha='center')
plt.tight_layout()
plt.savefig('F:/hypergraph_bistability/figures/microscopic/fig3_scaling.png', dpi=150)
plt.close()
print("Saved: fig3_scaling.png")

print("\n" + "=" * 60)
print("全部实验完成！")
print("=" * 60)
