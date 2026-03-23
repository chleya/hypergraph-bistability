"""
Group-Mean Driven 微观超图动力学
================================
核心：每个节点跟随所属群体的mean-field动力学
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import os

os.makedirs('F:/hypergraph_bistability/figures/microscopic/', exist_ok=True)
np.random.seed(42)

# ==================== 超图生成 ====================
def generate_uniform_hypergraph(N_nodes, num_edges, min_size=2, max_size=5):
    """生成随机均匀超图"""
    H = np.zeros((N_nodes, num_edges))
    for e in range(num_edges):
        size = np.random.randint(min_size, max_size + 1)
        nodes = np.random.choice(N_nodes, size=size, replace=False)
        H[nodes, e] = 1
    return H

# ==================== Group-Mean Driven 微观动力学 ====================
def group_mean_driven_dynamics(t, m, H, group_assign, layer_assign, Kc_list, lambda_, mu, L, k):
    """
    群体均值驱动的微观动力学
    m: (N_nodes,) 节点状态
    H: 超图关联矩阵
    group_assign, layer_assign: 节点→群体/层映射
    """
    N = len(m)
    
    # Step 1: 计算群体-层均值 M (L x k)
    M = np.zeros((L, k))
    counts = np.zeros((L, k))
    for v in range(N):
        i = group_assign[v]
        l = layer_assign[v]
        M[l, i] += m[v]
        counts[l, i] += 1
    M = M / (counts + 1e-8)
    
    # Step 2: 计算群体均值的导数 dM (完全复用mean-field公式)
    omega = np.ones(k)
    a_list = [-3.0 / kc for kc in Kc_list]
    b_list = [4.5 / kc for kc in Kc_list]
    c_list = [-1.5 / kc for kc in Kc_list]
    
    dM = np.zeros((L, k))
    for l in range(L):
        for i in range(k):
            self_term = omega[i] * (a_list[i]*M[l,i]**3 + b_list[i]*M[l,i]**2 + c_list[i]*M[l,i])
            cross_group = -lambda_ * M[l,i] * np.sum(M[l, np.arange(k)!=i])
            cross_layer = mu * np.sum(M[np.arange(L)!=l, i])
            dM[l,i] = self_term + cross_group + cross_layer
    
    # Step 3: 每个节点跟随所属群体的dM + 超边局部微调
    dm = np.zeros(N)
    for v in range(N):
        i = group_assign[v]
        l = layer_assign[v]
        dm[v] = dM[l, i]  # 核心：全局群体拉扯
        
        # 可选：超边局部微调（很弱）
        incident_edges = np.where(H[v] > 0)[0]
        if len(incident_edges) > 0:
            local_sum = 0
            for e in incident_edges:
                nodes_in_e = np.where(H[:, e] > 0)[0]
                local_sum += np.mean(m[nodes_in_e])
            local_mean = local_sum / len(incident_edges)
            dm[v] += 0.03 * (local_mean - m[v])  # 弱同步项
    
    return dm

# ==================== Mean-Field 版本（对比） ====================
def mean_field_dynamics(t, M_flat, omega, a_list, b_list, c_list, lambda_, mu, L, k):
    """原Mean-Field版本"""
    M = M_flat.reshape(L, k)
    dM = np.zeros((L, k))
    for l in range(L):
        for i in range(k):
            self_term = omega[i] * (a_list[i]*M[l,i]**3 + b_list[i]*M[l,i]**2 + c_list[i]*M[l,i])
            cross_group = -lambda_ * M[l,i] * np.sum(M[l, np.arange(k)!=i])
            cross_layer = mu * np.sum(M[np.arange(L)!=l, i])
            dM[l,i] = self_term + cross_group + cross_layer
    return dM.flatten()

# ==================== 聚类 ====================
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

# ==================== 实验 ====================
print("=" * 60)
print("Group-Mean Driven 微观 vs Mean-Field 对比")
print("=" * 60)

# 参数
L, k = 2, 3
Kc_list = [0.32, 0.40, 0.48]
omega = np.ones(k)
a_list = [-3.0/kc for kc in Kc_list]
b_list = [4.5/kc for kc in Kc_list]
c_list = [-1.5/kc for kc in Kc_list]

# 生成超图和分组
N_nodes = 100
num_edges = 200
group_assign = np.random.randint(0, k, N_nodes)
layer_assign = np.random.randint(0, L, N_nodes)
H = generate_uniform_hypergraph(N_nodes, num_edges, 2, 5)
print(f"超图: N={N_nodes}, E={num_edges}")

# 微观扫描
print("\n微观 (Group-Mean Driven):")
micro_results = []
for lam in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]:
    finals = []
    for _ in range(50):
        init = np.random.uniform(0.01, 0.99, N_nodes)
        sol = solve_ivp(group_mean_driven_dynamics, [0, 80], init,
                       args=(H, group_assign, layer_assign, Kc_list, lam, 0.0, L, k),
                       method='RK45', atol=1e-6, rtol=1e-6)
        finals.append(sol.y[:, -1])
    
    n, _ = simple_cluster(np.array(finals))
    micro_results.append((lam, n))
    print(f"  λ={lam}: {n}个吸引子")

# Mean-Field扫描
print("\nMean-Field:")
mf_results = []
for lam in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]:
    finals = []
    for _ in range(50):
        init = np.random.uniform(0.01, 0.99, L*k)
        sol = solve_ivp(mean_field_dynamics, [0, 80], init,
                       args=(omega, a_list, b_list, c_list, lam, 0.0, L, k),
                       method='RK45', atol=1e-6, rtol=1e-6)
        finals.append(sol.y[:, -1])
    
    n, _ = simple_cluster(np.array(finals))
    mf_results.append((lam, n))
    print(f"  λ={lam}: {n}个吸引子")

# 绘图
fig, ax = plt.subplots(figsize=(10, 6))
lam_m = [r[0] for r in micro_results]
n_m = [r[1] for r in micro_results]
lam_f = [r[0] for r in mf_results]
n_f = [r[1] for r in mf_results]

ax.plot(lam_m, n_m, 'o-', label='Microscopic (Group-Mean Driven)', 
        linewidth=2.5, markersize=8, color='steelblue')
ax.plot(lam_f, n_f, 's--', label='Mean-Field', 
        linewidth=2.5, markersize=8, color='coral')
ax.set_xlabel('λ (Coupling Strength)', fontsize=12)
ax.set_ylabel('Number of Attractors', fontsize=12)
ax.set_title('Microscopic Hypergraph vs Mean-Field\n(N=100 nodes, E=200 edges)', fontsize=14)
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 55)
plt.tight_layout()
plt.savefig('F:/hypergraph_bistability/figures/microscopic/group_mean_vs_mf.png', dpi=150)
plt.close()
print("\nSaved: group_mean_vs_mf.png")

# ==================== 验证：是否真正共享basin ====================
print("\n" + "=" * 60)
print("验证: 微观是否形成共享basin")
print("=" * 60)

# 取λ=0，看终态分布
lam = 0.0
finals = []
for _ in range(100):
    init = np.random.uniform(0.01, 0.99, N_nodes)
    sol = solve_ivp(group_mean_driven_dynamics, [0, 80], init,
                   args=(H, group_assign, layer_assign, Kc_list, lam, 0.0, L, k),
                   method='RK45', atol=1e-6, rtol=1e-6)
    finals.append(sol.y[:, -1])

finals = np.array(finals)
print(f"终态矩阵形状: {finals.shape}")

# 计算群体均值分布
group_means = np.zeros((100, L*k))
for idx, m in enumerate(finals):
    for v in range(N_nodes):
        i = group_assign[v]
        l = layer_assign[v]
        group_means[idx, l*k + i] = m[v]

n_unique, _ = simple_cluster(group_means, threshold=0.1)
print(f"群体均值聚类: {n_unique}个共享basin")
print(f"(如果每个独立，会接近100个)")

print("\n完成!")
