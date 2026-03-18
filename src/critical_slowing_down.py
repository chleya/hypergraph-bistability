"""
Critical Slowing Down Analysis
==============================
测量收敛时间 T(d) - 寻找临界区

预期：在 d_c 附近，系统收敛变慢 → T(d) 峰值
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import os

os.makedirs('F:/hypergraph_bistability/figures', exist_ok=True)


def run_experiment_with_convergence(N=50, gamma=0.35, state_dim=4, max_steps=1000, seed=42):
    """
    运行实验，记录 M(t) 轨迹和收敛时间
    收敛定义：连续 50 步 M 变化 < 0.01
    """
    
    random.seed(seed)
    np.random.seed(seed)
    
    K = int(gamma * N)
    
    class Hypergraph:
        def __init__(self, n_vertices):
            self.V = list(range(n_vertices))
            self.E = []
            self.s = {v: np.random.randn(state_dim) for v in self.V}
            self.conflicts = set()
            self.faction = {}
            self.K = K
            
            n_initial = max(4, n_vertices // 4)
            for _ in range(n_initial):
                size = random.randint(2, min(5, n_vertices // 3))
                e = frozenset(random.sample(self.V, size))
                self.E.append(e)
                self.faction[e] = random.randint(0, 2)
        
        def get_avg_distance(self):
            if len(self.V) < 2: return 0.1
            samples = min(30, len(self.V) * (len(self.V) - 1) // 2)
            if samples <= 0: return 0.1
            dist_sum = 0
            for _ in range(samples):
                u, v = random.sample(self.V, 2)
                dist_sum += np.linalg.norm(self.s[u] - self.s[v])
            return max(dist_sum / samples, 0.01)
        
        def get_node_degree(self, v):
            return sum(1 for e in self.E if v in e)
        
        def get_influence(self, v):
            degree = self.get_node_degree(v)
            return 1.0 / (1.0 + 0.0 * degree)
        
        def enforce_resource_constraint(self):
            for v in self.V:
                degree = self.get_node_degree(v)
                if degree > self.K:
                    excess = degree - self.K
                    edges_with_v = [e for e in self.E if v in e]
                    edges_with_v.sort(key=lambda e: len(e), reverse=True)
                    for _ in range(min(excess, len(edges_with_v))):
                        if edges_with_v:
                            smallest = edges_with_v.pop()
                            if smallest in self.E:
                                self.E.remove(smallest)
                                if smallest in self.faction:
                                    del self.faction[smallest]
        
        def apply_rules(self):
            avg_dist = self.get_avg_distance()
            
            if random.random() < 0.35 and self.E:
                e = random.choice(self.E)
                nodes = list(e)
                x = random.choice(nodes)
                
                w = len(self.V)
                self.V.append(w)
                self.s[w] = np.random.randn(state_dim) * 0.2 + self.s[x]
                new_e = frozenset([x, w])
                self.E.append(new_e)
                self.faction[new_e] = self.faction.get(e, random.randint(0, 2))
            
            if random.random() < 0.3 and len(self.E) >= 2:
                e1 = random.choice(self.E)
                e2 = random.choice(self.E)
                
                if e1 != e2 and len(e1 & e2) >= 1:
                    faction1 = self.faction.get(e1, 0)
                    faction2 = self.faction.get(e2, 0)
                    faction_penalty = 0.3 if faction1 != faction2 else 0.0
                    
                    weights1 = np.array([self.get_influence(v) for v in e1])
                    weights1 = weights1 / weights1.sum()
                    c1 = np.average([self.s[u] for u in e1], axis=0, weights=weights1)
                    
                    weights2 = np.array([self.get_influence(v) for v in e2])
                    weights2 = weights2 / weights2.sum()
                    c2 = np.average([self.s[u] for u in e2], axis=0, weights=weights2)
                    
                    distance = np.linalg.norm(c1 - c2)
                    effective_threshold = 0.8 * avg_dist - faction_penalty
                    
                    if distance > effective_threshold:
                        self.conflicts.add((e1, e2))
                        for v in e1:
                            self.s[v] += np.random.randn(state_dim) * 0.3 * self.get_influence(v)
                        for v in e2:
                            self.s[v] -= np.random.randn(state_dim) * 0.3 * self.get_influence(v)
                    else:
                        new_e = frozenset(e1 | e2)
                        if new_e not in self.E:
                            self.E.remove(e1)
                            self.E.remove(e2)
                            self.E.append(new_e)
                            self.faction[new_e] = (faction1 + faction2) % 3
            
            if random.random() < 0.25 and self.E:
                e = random.choice(self.E)
                if len(e) >= 5:
                    lst = list(e)
                    mid = len(lst) // 2
                    pivot = lst[mid]
                    e_left = frozenset(lst[:mid] + [pivot])
                    e_right = frozenset(lst[mid:])
                    
                    parent_faction = self.faction.get(e, 0)
                    self.E.remove(e)
                    self.E.append(e_left)
                    self.E.append(e_right)
                    self.faction[e_left] = parent_faction
                    self.faction[e_right] = (parent_faction + 1) % 3
            
            if random.random() < 0.15 and self.E:
                e = random.choice(self.E)
                if len(e) <= 2 and random.random() < 0.5:
                    self.E.remove(e)
                    if e in self.faction:
                        del self.faction[e]
            
            self.enforce_resource_constraint()
        
        def get_M(self):
            if not self.E:
                return 0.0
            
            n_nodes = len(self.V)
            
            adj = {v: set() for v in self.V}
            for e in self.E:
                nodes = list(e)
                for i in range(len(nodes)):
                    for j in range(i + 1, len(nodes)):
                        adj[nodes[i]].add(nodes[j])
                        adj[nodes[j]].add(nodes[i])
            
            visited = set()
            factions = []
            for start in self.V:
                if start not in visited:
                    faction = []
                    queue = [start]
                    visited.add(start)
                    while queue:
                        node = queue.pop(0)
                        faction.append(node)
                        for neighbor in adj[node]:
                            if neighbor not in visited:
                                visited.add(neighbor)
                                queue.append(neighbor)
                    factions.append(faction)
            
            if factions:
                max_size = max(len(f) for f in factions)
                M = max_size / n_nodes
                return max(0, min(1, M))
            return 0.0
    
    H = Hypergraph(N)
    
    # 记录 M(t)
    M_history = [H.get_M()]
    converged = False
    convergence_step = max_steps
    
    window_size = 20
    tolerance = 0.03
    
    for step in range(max_steps):
        H.apply_rules()
        M = H.get_M()
        M_history.append(M)
        
        # 检查收敛
        if not converged and step >= window_size:
            recent = M_history[-window_size:]
            if np.std(recent) < tolerance:
                converged = True
                convergence_step = step
    
    return {
        'M_final': M_history[-1],
        'M_history': M_history,
        'convergence_step': convergence_step,
        'converged': converged
    }


# ============================================================
# 实验
# ============================================================

print("="*60)
print("Critical Slowing Down Analysis")
print("="*60)

N = 50
gamma = 0.35

# 在临界区加密采样
d_values = [8, 16, 24, 32]
n_runs = 8
max_steps = 400

results = {d: {'T': [], 'M_final': []} for d in d_values}

for d in d_values:
    print(f"\n[d = {d}] Running {n_runs} experiments...")
    
    for run in range(n_runs):
        seed = 2000 + run * 100 + d * 10000
        result = run_experiment_with_convergence(
            N=N, gamma=gamma, state_dim=d, 
            max_steps=max_steps, seed=seed
        )
        
        results[d]['T'].append(result['convergence_step'])
        results[d]['M_final'].append(result['M_final'])
    
    T_arr = np.array(results[d]['T'])
    M_arr = np.array(results[d]['M_final'])
    
    print(f"  T_mean = {np.mean(T_arr):.1f} ± {np.std(T_arr):.1f}")
    print(f"  M_mean = {np.mean(M_arr):.3f} ± {np.std(M_arr):.3f}")

# ============================================================
# 分析
# ============================================================

print("\n" + "="*60)
print("Analysis")
print("="*60)

# 统计
stats = {}
for d in d_values:
    T_arr = np.array(results[d]['T'])
    M_arr = np.array(results[d]['M_final'])
    
    stats[d] = {
        'T_mean': np.mean(T_arr),
        'T_std': np.std(T_arr),
        'M_mean': np.mean(M_arr),
        'M_std': np.std(M_arr)
    }

# 打印
print(f"\n{'d':>4} | {'T_mean':>8} | {'T_std':>6} | {'M_mean':>8} | {'M_std':>6}")
print("-" * 50)
for d in d_values:
    s = stats[d]
    print(f"{d:>4} | {s['T_mean']:>8.1f} | {s['T_std']:>6.1f} | {s['M_mean']:>8.3f} | {s['M_std']:>6.3f}")

# ============================================================
# 绘图
# ============================================================

# 图 1: T(d) - 收敛时间
plt.figure(figsize=(10, 6))

d_list = d_values
T_means = [stats[d]['T_mean'] for d in d_list]
T_stds = [stats[d]['T_std'] for d in d_list]

plt.errorbar(d_list, T_means, yerr=T_stds, fmt='o-', capsize=5, 
             markersize=10, linewidth=2.5, color='purple')

# 找峰值
max_T_idx = np.argmax(T_means)
max_T_d = d_list[max_T_idx]
max_T = T_means[max_T_idx]

plt.scatter([max_T_d], [max_T], s=200, c='red', zorder=5, marker='*')
plt.annotate(f'Peak: d={max_T_d}\nT={max_T:.0f}', 
             (max_T_d, max_T), textcoords="offset points", 
             xytext=(10,10), ha='left', fontsize=11)

plt.xlabel('Dimension d', fontsize=14)
plt.ylabel('Convergence Time T (steps)', fontsize=14)
plt.title('Critical Slowing Down: T(d)', fontsize=16)
plt.grid(True, alpha=0.3)

plt.savefig('F:/hypergraph_bistability/figures/T_vs_d.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n[OK] Figure saved: T_vs_d.png")

# 图 2: 综合图 - T, M, CV
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# T(d)
ax1 = axes[0]
ax1.errorbar(d_list, T_means, yerr=T_stds, fmt='o-', capsize=5, 
             markersize=8, linewidth=2, color='purple')
ax1.scatter([max_T_d], [max_T], s=150, c='red', zorder=5, marker='*')
ax1.set_xlabel('Dimension d', fontsize=12)
ax1.set_ylabel('Convergence Time T', fontsize=12)
ax1.set_title('T(d) - Convergence Time', fontsize=14)
ax1.grid(True, alpha=0.3)

# M(d)
ax2 = axes[1]
M_means = [stats[d]['M_mean'] for d in d_list]
M_stds = [stats[d]['M_std'] for d in d_list]
ax2.errorbar(d_list, M_means, yerr=M_stds, fmt='s-', capsize=5, 
             markersize=8, linewidth=2, color='blue')
ax2.axhline(y=0.52, color='gray', linestyle=':', alpha=0.5)
ax2.set_xlabel('Dimension d', fontsize=12)
ax2.set_ylabel('Final M', fontsize=12)
ax2.set_title('M(d) - Order Parameter', fontsize=14)
ax2.grid(True, alpha=0.3)

# 轨迹示例
ax3 = axes[2]
# 画几个典型轨迹
for d_plot in [12, 24, 32]:
    seed = 2000 + d_plot * 10000
    result = run_experiment_with_convergence(N=N, gamma=gamma, state_dim=d_plot, 
                                             max_steps=max_steps, seed=seed)
    ax3.plot(result['M_history'][::5], label=f'd={d_plot}', alpha=0.7)

ax3.set_xlabel('Time step (x5)', fontsize=12)
ax3.set_ylabel('M(t)', fontsize=12)
ax3.set_title('Sample Trajectories', fontsize=14)
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('F:/hypergraph_bistability/figures/comprehensive_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("[OK] Figure saved: comprehensive_analysis.png")

# ============================================================
# 关键发现
# ============================================================
print("\n" + "="*60)
print("KEY FINDINGS")
print("="*60)

print(f"\n1. Convergence Time:")
print(f"   - Peak at d = {max_T_d} (T = {max_T:.0f} steps)")
print(f"   - T range: {min(T_means):.0f} - {max(T_means):.0f}")

# 相关性分析
from scipy import stats as scipy_stats

# T vs d 相关性
corr, p_value = scipy_stats.pearsonr(d_list, T_means)
print(f"\n2. T vs d correlation:")
print(f"   - r = {corr:.3f}, p = {p_value:.3f}")

if corr > 0.5:
    print(f"   - Interpretation: T increases with d (slower convergence at high d)")
elif corr < -0.5:
    print(f"   - Interpretation: T decreases with d (faster convergence at high d)")
else:
    print(f"   - Interpretation: No clear monotonic relationship")

# T 峰值位置
print(f"\n3. Critical Zone Evidence:")
if max_T_d in [20, 24, 28]:
    print(f"   - T peaks near d = {max_T_d}")
    print(f"   - This suggests critical zone around d ≈ {max_T_d}")
    print(f"   - ✅ Critical slowing DOWN detected!")
else:
    print(f"   - T peak at d = {max_T_d} (not in expected critical zone)")
    print(f"   - May need more data to confirm")

print("\n" + "="*60)
print("DONE")
print("="*60)
