"""
Step 1: M(d) 曲线 - 使用验证过的代码结构
=========================================
基于 robustness_test.py 的验证代码
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import os

os.makedirs('F:/hypergraph_bistability/figures', exist_ok=True)


def run_experiment(N=50, gamma=0.35, state_dim=4, steps=500, seed=42):
    """运行实验，返回稳态 M"""
    
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
            
            # 初始超边
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
            return 1.0 / (1.0 + 0.0 * degree)  # alpha = 0
        
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
            
            # 支化 - 随机
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
            
            # 融合
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
            
            # 分裂
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
            
            # 消除
            if random.random() < 0.15 and self.E:
                e = random.choice(self.E)
                if len(e) <= 2 and random.random() < 0.5:
                    self.E.remove(e)
                    if e in self.faction:
                        del self.faction[e]
            
            self.enforce_resource_constraint()
        
        def get_M(self):
            """M = 最大阵营占比（用当前节点数归一化）"""
            if not self.E:
                return 0.0
            
            n_nodes = len(self.V)  # 当前节点数
            
            # 找连通分量
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
                # 断言确保在 [0, 1]
                assert 0 <= M <= 1, f"M out of range: {M}"
                return M
            return 0.0
    
    H = Hypergraph(N)
    
    Ms = []
    for _ in range(steps):
        H.apply_rules()
        Ms.append(H.get_M())
    
    return np.mean(Ms[-100:])


# ============================================================
# 实验
# ============================================================

print("="*60)
print("Step 1: M(d) Curve with Error Bars")
print("="*60)

# 固定参数
N = 50
gamma = 0.35
d_values = [2, 4, 8, 16, 32]
n_runs = 5
steps = 500

results = {d: [] for d in d_values}

for d in d_values:
    print(f"\nd = {d}:")
    for run in range(n_runs):
        seed = 42 + run * 100 + d
        M = run_experiment(N=N, gamma=gamma, state_dim=d, steps=steps, seed=seed)
        results[d].append(M)
        
        # 验证 M 在范围内
        if not (0 <= M <= 1):
            print(f"  WARNING: M out of range: {M}")
        
        print(f"  run {run+1}: M = {M:.4f}")

# 统计
print("\n" + "="*60)
print("Statistics")
print("="*60)

means = []
stds = []
for d in d_values:
    mean_M = np.mean(results[d])
    std_M = np.std(results[d])
    means.append(mean_M)
    stds.append(std_M)
    print(f"d = {d:2d}: M* = {mean_M:.4f} +/- {std_M:.4f}")

# 绘图
plt.figure(figsize=(10, 6))
plt.errorbar(d_values, means, yerr=stds, fmt='o-', capsize=5, capthick=2, 
             markersize=10, linewidth=2, color='blue')
plt.axhline(y=0.45, color='red', linestyle='--', alpha=0.5, label='M* = 0.45')
plt.xscale('log')
plt.xlabel('Dimension d', fontsize=14)
plt.ylabel('M* (max faction ratio)', fontsize=14)
plt.title(f'M(d) Curve (N={N}, gamma={gamma}, {n_runs} runs each)', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(0, 1)

# 保存
plt.savefig('F:/hypergraph_bistability/figures/M_d_curve.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\nFigure saved: figures/M_d_curve.png")

# 分析
print("\n" + "="*60)
print("Analysis")
print("="*60)

# 检查趋势
print("\nTrend:")
for i, d in enumerate(d_values):
    if i > 0:
        delta = means[i] - means[i-1]
        print(f"  d {d_values[i-1]} -> {d}: delta = {delta:+.4f}")

# 检查是否收敛
if means[-1] < 0.55:
    print(f"\n[d={d_values[-1]}] M* = {means[-1]:.4f}, CLOSE to 0.45")
else:
    print(f"\n[d={d_values[-1]}] M* = {means[-1]:.4f}, NOT yet at 0.45")

# 方差分析
print("\nVariance analysis:")
for d, std in zip(d_values, stds):
    cv = std / means[d_values.index(d)] * 100
    status = "OK" if cv < 20 else "HIGH"
    print(f"  d = {d:2d}: CV = {cv:5.1f}% [{status}]")
