"""
Step 1 Revised: M Distribution Analysis
======================================
核心：不是找 mean M(d)，而是找 P(basin | d)

每个 d 跑 20 次
画 histogram
看是否有双峰
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
    
    for _ in range(steps):
        H.apply_rules()
    
    return H.get_M()


# ============================================================
# 实验：每个 d 跑 20 次
# ============================================================

print("="*60)
print("Step 1 Revised: M Distribution Analysis")
print("="*60)

# 固定参数
N = 50
gamma = 0.35
d_values = [2, 4, 8, 16, 32]
n_runs = 5
steps = 200

results = {d: [] for d in d_values}

for d in d_values:
    print(f"\nd = {d}:")
    for run in range(n_runs):
        seed = 42 + run * 100 + d * 1000
        M = run_experiment(N=N, gamma=gamma, state_dim=d, steps=steps, seed=seed)
        results[d].append(M)
        
        if run % 5 == 0:
            print(f"  run {run+1}: M = {M:.4f}")

# 统计分析
print("\n" + "="*60)
print("Statistics")
print("="*60)

# 分类阈值
THRESHOLD = 0.52

stats = {}
for d in d_values:
    Ms = results[d]
    mean_M = np.mean(Ms)
    std_M = np.std(Ms)
    cv = std_M / mean_M * 100 if mean_M > 0 else 0
    
    # 分类统计
    low_count = sum(1 for m in Ms if m < THRESHOLD)
    high_count = n_runs - low_count
    P_low = low_count / n_runs * 100
    P_high = high_count / n_runs * 100
    
    stats[d] = {
        'mean': mean_M,
        'std': std_M,
        'cv': cv,
        'low_count': low_count,
        'high_count': high_count,
        'P_low': P_low,
        'P_high': P_high
    }
    
    print(f"d = {d:2d}: mean={mean_M:.3f}, std={std_M:.3f}, CV={cv:5.1f}%, P(low)={P_low:5.1f}%, P(high)={P_high:5.1f}%")

# 画 histogram 图
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for i, d in enumerate(d_values):
    ax = axes[i]
    Ms = results[d]
    
    ax.hist(Ms, bins=10, alpha=0.7, edgecolor='black')
    ax.axvline(x=THRESHOLD, color='red', linestyle='--', label=f'threshold={THRESHOLD}')
    ax.axvline(x=np.mean(Ms), color='blue', linestyle='-', label=f'mean={np.mean(Ms):.2f}')
    ax.set_xlabel('M')
    ax.set_ylabel('Count')
    ax.set_title(f'd = {d}')
    ax.set_xlim(0, 1)
    ax.legend(fontsize=8)

# 隐藏多余的 subplot
axes[-1].axis('off')

plt.suptitle(f'M Distribution by Dimension (N={N}, gamma={gamma}, {n_runs} runs each)', fontsize=14)
plt.tight_layout()
plt.savefig('F:/hypergraph_bistability/figures/M_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\nFigure saved: figures/M_distribution.png")

# 画 P(low) vs d 图
plt.figure(figsize=(10, 6))
d_list = list(d_values)
P_low_list = [stats[d]['P_low'] for d in d_list]

plt.plot(d_list, P_low_list, 'o-', markersize=10, linewidth=2, color='blue')
plt.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50%')
plt.xlabel('Dimension d', fontsize=14)
plt.ylabel('P(low basin) %', fontsize=14)
plt.title('Basin Selection Probability vs Dimension', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(0, 100)

plt.savefig('F:/hypergraph_bistability/figures/P_low_vs_d.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"Figure saved: figures/P_low_vs_d.png")

# 关键发现
print("\n" + "="*60)
print("KEY FINDINGS")
print("="*60)

# 找临界区
max_cv_d = max(d_values, key=lambda d: stats[d]['cv'])
print(f"\n1. Maximum variance at d = {max_cv_d} (CV = {stats[max_cv_d]['cv']:.1f}%)")

# 找趋势
if P_low_list[0] < P_low_list[-1]:
    print(f"2. Trend: P(low) increases with d ({P_low_list[0]:.0f}% -> {P_low_list[-1]:.0f}%)")
    print("   => Higher dimension favors LOW basin")
else:
    print(f"2. Trend: P(low) decreases with d ({P_low_list[0]:.0f}% -> {P_low_list[-1]:.0f}%)")
    print("   => Higher dimension favors HIGH basin")

# 双峰检测
print(f"\n3. Critical zone analysis:")
for d in d_values:
    Ms = results[d]
    low_Ms = [m for m in Ms if m < THRESHOLD]
    high_Ms = [m for m in Ms if m >= THRESHOLD]
    
    if len(low_Ms) > 3 and len(high_Ms) > 3:
        print(f"   d = {d}: BIMODAL (low={len(low_Ms)}, high={len(high_Ms)})")
    elif len(low_Ms) > n_runs * 0.7:
        print(f"   d = {d}: LOW dominant ({len(low_Ms)}/{n_runs})")
    elif len(high_Ms) > n_runs * 0.7:
        print(f"   d = {d}: HIGH dominant ({len(high_Ms)}/{n_runs})")
    else:
        print(f"   d = {d}: MIXED (low={len(low_Ms)}, high={len(high_Ms)})")
