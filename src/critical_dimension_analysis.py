"""
Step 1 Final: Critical Dimension Analysis
==========================================
目标：定位临界维度 d_c，描述 basin selection 相变

设计：
- d: 8, 12, 16, 20, 24, 32
- 每个 d: 20+ runs
- 画: histogram, P(low) vs d, CV vs d
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import os
import json

os.makedirs('F:/hypergraph_bistability/figures', exist_ok=True)


def run_experiment(N=50, gamma=0.35, state_dim=4, steps=300, seed=42, verbose=False):
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
# 实验设计：d 加密 + 20 runs
# ============================================================

print("="*60)
print("Critical Dimension Analysis")
print("="*60)

# 固定参数
N = 50
gamma = 0.35

# 加密采样：在 d≈16 附近加密
d_values = [8, 12, 16, 20, 24, 32]
n_runs = 20
steps = 300

results = {d: [] for d in d_values}

# 运行实验
for d in d_values:
    print(f"\n[d = {d}] Running {n_runs} experiments...")
    for run in range(n_runs):
        seed = 1000 + run * 100 + d * 10000
        M = run_experiment(N=N, gamma=gamma, state_dim=d, steps=steps, seed=seed)
        results[d].append(M)
        
        if (run + 1) % 5 == 0:
            print(f"  {run+1}/{n_runs} done, current M = {M:.4f}")
    
    Ms = results[d]
    print(f"  d={d}: mean={np.mean(Ms):.3f}, std={np.std(Ms):.3f}")

# 保存原始数据
data_to_save = {str(d): results[d] for d in d_values}
with open('F:/hypergraph_bistability/figures/distribution_data.json', 'w') as f:
    json.dump(data_to_save, f, indent=2)

print("\n" + "="*60)
print("Analysis")
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

# 打印统计
print("\n--- Statistics ---")
print(f"{'d':>4} | {'mean M':>8} | {'std':>6} | {'CV':>6} | {'P(low)':>8} | {'P(high)':>8}")
print("-" * 55)
for d in d_values:
    s = stats[d]
    print(f"{d:>4} | {s['mean']:>8.3f} | {s['std']:>6.3f} | {s['cv']:>5.1f}% | {s['P_low']:>7.1f}% | {s['P_high']:>7.1f}%")

# ============================================================
# 图 1: Histogram
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, d in enumerate(d_values):
    ax = axes[i]
    Ms = results[d]
    
    # 画 histogram
    ax.hist(Ms, bins=10, alpha=0.7, edgecolor='black', color='steelblue')
    ax.axvline(x=THRESHOLD, color='red', linestyle='--', linewidth=2, label=f'threshold={THRESHOLD}')
    ax.axvline(x=np.mean(Ms), color='orange', linestyle='-', linewidth=2, label=f'mean={np.mean(Ms):.2f}')
    
    # 标记 low/high
    low_Ms = [m for m in Ms if m < THRESHOLD]
    high_Ms = [m for m in Ms if m >= THRESHOLD]
    ax.text(0.05, 0.95, f'Low: {len(low_Ms)}/20\nHigh: {len(high_Ms)}/20', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel('M', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'd = {d} (CV = {stats[d]["cv"]:.1f}%)', fontsize=14)
    ax.set_xlim(0, 1)
    ax.legend(fontsize=9)

plt.suptitle(f'M Distribution by Dimension (N={N}, gamma={gamma}, {n_runs} runs each)', fontsize=16)
plt.tight_layout()
plt.savefig('F:/hypergraph_bistability/figures/M_histograms.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n[OK] Figure saved: M_histograms.png")

# ============================================================
# 图 2: P(low) vs d (核心图)
# ============================================================
plt.figure(figsize=(10, 6))

d_list = d_values
P_low_list = [stats[d]['P_low'] for d in d_list]
P_high_list = [stats[d]['P_high'] for d in d_list]

# 画点和线
plt.plot(d_list, P_low_list, 'o-', markersize=12, linewidth=2.5, 
         color='blue', label='P(low basin)')
plt.plot(d_list, P_high_list, 's--', markersize=10, linewidth=2, 
         color='red', alpha=0.7, label='P(high basin)')

# 50% 线
plt.axhline(y=50, color='gray', linestyle=':', alpha=0.7, linewidth=2)

# 标注
for i, d in enumerate(d_list):
    plt.annotate(f'{P_low_list[i]:.0f}%', (d, P_low_list[i]), 
                 textcoords="offset points", xytext=(0,10), ha='center', fontsize=10)

plt.xlabel('Dimension d', fontsize=14)
plt.ylabel('Probability %', fontsize=14)
plt.title('Basin Selection Probability vs Dimension', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.ylim(0, 100)

plt.savefig('F:/hypergraph_bistability/figures/P_low_vs_d.png', dpi=150, bbox_inches='tight')
plt.close()
print("[OK] Figure saved: P_low_vs_d.png")

# ============================================================
# 图 3: CV vs d (susceptibility)
# ============================================================
plt.figure(figsize=(10, 6))

cv_list = [stats[d]['cv'] for d in d_list]

plt.plot(d_list, cv_list, 'o-', markersize=12, linewidth=2.5, color='green')

# 找峰值
max_cv_idx = np.argmax(cv_list)
max_cv_d = d_list[max_cv_idx]
max_cv = cv_list[max_cv_idx]

# 标记峰值
plt.scatter([max_cv_d], [max_cv], s=200, c='red', zorder=5, marker='*')
plt.annotate(f'Peak: d={max_cv_d}\nCV={max_cv:.1f}%', 
             (max_cv_d, max_cv), textcoords="offset points", 
             xytext=(10,10), ha='left', fontsize=11,
             arrowprops=dict(arrowstyle='->', color='red'))

plt.xlabel('Dimension d', fontsize=14)
plt.ylabel('Coefficient of Variation %', fontsize=14)
plt.title('Susceptibility (CV) vs Dimension', fontsize=16)
plt.grid(True, alpha=0.3)

plt.savefig('F:/hypergraph_bistability/figures/CV_vs_d.png', dpi=150, bbox_inches='tight')
plt.close()
print("[OK] Figure saved: CV_vs_d.png")

# ============================================================
# 关键发现总结
# ============================================================
print("\n" + "="*60)
print("KEY FINDINGS")
print("="*60)

# 1. 临界区
print(f"\n1. Critical Zone:")
print(f"   - Maximum CV at d = {max_cv_d} (CV = {max_cv:.1f}%)")

# 2. 趋势
print(f"\n2. Trend Analysis:")
if P_low_list[0] < P_low_list[-1]:
    print(f"   - P(low) increases: {P_low_list[0]:.0f}% -> {P_low_list[-1]:.0f}%")
    print(f"   - Interpretation: Higher d favors LOW basin")
else:
    print(f"   - P(low) decreases: {P_low_list[0]:.0f}% -> {P_low_list[-1]:.0f}%")
    print(f"   - Interpretation: Higher d favors HIGH basin")

# 3. 相变点估计
print(f"\n3. Critical Point Estimate:")
# 找 P(low) 接近 50% 的点
for i, d in enumerate(d_list):
    if 40 <= P_low_list[i] <= 60:
        print(f"   - d_c ≈ {d} (P(low) = {P_low_list[i]:.0f}%)")
        break
else:
    # 外推
    for i in range(len(d_list) - 1):
        if (P_low_list[i] < 50 and P_low_list[i+1] > 50) or (P_low_list[i] > 50 and P_low_list[i+1] < 50):
            print(f"   - d_c ≈ between {d_list[i]} and {d_list[i+1]}")
            break
    else:
        print(f"   - No clear crossing in current range")

# 4. 双峰检测
print(f"\n4. Bimodality Check:")
for d in d_values:
    Ms = results[d]
    low_Ms = [m for m in Ms if m < THRESHOLD]
    high_Ms = [m for m in Ms if m >= THRESHOLD]
    
    if len(low_Ms) > 5 and len(high_Ms) > 5:
        print(f"   d={d}: BIMODAL (low={len(low_Ms)}, high={len(high_Ms)})")
    elif len(low_Ms) > n_runs * 0.6:
        print(f"   d={d}: LOW dominant ({len(low_Ms)}/20)")
    elif len(high_Ms) > n_runs * 0.6:
        print(f"   d={d}: HIGH dominant ({len(high_Ms)}/20)")
    else:
        print(f"   d={d}: MIXED (low={len(low_Ms)}, high={len(high_Ms)})")

print("\n" + "="*60)
print("DONE - All figures saved to figures/")
print("="*60)
