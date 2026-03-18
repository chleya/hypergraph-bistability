"""
Switching Probability Analysis
==============================
测量跨 basin 切换概率 - barrier 高度的直接证据

方法：
- 在 low basin 初始化 → 看是否跳到 high
- 在 high basin 初始化 → 看是否掉到 low
- 在临界区（d≈24-32）：切换概率应该最高
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import os

os.makedirs('F:/hypergraph_bistability/figures', exist_ok=True)


def create_hypergraph(N, gamma, state_dim, initial_mode='random'):
    """创建超图，可指定初始模式"""
    
    K = int(gamma * N)
    
    class Hypergraph:
        def __init__(self):
            self.V = list(range(N))
            self.E = []
            self.s = {v: np.random.randn(state_dim) for v in self.V}
            self.conflicts = set()
            self.faction = {}
            self.K = K
            
            # 根据初始模式创建不同的初始结构
            if initial_mode == 'single_faction':
                # 单 faction - high basin
                e = frozenset(range(min(8, N)))
                self.E.append(e)
                self.faction[e] = 0
            
            elif initial_mode == 'multi_faction':
                # 多个小 faction - low basin
                for i in range(0, min(N, 10), 2):
                    e = frozenset([i, i+1] if i+1 < N else [i])
                    self.E.append(e)
                    self.faction[e] = i % 3
            
            else:  # random
                n_initial = max(4, N // 4)
                for _ in range(n_initial):
                    size = random.randint(2, min(5, N // 3))
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
    
    return Hypergraph()


def run_switching_experiment(N, gamma, state_dim, initial_mode, steps, seed):
    """运行切换实验"""
    
    random.seed(seed)
    np.random.seed(seed)
    
    H = create_hypergraph(N, gamma, state_dim, initial_mode)
    
    # 记录 M 轨迹
    M_history = [H.get_M()]
    
    for _ in range(steps):
        H.apply_rules()
        M_history.append(H.get_M())
    
    final_M = M_history[-1]
    return final_M, M_history


# ============================================================
# 实验设计
# ============================================================

print("="*60)
print("Switching Probability Analysis")
print("="*60)

N = 50
gamma = 0.35

# 关键维度：测试 d = 24, 32 (临界区)
d_values = [24, 32]

# 两种初始模式
modes = ['single_faction', 'multi_faction']  # high basin, low basin

n_runs = 5
steps = 1000

# 阈值判断
LOW_THRESHOLD = 0.52  # M < 0.52 = low basin

results = {}

for mode in modes:
    results[mode] = {}
    
    for d in d_values:
        print(f"\n[mode={mode}, d={d}] Running {n_runs} experiments...")
        
        final_Ms = []
        switches = 0
        
        for run in range(n_runs):
            seed = 3000 + run * 100 + d * 100 + (0 if mode == 'single_faction' else 1000)
            
            final_M, trajectory = run_switching_experiment(
                N=N, gamma=gamma, state_dim=d,
                initial_mode=mode, steps=steps, seed=seed
            )
            
            final_Ms.append(final_M)
            
            # 判断初始状态
            initial_M = trajectory[0]
            if mode == 'single_faction':
                # 初始是 high，看是否切换到 low
                if final_M < LOW_THRESHOLD:
                    switches += 1
            else:  # multi_faction
                # 初始是 low，看是否切换到 high
                if final_M >= LOW_THRESHOLD:
                    switches += 1
        
        # 统计
        mean_M = np.mean(final_Ms)
        std_M = np.std(final_Ms)
        switch_prob = switches / n_runs * 100
        
        results[mode][d] = {
            'M_mean': mean_M,
            'M_std': std_M,
            'switches': switches,
            'switch_prob': switch_prob,
            'final_Ms': final_Ms
        }
        
        print(f"  Final M: {mean_M:.3f} ± {std_M:.3f}")
        print(f"  Switching: {switches}/{n_runs} = {switch_prob:.1f}%")

# ============================================================
# 分析
# ============================================================

print("\n" + "="*60)
print("Analysis")
print("="*60)

print("\n--- Switching Probability Summary ---")
print(f"{'Mode':<16} | {'d':>4} | {'Final M':>8} | {'Switch %':>10}")
print("-" * 45)

for mode in modes:
    for d in d_values:
        r = results[mode][d]
        print(f"{mode:<16} | {d:>4} | {r['M_mean']:>8.3f} | {r['switch_prob']:>9.1f}%")

# 计算切换概率差异
print("\n--- Key Insight ---")

# 对于 single_faction (high basin 初始)
# P(high → low) 应该随 d 增加而增加（因为 low basin 变深）
high_to_low = [results['single_faction'][d]['switch_prob'] for d in d_values]

# 对于 multi_faction (low basin 初始)  
# P(low → high) 应该随 d 增加而减少
low_to_high = [results['multi_faction'][d]['switch_prob'] for d in d_values]

print(f"\nP(high → low): {high_to_low}")
print(f"P(low → high): {low_to_high}")

# 净效应
net_effect = [h - l for h, l in zip(high_to_low, low_to_high)]
print(f"Net (H→L - L→H): {net_effect}")

# ============================================================
# 绘图
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 图 1: 切换概率 vs d
ax1 = axes[0]
x = np.arange(len(d_values))
width = 0.35

bars1 = ax1.bar(x - width/2, high_to_low, width, label='high → low', color='red', alpha=0.7)
bars2 = ax1.bar(x + width/2, low_to_high, width, label='low → high', color='blue', alpha=0.7)

ax1.set_xlabel('Dimension d', fontsize=12)
ax1.set_ylabel('Switching Probability %', fontsize=12)
ax1.set_title('Switching Probability vs d', fontsize=14)
ax1.set_xticks(x)
ax1.set_xticklabels(d_values)
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bar, val in zip(bars1, high_to_low):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{val:.0f}%', ha='center', fontsize=9)
for bar, val in zip(bars2, low_to_high):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{val:.0f}%', ha='center', fontsize=9)

# 图 2: 最终 M 分布
ax2 = axes[1]

for i, mode in enumerate(modes):
    Ms_by_d = [results[mode][d]['M_mean'] for d in d_values]
    errs = [results[mode][d]['M_std'] for d in d_values]
    
    label = 'from high' if mode == 'single_faction' else 'from low'
    marker = 's' if mode == 'single_faction' else 'o'
    color = 'red' if mode == 'single_faction' else 'blue'
    
    ax2.errorbar(d_values, Ms_by_d, yerr=errs, fmt=marker, 
                 label=label, capsize=5, color=color, alpha=0.7)

ax2.axhline(y=LOW_THRESHOLD, color='gray', linestyle='--', alpha=0.5, label='threshold')
ax2.set_xlabel('Dimension d', fontsize=12)
ax2.set_ylabel('Final M', fontsize=12)
ax2.set_title('Final M: Initial Condition Dependence', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)

# 图 3: 净效应
ax3 = axes[2]
ax3.bar(d_values, net_effect, color='purple', alpha=0.7)
ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax3.set_xlabel('Dimension d', fontsize=12)
ax3.set_ylabel('Net Flow (H→L - L→H)', fontsize=12)
ax3.set_title('Net Basin Flow', fontsize=14)
ax3.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('F:/hypergraph_bistability/figures/switching_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n[OK] Figure saved: switching_analysis.png")

# ============================================================
# 关键发现
# ============================================================
print("\n" + "="*60)
print("KEY FINDINGS")
print("="*60)

print("\n1. Barrier Asymmetry:")
for i, d in enumerate(d_values):
    h2l = high_to_low[i]
    l2h = low_to_high[i]
    
    if h2l > l2h:
        print(f"   d={d}: HIGH basin less stable (H→L: {h2l}% > L→H: {l2h}%)")
    elif h2l < l2h:
        print(f"   d={d}: LOW basin less stable (L→H: {l2h}% > H→L: {h2l}%)")
    else:
        print(f"   d={d}: Symmetric ({h2l}% = {l2h}%)")

print("\n2. Trend Analysis:")
if high_to_low[-1] > high_to_low[0]:
    print(f"   P(high→low) increases with d: {high_to_low[0]}% → {high_to_low[-1]}%")
    print(f"   → Higher d makes HIGH basin less stable")
    
if low_to_high[-1] < low_to_high[0]:
    print(f"   P(low→high) decreases with d: {low_to_high[0]}% → {low_to_high[-1]}%")
    print(f"   → Higher d makes LOW basin more stable")

print("\n3. Critical Zone Evidence:")
if max(net_effect) > 10 or min(net_effect) < -10:
    print(f"   Significant net flow detected!")
    max_idx = np.argmax(net_effect)
    print(f"   → Maximum asymmetry at d = {d_values[max_idx]}")
else:
    print(f"   Net flow is small, may need more data")

print("\n" + "="*60)
print("DONE")
print("="*60)
