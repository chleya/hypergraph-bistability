"""
有限尺寸标度分析
1. K/N 归一化
2. Data collapse
3. 临界涨落量化
"""

import numpy as np
import random

def run_for_N_K(N, K, steps=800, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    
    class Hypergraph:
        def __init__(self, n_vertices):
            self.V = list(range(n_vertices))
            self.E = []
            self.s = {v: np.random.randn(4) for v in self.V}
            self.conflicts = set()
            self.faction = {}
            self.rejection_count = 0
            self.scar_node = None
            self.ALPHA = 0
            self.K = K
            
            for _ in range(max(4, N//4)):
                size = random.randint(2, min(5, N//3))
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
            return 1.0 / (1.0 + self.ALPHA * degree)
        
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
            changed = False
            avg_dist = self.get_avg_distance()
            
            if random.random() < 0.35 and self.E:
                e = random.choice(self.E)
                nodes = list(e)
                weights = np.array([self.get_influence(v) for v in nodes])
                weights = weights / weights.sum()
                x = np.random.choice(nodes, p=weights)
                
                w = len(self.V)
                self.V.append(w)
                self.s[w] = self.s[x] + np.random.randn(4) * 0.2
                new_e = frozenset([x, w])
                self.E.append(new_e)
                self.faction[new_e] = self.faction.get(e, random.randint(0, 2))
                changed = True
            
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
                        self.rejection_count += 1
                        self.conflicts.add((e1, e2))
                        self.conflicts.add((e2, e1))
                        
                        for v in e1:
                            self.s[v] += np.random.randn(4) * 0.3 * self.get_influence(v)
                        for v in e2:
                            self.s[v] -= np.random.randn(4) * 0.3 * self.get_influence(v)
                        
                        if faction1 != faction2:
                            pass
                        else:
                            self.faction[e1] = faction1
                            self.faction[e2] = (faction1 + 1) % 3
                    else:
                        new_e = frozenset(e1 | e2)
                        if new_e not in self.E:
                            self.E.remove(e1)
                            self.E.remove(e2)
                            self.E.append(new_e)
                            
                            shared = e1 & e2
                            if shared:
                                for v in shared:
                                    influence = self.get_influence(v)
                                    self.s[v] = (1 - influence) * self.s[v] + influence * np.mean([self.s[u] for u in shared], axis=0)
                            
                            self.faction[new_e] = (faction1 + faction2) % 3
                            changed = True
            
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
                    changed = True
            
            if random.random() < 0.15 and self.E:
                e = random.choice(self.E)
                if len(e) <= 2 and random.random() < 0.5:
                    self.E.remove(e)
                    if e in self.faction:
                        del self.faction[e]
                    changed = True
            
            return changed
        
        def inject_you_pulse(self):
            if not self.V: return
            
            influences = {v: self.get_influence(v) for v in self.V}
            if random.random() < 0.3:
                target = max(influences, key=influences.get)
            else:
                target = random.choice(self.V)
            
            self.scar_node = target
            self.s[target] += np.array([5.0, -5.0, 5.0, -5.0])
        
        def get_stats(self):
            n_edges = len(self.E)
            
            faction_counts = {}
            for e in self.E:
                f = self.faction.get(e, 0)
                faction_counts[f] = faction_counts.get(f, 0) + 1
            
            max_degree = max([self.get_node_degree(v) for v in self.V]) if self.V else 0
            
            max_faction_pct = max(faction_counts.values()) / n_edges * 100 if n_edges > 0 else 0
            n_factions = len([c for c in faction_counts.values() if c > 0])
            
            if n_edges > 0:
                probs = [c / n_edges for c in faction_counts.values() if c > 0]
                shannon_entropy = -sum(p * np.log(p + 1e-10) for p in probs)
            else:
                shannon_entropy = 0
            
            return {
                'n_edges': n_edges,
                'max_degree': max_degree,
                'max_faction_pct': max_faction_pct,
                'n_factions': n_factions,
                'shannon_entropy': shannon_entropy,
            }
        
        def run_with_timeline(self, steps):
            for t in range(1, steps + 1):
                self.apply_rules()
                
                if t % 50 == 0:
                    self.enforce_resource_constraint()
                
                if t % 50 == 0:
                    self.inject_you_pulse()
            
            return self.get_stats()
    
    H = Hypergraph(n_vertices=N)
    return H.run_with_timeline(steps=steps)


# 用归一化 K/N 扫描
# N = 15, 30, 50, 100
# K/N = 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0

N_values = [15, 30, 50, 100]
K_N_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
n_runs = 5

print("=" * 100)
print("有限尺寸标度分析 - K/N 归一化")
print("=" * 100)

results = []

for N in N_values:
    print(f"\n--- N = {N} ---")
    
    for ratio in K_N_ratios:
        K = int(N * ratio)
        if K < 2: K = 2
        
        max_pcts = []
        entropies = []
        
        for run in range(n_runs):
            seed = N * 1000 + run * 100 + int(ratio * 100)
            result = run_for_N_K(N, K, steps=800, seed=seed)
            max_pcts.append(result['max_faction_pct'])
            entropies.append(result['shannon_entropy'])
        
        avg_max_pct = np.mean(max_pcts)
        std_max_pct = np.std(max_pcts)
        avg_entropy = np.mean(entropies)
        
        results.append({
            'N': N,
            'K': K,
            'K/N': ratio,
            'max_pct': avg_max_pct,
            'std': std_max_pct,
            'entropy': avg_entropy,
        })
        
        print(f"K/N={ratio:.1f} (K={K}): max_pct={avg_max_pct:.1f}% +/- {std_max_pct:.1f}%, H={avg_entropy:.3f}")

# Data collapse 分析
print("\n" + "=" * 100)
print("=== Data Collapse 分析 ===")
print("=" * 100)

# 按 K/N 分组
print("\n按 K/N 归一化后的结果:")
print(f"{'K/N':>6} | {'N=15':>12} | {'N=30':>12} | {'N=50':>12} | {'N=100':>12}")
print("-" * 60)

for ratio in K_N_ratios:
    row = f"{ratio:>6.1f}"
    for N in N_values:
        # 找对应的结果
        for r in results:
            if r['N'] == N and abs(r['K/N'] - ratio) < 0.05:
                row += f" | {r['max_pct']:>10.1f}%"
                break
        else:
            row += " | " + " " * 10
    print(row)

# 临界涨落分析
print("\n" + "=" * 100)
print("=== 临界涨落 (Std Dev) ===")
print("=" * 100)

print(f"{'K/N':>6} | {'N=15':>10} | {'N=30':>10} | {'N=50':>10} | {'N=100':>10}")
print("-" * 55)

for ratio in K_N_ratios:
    row = f"{ratio:>6.1f}"
    for N in N_values:
        for r in results:
            if r['N'] == N and abs(r['K/N'] - ratio) < 0.05:
                row += f" | {r['std']:>9.1f}%"
                break
        else:
            row += " | " + " " * 8
    print(row)

# 找临界点附近
print("\n" + "=" * 100)
print("=== 寻找临界涨落峰 ===")
print("=" * 100)

for N in N_values:
    print(f"\nN = {N}:")
    N_results = [r for r in results if r['N'] == N]
    
    # 找最大 std
    max_std = max(N_results, key=lambda x: x['std'])
    min_pct = min(N_results, key=lambda x: x['max_pct'])
    
    print(f"  最大涨落: K/N = {max_std['K/N']:.1f}, std = {max_std['std']:.1f}%")
    print(f"  最小占比: K/N = {min_pct['K/N']:.1f}, pct = {min_pct['max_pct']:.1f}%")

# 理论分析
print("\n" + "=" * 100)
print("=== Scaling 假设检验 ===")
print("=" * 100)

# 假设 K_c / N 是一个常数
print("\n假设 K_c / N = 常数:")
for ratio in K_N_ratios:
    print(f"\nK/N = {ratio}:")
    for N in N_values:
        for r in results:
            if r['N'] == N and abs(r['K/N'] - ratio) < 0.05:
                print(f"  N={N}: {r['max_pct']:.1f}%")
                break
