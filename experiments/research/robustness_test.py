"""
规则扰动实验 - 验证结果是否机制不变量
目标：测试 γ_c ≈ 0.35, M* ≈ 0.45 是否稳定
"""

import numpy as np
import random

# ============================================================
# 基础版本 (Original)
# ============================================================
def run_original(N, K, steps=800, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    
    class Hypergraph:
        def __init__(self, n_vertices):
            self.V = list(range(n_vertices))
            self.E = []
            self.s = {v: np.random.randn(4) for v in self.V}
            self.conflicts = set()
            self.faction = {}
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
                x = random.choice(nodes)  # 随机选择，不加权
                
                w = len(self.V)
                self.V.append(w)
                self.s[w] = self.s[x] + np.random.randn(4) * 0.2
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
                            self.s[v] += np.random.randn(4) * 0.3 * self.get_influence(v)
                        for v in e2:
                            self.s[v] -= np.random.randn(4) * 0.3 * self.get_influence(v)
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
        
        def get_stats(self):
            n_edges = len(self.E)
            faction_counts = {}
            for e in self.E:
                f = self.faction.get(e, 0)
                faction_counts[f] = faction_counts.get(f, 0) + 1
            
            max_faction_pct = max(faction_counts.values()) / n_edges * 100 if n_edges > 0 else 0
            return max_faction_pct
    
    H = Hypergraph(N)
    for t in range(1, steps + 1):
        H.apply_rules()
        if t % 50 == 0:
            H.enforce_resource_constraint()
    
    return H.get_stats()


# ============================================================
# 改动1: Preferential Attachment
# ============================================================
def run_preferential(N, K, steps=800, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    
    class HypergraphPA:
        def __init__(self, n_vertices):
            self.V = list(range(n_vertices))
            self.E = []
            self.s = {v: np.random.randn(4) for v in self.V}
            self.conflicts = set()
            self.faction = {}
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
            # Preferential: higher degree = higher influence
            return 1 + degree  # 改成这种形式
        
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
            
            # 支化 - Preferential Attachment
            if random.random() < 0.35 and self.E:
                e = random.choice(self.E)
                nodes = list(e)
                # 用 degree 作为权重
                degrees = np.array([max(1, self.get_node_degree(v)) for v in nodes])
                probs = degrees / degrees.sum()
                x = np.random.choice(nodes, p=probs)  # 优先选高度数节点
                
                w = len(self.V)
                self.V.append(w)
                self.s[w] = self.s[x] + np.random.randn(4) * 0.2
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
                            self.s[v] += np.random.randn(4) * 0.3
                        for v in e2:
                            self.s[v] -= np.random.randn(4) * 0.3
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
        
        def get_stats(self):
            n_edges = len(self.E)
            faction_counts = {}
            for e in self.E:
                f = self.faction.get(e, 0)
                faction_counts[f] = faction_counts.get(f, 0) + 1
            
            max_faction_pct = max(faction_counts.values()) / n_edges * 100 if n_edges > 0 else 0
            return max_faction_pct
    
    H = HypergraphPA(N)
    for t in range(1, steps + 1):
        H.apply_rules()
        if t % 50 == 0:
            H.enforce_resource_constraint()
    
    return H.get_stats()


# ============================================================
# 改动2: 节点聚类定义阵营
# ============================================================
def run_node_clustering(N, K, steps=800, seed=42):
    """用节点状态聚类来定义阵营，而不是超边"""
    random.seed(seed)
    np.random.seed(seed)
    
    class HypergraphNodeCluster:
        def __init__(self, n_vertices):
            self.V = list(range(n_vertices))
            self.E = []
            self.s = {v: np.random.randn(4) for v in self.V}
            self.conflicts = set()
            self.node_faction = {}  # 用节点状态聚类
            self.K = K
            
            for _ in range(max(4, N//4)):
                size = random.randint(2, min(5, N//3))
                e = frozenset(random.sample(self.V, size))
                self.E.append(e)
        
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
        
        def assign_factions(self):
            """用K-means聚类来分配阵营"""
            if len(self.V) < 3: 
                self.node_faction = {v: 0 for v in self.V}
                return
            
            # 简单聚类：用状态向量的符号
            states = np.array([self.s[v] for v in self.V])
            
            # 用状态距离聚类
            centroids = []
            assignments = []
            for v in self.V:
                s = self.s[v]
                assigned = False
                for i, c in enumerate(centroids):
                    if np.linalg.norm(s - c) < 1.0:
                        assignments.append(i)
                        assigned = True
                        break
                if not assigned:
                    centroids.append(s)
                    assignments.append(len(centroids) - 1)
            
            # 限制阵营数
            max_factions = 3
            if len(centroids) > max_factions:
                # 取最大的三个簇
                counts = {}
                for a in assignments:
                    counts[a] = counts.get(a, 0) + 1
                top_factions = sorted(counts.keys(), key=lambda x: counts[x], reverse=True)[:max_factions]
                for v, a in zip(self.V, assignments):
                    self.node_faction[v] = top_factions.index(a) if a in top_factions else 0
            else:
                for v, a in zip(self.V, assignments):
                    self.node_faction[v] = a
        
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
        
        def apply_rules(self):
            avg_dist = self.get_avg_distance()
            
            # 每20步重新计算阵营
            pass  # 在主循环中处理
            
            # 支化
            if random.random() < 0.35 and self.E:
                e = random.choice(self.E)
                nodes = list(e)
                x = random.choice(nodes)
                
                w = len(self.V)
                self.V.append(w)
                self.s[w] = self.s[x] + np.random.randn(4) * 0.2
                new_e = frozenset([x, w])
                self.E.append(new_e)
            
            # 融合 - 用节点阵营
            if random.random() < 0.3 and len(self.E) >= 2:
                e1 = random.choice(self.E)
                e2 = random.choice(self.E)
                
                if e1 != e2 and len(e1 & e2) >= 1:
                    # 用节点平均状态判断阵营
                    c1 = np.mean([self.s[u] for u in e1], axis=0)
                    c2 = np.mean([self.s[u] for u in e2], axis=0)
                    
                    distance = np.linalg.norm(c1 - c2)
                    effective_threshold = 0.8 * avg_dist - 0.3  # 固定惩罚
                    
                    if distance > effective_threshold:
                        self.conflicts.add((e1, e2))
                        for v in e1:
                            self.s[v] += np.random.randn(4) * 0.3
                        for v in e2:
                            self.s[v] -= np.random.randn(4) * 0.3
                    else:
                        new_e = frozenset(e1 | e2)
                        if new_e not in self.E:
                            self.E.remove(e1)
                            self.E.remove(e2)
                            self.E.append(new_e)
            
            # 分裂
            if random.random() < 0.25 and self.E:
                e = random.choice(self.E)
                if len(e) >= 5:
                    lst = list(e)
                    mid = len(lst) // 2
                    e_left = frozenset(lst[:mid])
                    e_right = frozenset(lst[mid:])
                    self.E.remove(e)
                    self.E.append(e_left)
                    self.E.append(e_right)
            
            # 消除
            if random.random() < 0.15 and self.E:
                e = random.choice(self.E)
                if len(e) <= 2 and random.random() < 0.5:
                    self.E.remove(e)
        
        def get_stats(self):
            # 重新计算阵营
            self.assign_factions()
            
            n_edges = len(self.E)
            
            # 按节点阵营统计
            node_faction_counts = {}
            for v, f in self.node_faction.items():
                node_faction_counts[f] = node_faction_counts.get(f, 0) + 1
            
            max_faction_pct = max(node_faction_counts.values()) / len(self.V) * 100 if self.V else 0
            return max_faction_pct
    
    H = HypergraphNodeCluster(N)
    H.assign_factions()
    
    for t in range(1, steps + 1):
        H.apply_rules()
        if t % 50 == 0:
            H.enforce_resource_constraint()
        if t % 20 == 0:
            H.assign_factions()
    
    return H.get_stats()


# ============================================================
# 改动3: 异步更新
# ============================================================
def run_async(N, K, steps=800, seed=42):
    """异步更新"""
    random.seed(seed)
    np.random.seed(seed)
    
    class HypergraphAsync:
        def __init__(self, n_vertices):
            self.V = list(range(n_vertices))
            self.E = []
            self.s = {v: np.random.randn(4) for v in self.V}
            self.conflicts = set()
            self.faction = {}
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
        
        def apply_rules_async(self):
            """每次只更新一个规则（异步）"""
            avg_dist = self.get_avg_distance()
            
            r = random.random()
            
            # 支化
            if r < 0.15 and self.E:
                e = random.choice(self.E)
                nodes = list(e)
                x = random.choice(nodes)
                
                w = len(self.V)
                self.V.append(w)
                self.s[w] = self.s[x] + np.random.randn(4) * 0.2
                new_e = frozenset([x, w])
                self.E.append(new_e)
                self.faction[new_e] = self.faction.get(e, random.randint(0, 2))
            
            # 融合
            elif r < 0.45 and len(self.E) >= 2:
                e1 = random.choice(self.E)
                e2 = random.choice(self.E)
                
                if e1 != e2 and len(e1 & e2) >= 1:
                    faction1 = self.faction.get(e1, 0)
                    faction2 = self.faction.get(e2, 0)
                    faction_penalty = 0.3 if faction1 != faction2 else 0.0
                    
                    c1 = np.mean([self.s[u] for u in e1], axis=0)
                    c2 = np.mean([self.s[u] for u in e2], axis=0)
                    
                    distance = np.linalg.norm(c1 - c2)
                    effective_threshold = 0.8 * avg_dist - faction_penalty
                    
                    if distance > effective_threshold:
                        self.conflicts.add((e1, e2))
                        for v in e1:
                            self.s[v] += np.random.randn(4) * 0.3 * self.get_influence(v)
                        for v in e2:
                            self.s[v] -= np.random.randn(4) * 0.3 * self.get_influence(v)
                    else:
                        new_e = frozenset(e1 | e2)
                        if new_e not in self.E:
                            self.E.remove(e1)
                            self.E.remove(e2)
                            self.E.append(new_e)
                            self.faction[new_e] = (faction1 + faction2) % 3
            
            # 分裂
            elif r < 0.60 and self.E:
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
            elif r < 0.70 and self.E:
                e = random.choice(self.E)
                if len(e) <= 2 and random.random() < 0.5:
                    self.E.remove(e)
                    if e in self.faction:
                        del self.faction[e]
        
        def get_stats(self):
            n_edges = len(self.E)
            faction_counts = {}
            for e in self.E:
                f = self.faction.get(e, 0)
                faction_counts[f] = faction_counts.get(f, 0) + 1
            
            max_faction_pct = max(faction_counts.values()) / n_edges * 100 if n_edges > 0 else 0
            return max_faction_pct
    
    H = HypergraphAsync(N)
    for t in range(1, steps + 1):
        H.apply_rules_async()
        if t % 50 == 0:
            H.enforce_resource_constraint()
    
    return H.get_stats()


# ============================================================
# 运行实验
# ============================================================
N = 50
gammas = [0.2, 0.3, 0.35, 0.4, 0.5]
n_runs = 5

print("=" * 80)
print("规则扰动实验 - 验证机制不变量")
print("=" * 80)

results = []

# 测试不同变体
variants = [
    ("Original", run_original),
    ("Preferential", run_preferential),
    ("NodeCluster", run_node_clustering),
    ("Async", run_async),
]

for variant_idx, (variant_name, run_fn) in enumerate(variants):
    print(f"\n--- {variant_name} ---")
    
    for gamma in gammas:
        K = int(N * gamma)
        if K < 2: K = 2
        
        max_pcts = []
        for run in range(n_runs):
            seed = 42 + variant_idx * 10000 + run * 100 + int(gamma * 1000)
            M = run_fn(N, K, steps=800, seed=seed)
            max_pcts.append(M)
        
        avg_M = np.mean(max_pcts)
        std_M = np.std(max_pcts)
        
        results.append({
            'variant': variant_name,
            'gamma': gamma,
            'K': K,
            'M': avg_M,
            'std': std_M,
        })
        
        print(f"gamma={gamma:.2f} (K={K}): M={avg_M:.1f}% +/- {std_M:.1f}%")

# 总结
print("\n" + "=" * 80)
print("=== 机制不变量检验 ===")
print("=" * 80)

print("\n各变体的 M vs gamma:")
print(f"{'Variant':>15} | {'gamma=0.2':>10} | {'gamma=0.3':>10} | {'gamma=0.35':>10} | {'gamma=0.4':>10} | {'gamma=0.5':>10}")
print("-" * 75)

for variant_name, _ in variants:
    row = f"{variant_name:>15}"
    for gamma in [0.2, 0.3, 0.35, 0.4, 0.5]:
        for r in results:
            if r['variant'] == variant_name and abs(r['gamma'] - gamma) < 0.01:
                row += f" | {r['M']:>9.1f}%"
                break
        else:
            row += " | " + " " * 8
    print(row)

# 检查是否保持稳定
print("\n=== 不变性分析 ===")
for gamma in [0.3, 0.35, 0.4]:
    gamma_results = [r for r in results if abs(r['gamma'] - gamma) < 0.01]
    Ms = [r['M'] for r in gamma_results]
    print(f"gamma={gamma}: M的范围 = {min(Ms):.1f}% - {max(Ms):.1f}%, 均值 = {np.mean(Ms):.1f}%")
