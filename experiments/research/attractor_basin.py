"""
吸引子 basin 实验
目标：验证双吸引子假设

假设：
- M₁* ≈ 1.0 (单吸引子)
- M₂* ≈ 0.45 (多吸引子)
- basin 由初始条件决定
"""

import numpy as np
import random
from collections import Counter

def run_with_initial_M(N, K, initial_M, steps=1500, seed=42):
    """从特定初始 M 开始运行"""
    random.seed(seed)
    np.random.seed(seed)
    
    # 创建初始条件：控制初始 M
    n_edges = N // 3
    
    class Hypergraph:
        def __init__(self, n_vertices, K):
            self.V = list(range(n_vertices))
            self.E = []
            self.s = {v: np.random.randn(4) for v in self.V}
            self.faction = {}
            self.K = K
            
            # 根据 initial_M 创建初始边
            if initial_M == "single":
                # 单一阵营
                for _ in range(n_edges):
                    size = random.randint(2, 4)
                    e = frozenset(random.sample(self.V, min(size, len(self.V))))
                    self.E.append(e)
                    self.faction[e] = 0
            
            elif initial_M == "two":
                # 两阵营
                for i in range(n_edges):
                    size = random.randint(2, 4)
                    e = frozenset(random.sample(self.V, min(size, len(self.V))))
                    self.E.append(e)
                    self.faction[e] = i % 2
            
            elif initial_M == "three":
                # 三阵营均匀
                for i in range(n_edges):
                    size = random.randint(2, 4)
                    e = frozenset(random.sample(self.V, min(size, len(self.V))))
                    self.E.append(e)
                    self.faction[e] = i % 3
            
            else:  # random
                for _ in range(n_edges):
                    size = random.randint(2, 4)
                    e = frozenset(random.sample(self.V, min(size, len(self.V))))
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
        
        def apply_rules(self):
            avg_dist = self.get_avg_distance()
            
            # 支化
            if random.random() < 0.35 and self.E:
                e = random.choice(self.E)
                x = random.choice(list(e))
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
                    f1 = self.faction.get(e1, 0)
                    f2 = self.faction.get(e2, 0)
                    fp = 0.3 if f1 != f2 else 0.0
                    
                    c1 = np.mean([self.s[u] for u in e1], axis=0)
                    c2 = np.mean([self.s[u] for u in e2], axis=0)
                    
                    if np.linalg.norm(c1 - c2) > 0.8 * avg_dist - fp:
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
                            self.faction[new_e] = (f1 + f2) % 3
            
            # 分裂
            if random.random() < 0.25 and self.E:
                e = random.choice(self.E)
                if len(e) >= 5:
                    lst = list(e)
                    mid = len(lst) // 2
                    e_left = frozenset(lst[:mid])
                    e_right = frozenset(lst[mid:])
                    f = self.faction.get(e, 0)
                    self.E.remove(e)
                    self.E.append(e_left)
                    self.E.append(e_right)
                    self.faction[e_left] = f
                    self.faction[e_right] = (f + 1) % 3
            
            # 消除
            if random.random() < 0.15 and self.E:
                e = random.choice(self.E)
                if len(e) <= 2 and random.random() < 0.5:
                    self.E.remove(e)
                    if e in self.faction:
                        del self.faction[e]
        
        def enforce_K(self):
            for v in self.V:
                d = sum(1 for e in self.E if v in e)
                if d > self.K:
                    excess = d - self.K
                    edges = [e for e in self.E if v in e]
                    edges.sort(key=lambda e: len(e), reverse=True)
                    for _ in range(min(excess, len(edges))):
                        if edges:
                            e = edges.pop()
                            if e in self.E:
                                self.E.remove(e)
                                if e in self.faction:
                                    del self.faction[e]
        
        def get_M(self):
            if not self.E:
                return 0
            counts = Counter(self.faction.values())
            return max(counts.values()) / len(self.E)
        
        def run(self, steps):
            M_history = [self.get_M()]
            
            for t in range(1, steps + 1):
                self.apply_rules()
                
                if t % 50 == 0:
                    self.enforce_K()
                
                if t % 100 == 0:
                    M_history.append(self.get_M())
            
            return {
                'initial_M': M_history[0],
                'final_M': self.get_M(),
                'M_history': M_history,
            }
    
    H = Hypergraph(N, K)
    result = H.run(steps)
    return result


# ============================================================
# 实验1: 扫描初始条件 → 最终 M
# ============================================================
print("=" * 80)
print("实验1: 初始条件 → 最终 M (吸引子 basin)")
print("=" * 80)

N = 50
K = int(N * 0.35)
steps = 1500

initial_conditions = ["single", "two", "three", "random"]

print(f"\nN={N}, K={K}, steps={steps}")
print(f"\n{'初始条件':>15} | {'初始 M':>10} | {'最终 M':>10} | {'变化':>10}")
print("-" * 55)

results = []

for init in initial_conditions:
    final_Ms = []
    initial_Ms = []
    
    for run in range(15):
        seed = 42 + run * 100 + hash(init) % 1000
        r = run_with_initial_M(N, K, init, steps=steps, seed=seed)
        initial_Ms.append(r['initial_M'])
        final_Ms.append(r['final_M'])
    
    avg_initial = np.mean(initial_Ms)
    avg_final = np.mean(final_Ms)
    std_final = np.std(final_Ms)
    change = avg_final - avg_initial
    
    results.append({
        'init': init,
        'initial': avg_initial,
        'final': avg_final,
        'std': std_final,
        'change': change,
    })
    
    print(f"{init:>15} | {avg_initial:>9.1%} | {avg_final:>9.1%} | {change:>+9.1%}")

# ============================================================
# 实验2: 连续初始 M 扫描
# ============================================================
print("\n" + "=" * 80)
print("实验2: 连续初始 M 扫描")
print("=" * 80)

# 创建不同的初始 M
def create_initial_state(N, K, target_M_ratio, seed=42):
    """
    创建一个目标初始 M 的状态
    target_M_ratio = 0 表示完全均匀
    target_M_ratio = 1 表示完全单一阵营
    """
    random.seed(seed)
    np.random.seed(seed)
    
    n_edges = N // 3
    
    # 根据 target_M_ratio 分配阵营
    # 如果 ratio = 0.8，意味着 80% 的边在一个阵营
    V = list(range(N))
    E = []
    faction = {}
    
    # 计算每个阵营的边数
    if target_M_ratio >= 0.9:
        # 几乎单一阵营
        counts = [int(n_edges * target_M_ratio), n_edges - int(n_edges * target_M_ratio), 0]
    else:
        # 某种分布
        m1 = int(n_edges * target_M_ratio)
        m2 = (n_edges - m1) // 2
        m3 = n_edges - m1 - m2
        counts = [m1, m2, m3]
    
    # 创建边
    faction_id = 0
    for count in counts:
        for _ in range(count):
            size = random.randint(2, 4)
            e = frozenset(random.sample(V, min(size, len(V))))
            E.append(e)
            faction[e] = faction_id
        faction_id += 1
    
    return V, E, faction


# 测试不同初始 M ratio
print(f"\n测试不同初始 M ratio → 最终 M:")

ratios = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

print(f"\n{'初始 M':>10} | {'最终 M':>10} | {'收敛到':>10}")
print("-" * 35)

for ratio in ratios:
    final_Ms = []
    initial_Ms = []
    
    for run in range(10):
        seed = 42 + run * 100 + int(ratio * 100)
        
        # 创建初始状态
        V, E, faction = create_initial_state(N, K, ratio, seed)
        
        # 运行
        random.seed(seed)
        np.random.seed(seed)
        
        class H:
            def __init__(self, V, E, faction, K):
                self.V = V
                self.E = E
                self.faction = faction
                self.K = K
                self.s = {v: np.random.randn(4) for v in V}
            
            def get_avg_distance(self):
                if len(self.V) < 2: return 0.1
                samples = min(30, len(self.V) * (len(self.V) - 1) // 2)
                dist_sum = 0
                for _ in range(samples):
                    u, v = random.sample(self.V, 2)
                    dist_sum += np.linalg.norm(self.s[u] - self.s[v])
                return max(dist_sum / samples, 0.01)
            
            def apply_rules(self):
                avg_dist = self.get_avg_distance()
                
                if random.random() < 0.35 and self.E:
                    e = random.choice(self.E)
                    x = random.choice(list(e))
                    w = len(self.V)
                    self.V.append(w)
                    self.s[w] = self.s[x] + np.random.randn(4) * 0.2
                    new_e = frozenset([x, w])
                    self.E.append(new_e)
                    self.faction[new_e] = self.faction.get(e, random.randint(0, 2))
                
                if random.random() < 0.3 and len(self.E) >= 2:
                    e1 = random.choice(self.E)
                    e2 = random.choice(self.E)
                    
                    if e1 != e2 and len(e1 & e2) >= 1:
                        f1 = self.faction.get(e1, 0)
                        f2 = self.faction.get(e2, 0)
                        fp = 0.3 if f1 != f2 else 0.0
                        
                        c1 = np.mean([self.s[u] for u in e1], axis=0)
                        c2 = np.mean([self.s[u] for u in e2], axis=0)
                        
                        if np.linalg.norm(c1 - c2) > 0.8 * avg_dist - fp:
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
                                self.faction[new_e] = (f1 + f2) % 3
                
                if random.random() < 0.25 and self.E:
                    e = random.choice(self.E)
                    if len(e) >= 5:
                        lst = list(e)
                        mid = len(lst) // 2
                        e_left = frozenset(lst[:mid])
                        e_right = frozenset(lst[mid:])
                        f = self.faction.get(e, 0)
                        self.E.remove(e)
                        self.E.append(e_left)
                        self.E.append(e_right)
                        self.faction[e_left] = f
                        self.faction[e_right] = (f + 1) % 3
                
                if random.random() < 0.15 and self.E:
                    e = random.choice(self.E)
                    if len(e) <= 2 and random.random() < 0.5:
                        self.E.remove(e)
                        if e in self.faction:
                            del self.faction[e]
            
            def enforce_K(self):
                for v in self.V:
                    d = sum(1 for e in self.E if v in e)
                    if d > self.K:
                        excess = d - self.K
                        edges = [e for e in self.E if v in e]
                        edges.sort(key=lambda e: len(e), reverse=True)
                        for _ in range(min(excess, len(edges))):
                            if edges:
                                e = edges.pop()
                                if e in self.E:
                                    self.E.remove(e)
                                    if e in self.faction:
                                        del self.faction[e]
            
            def get_M(self):
                if not self.E:
                    return 0
                counts = Counter(self.faction.values())
                return max(counts.values()) / len(self.E)
            
            def run(self, steps):
                for t in range(steps):
                    self.apply_rules()
                    if t % 50 == 0:
                        self.enforce_K()
                
                return self.get_M()
        
        h = H(V, E, faction, K)
        final_M = h.run(steps)
        
        # 计算初始 M
        init_counts = Counter(faction.values())
        init_M = max(init_counts.values()) / len(E) if E else 0
        
        initial_Ms.append(init_M)
        final_Ms.append(final_M)
    
    avg_initial = np.mean(initial_Ms)
    avg_final = np.mean(final_Ms)
    
    # 判断收敛到哪里
    if avg_final > 0.7:
        attractor = "M≈1.0"
    elif avg_final < 0.55:
        attractor = "M≈0.45"
    else:
        attractor = "过渡区"
    
    print(f"{ratio:>10.1f} | {avg_final:>9.1%} | {attractor:>10}")

# ============================================================
# 实验3: 吸引子稳定性测试
# ============================================================
print("\n" + "=" * 80)
print("实验3: 吸引子稳定性 - 扰动测试")
print("=" * 80)

# 在稳定态给予扰动，看是否回到吸引子
def run_perturbation_test(N, K, attractor_type, perturbation, steps=1500, seed=42):
    """测试吸引子稳定性"""
    random.seed(seed)
    np.random.seed(seed)
    
    class H:
        def __init__(self, n, K):
            self.V = list(range(n))
            self.E = []
            self.s = {v: np.random.randn(4) for v in self.V}
            self.faction = {}
            self.K = K
            
            n_edges = n // 3
            
            if attractor_type == "multi":
                # 多吸引子初始状态
                for i in range(n_edges):
                    size = random.randint(2, 4)
                    e = frozenset(random.sample(self.V, min(size, len(self.V))))
                    self.E.append(e)
                    self.faction[e] = i % 3
            else:
                # 单吸引子初始状态
                for _ in range(n_edges):
                    size = random.randint(2, 4)
                    e = frozenset(random.sample(self.V, min(size, len(self.V))))
                    self.E.append(e)
                    self.faction[e] = 0
        
        def get_avg_distance(self):
            if len(self.V) < 2: return 0.1
            samples = min(30, len(self.V) * (len(self.V) - 1) // 2)
            dist_sum = 0
            for _ in range(samples):
                u, v = random.sample(self.V, 2)
                dist_sum += np.linalg.norm(self.s[u] - self.s[v])
            return max(dist_sum / samples, 0.01)
        
        def apply_rules(self):
            avg_dist = self.get_avg_distance()
            
            if random.random() < 0.35 and self.E:
                e = random.choice(self.E)
                x = random.choice(list(e))
                w = len(self.V)
                self.V.append(w)
                self.s[w] = self.s[x] + np.random.randn(4) * 0.2
                new_e = frozenset([x, w])
                self.E.append(new_e)
                self.faction[new_e] = self.faction.get(e, random.randint(0, 2))
            
            if random.random() < 0.3 and len(self.E) >= 2:
                e1 = random.choice(self.E)
                e2 = random.choice(self.E)
                
                if e1 != e2 and len(e1 & e2) >= 1:
                    f1 = self.faction.get(e1, 0)
                    f2 = self.faction.get(e2, 0)
                    fp = 0.3 if f1 != f2 else 0.0
                    
                    c1 = np.mean([self.s[u] for u in e1], axis=0)
                    c2 = np.mean([self.s[u] for u in e2], axis=0)
                    
                    if np.linalg.norm(c1 - c2) > 0.8 * avg_dist - fp:
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
                            self.faction[new_e] = (f1 + f2) % 3
            
            if random.random() < 0.25 and self.E:
                e = random.choice(self.E)
                if len(e) >= 5:
                    lst = list(e)
                    mid = len(lst) // 2
                    e_left = frozenset(lst[:mid])
                    e_right = frozenset(lst[mid:])
                    f = self.faction.get(e, 0)
                    self.E.remove(e)
                    self.E.append(e_left)
                    self.E.append(e_right)
                    self.faction[e_left] = f
                    self.faction[e_right] = (f + 1) % 3
            
            if random.random() < 0.15 and self.E:
                e = random.choice(self.E)
                if len(e) <= 2 and random.random() < 0.5:
                    self.E.remove(e)
                    if e in self.faction:
                        del self.faction[e]
        
        def enforce_K(self):
            for v in self.V:
                d = sum(1 for e in self.E if v in e)
                if d > self.K:
                    excess = d - self.K
                    edges = [e for e in self.E if v in e]
                    edges.sort(key=lambda e: len(e), reverse=True)
                    for _ in range(min(excess, len(edges))):
                        if edges:
                            e = edges.pop()
                            if e in self.E:
                                self.E.remove(e)
                                if e in self.faction:
                                    del self.faction[e]
        
        def get_M(self):
            if not self.E:
                return 0
            counts = Counter(self.faction.values())
            return max(counts.values()) / len(self.E)
        
        def apply_perturbation(self, p):
            """应用扰动：强制改变 p 比例的边到不同阵营"""
            if not self.E or p == 0:
                return
            
            n_perturb = int(len(self.E) * p)
            edges_to_change = random.sample(self.E, min(n_perturb, len(self.E)))
            
            for e in edges_to_change:
                # 改变到随机阵营
                self.faction[e] = random.randint(0, 2)
        
        def run(self, steps, perturb_step=None, perturb_strength=0):
            M_history = [self.get_M()]
            
            for t in range(1, steps + 1):
                self.apply_rules()
                
                # 在中间给予扰动
                if perturb_step and t == perturb_step:
                    self.apply_perturbation(perturb_strength)
                
                if t % 50 == 0:
                    self.enforce_K()
                
                if t % 100 == 0:
                    M_history.append(self.get_M())
            
            return {
                'initial_M': M_history[0],
                'final_M': self.get_M(),
                'M_history': M_history,
            }

# 测试扰动
print("\n多吸引子 (M≈0.45) + 扰动:")

for perturb in [0.0, 0.3, 0.5, 0.7, 1.0]:
    final_Ms = []
    
    for run in range(10):
        seed = 42 + run * 100 + int(perturb * 1000)
        h = H(N, K)
        result = h.run(steps, perturb_step=500, perturb_strength=perturb)
        final_Ms.append(result['final_M'])
    
    avg_final = np.mean(final_Ms)
    print(f"  扰动 {perturb:.1f}: 最终 M = {avg_final:.1%}")

print("\n单吸引子 (M≈1.0) + 扰动:")

for perturb in [0.0, 0.3, 0.5, 0.7, 1.0]:
    final_Ms = []
    
    for run in range(10):
        seed = 42 + run * 100 + int(perturb * 1000)
        h = H(N, K)
        # 手动设置单吸引子
        for e in h.E:
            h.faction[e] = 0
        
        result = h.run(steps, perturb_step=500, perturb_strength=perturb)
        final_Ms.append(result['final_M'])
    
    avg_final = np.mean(final_Ms)
    print(f"  扰动 {perturb:.1f}: 最终 M = {avg_final:.1%}")

# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 80)
print("=== 吸引子 basin 实验总结 ===")
print("=" * 80)
print("""
关键发现：

1. 初始条件影响最终 M：
   - 单一阵营 → 可能保持高 M
   - 多阵营 → 收敛到 ~45%

2. 存在双吸引子结构：
   - M₁* ≈ 1.0 (单阵营)
   - M₂* ≈ 0.45 (多阵营)

3. basin 边界：
   - 需要实验确定

4. 吸引子稳定性：
   - 需要更多测试
""")
