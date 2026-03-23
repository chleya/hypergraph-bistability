"""
Step 1: M(d) 曲线 with Error Bars
================================
目标：确认 M*(d) 是否收敛到 ~0.45

固定：
- N = 100
- γ = 0.35
- capacity 规则

变量：d ∈ {2, 4, 8, 16, 32}
每个 d 跑 5 次
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import os

os.makedirs('F:/hypergraph_bistability/figures', exist_ok=True)

# ============================================================
# 基础超图模型（定义明确版本）
# ============================================================

class Hypergraph:
    def __init__(self, n_vertices=100, K=35, state_dim=4, seed=42):
        """
        初始化
        - n_vertices: N
        - K: capacity = γ * N
        - state_dim: d
        """
        self.N = n_vertices
        self.K = K
        self.d = state_dim
        
        random.seed(seed)
        np.random.seed(seed)
        
        # 节点状态
        self.V = list(range(n_vertices))
        self.s = {v: np.random.randn(state_dim) for v in self.V}
        
        # 超边（初始：每个节点独立 → 慢慢融合）
        self.E = []
        
        # 初始创建一些超边（避免从 0 开始）
        n_initial = max(10, n_vertices // 5)
        for _ in range(n_initial):
            size = random.randint(2, min(6, n_vertices // 3))
            if size >= 2:
                e = frozenset(random.sample(self.V, size))
                self.E.append(e)
        
        # 阵营定义
        self.faction = {}
        for e in self.E:
            self.faction[e] = random.randint(0, 2)
    
    def get_node_degree(self, v):
        return sum(1 for e in self.E if v in e)
    
    def get_influence(self, v):
        """影响力 = 1 / degree"""
        degree = self.get_node_degree(v)
        return 1.0 / (1 + degree * 0.1)  # 软惩罚
    
    def get_factions(self):
        """找连通分量（阵营）"""
        if not self.E:
            return [[v] for v in self.V]
        
        # 构建邻接表
        adj = {v: set() for v in self.V}
        for e in self.E:
            nodes = list(e)
            for i in range(len(nodes)):
                for j in range(i+1, len(nodes)):
                    adj[nodes[i]].add(nodes[j])
                    adj[nodes[j]].add(nodes[i])
        
        # BFS 找连通分量
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
        
        return factions
    
    def get_M(self):
        """
        M 定义：最大超边占比
        M = max_hyperedge_size / N_initial
        这代表"中心化程度"——越大越集中
        """
        if self.E:
            # 只考虑包含原始节点的超边
            original_nodes = set(range(self.N))
            max_size = 0
            for e in self.E:
                # 只计算原始节点
                original_in_e = len(e & original_nodes)
                max_size = max(max_size, original_in_e)
            
            M = max_size / self.N
            # 断言：确保 M 在 [0, 1]
            assert 0 <= M <= 1, f"M out of range: {M}, max_size={max_size}, N={self.N}"
            return M
        return 0.0
    
    def enforce_capacity(self):
        """强制容量约束：每个节点最多 K 个连接"""
        for v in self.V:
            degree = self.get_node_degree(v)
            if degree > self.K:
                excess = degree - self.K
                edges_with_v = [e for e in self.E if v in e]
                # 按超边大小排序，优先移除小的
                edges_with_v.sort(key=lambda e: len(e))
                for _ in range(min(excess, len(edges_with_v))):
                    if edges_with_v:
                        e = edges_with_v[0]
                        if e in self.E:
                            self.E.remove(e)
                            if e in self.faction:
                                del self.faction[e]
    
    def apply_rules(self, steps=200):
        """运行 steps 步"""
        avg_distance = self._get_avg_distance()
        
        for _ in range(steps):
            # 1. 支化 (Growth) - 35% 概率
            if random.random() < 0.35 and self.E:
                e = random.choice(self.E)
                nodes = list(e)
                x = random.choice(nodes)
                
                # 新节点
                w = len(self.V)
                self.V.append(w)
                self.s[w] = self.s[x] + np.random.randn(self.d) * 0.2
                
                # 新超边
                new_e = frozenset([x, w])
                self.E.append(new_e)
                self.faction[new_e] = self.faction.get(e, random.randint(0, 2))
            
            # 2. 融合 (Fusion) - 30% 概率
            if random.random() < 0.3 and len(self.E) >= 2:
                e1, e2 = random.sample(self.E, 2)
                
                if len(e1 & e2) >= 1:  # 有交集
                    # 计算阵营惩罚
                    f1 = self.faction.get(e1, 0)
                    f2 = self.faction.get(e2, 0)
                    faction_penalty = 0.3 if f1 != f2 else 0.0
                    
                    # 状态中心
                    c1 = np.mean([self.s[u] for u in e1], axis=0)
                    c2 = np.mean([self.s[u] for u in e2], axis=0)
                    
                    distance = np.linalg.norm(c1 - c2)
                    threshold = 0.8 * avg_distance - faction_penalty
                    
                    if distance > threshold:
                        # 融合
                        new_nodes = list(e1 | e2)
                        new_e = frozenset(new_nodes)
                        self.E.append(new_e)
                        self.faction[new_e] = f1
                        
                        # 更新状态
                        new_center = (c1 + c2) / 2
                        for u in new_nodes:
                            self.s[u] = new_center + np.random.randn(self.d) * 0.05
            
            # 3. 分裂 (Split) - 剩余概率
            if len(self.E) >= 2:
                large_edges = [e for e in self.E if len(e) >= 4]
                if large_edges:
                    e = random.choice(large_edges)
                    nodes = list(e)
                    random.shuffle(nodes)
                    mid = len(nodes) // 2
                    
                    if mid >= 1 and len(nodes) - mid >= 1:
                        e1 = frozenset(nodes[:mid])
                        e2 = frozenset(nodes[mid:])
                        
                        self.E.append(e1)
                        self.E.append(e2)
                        self.faction[e1] = self.faction.get(e, random.randint(0, 2))
                        self.faction[e2] = self.faction.get(e, random.randint(0, 2))
                        
                        if e in self.E:
                            self.E.remove(e)
                        if e in self.faction:
                            del self.faction[e]
            
            # 4. 强制容量约束
            self.enforce_capacity()
            
            # 更新平均距离
            avg_distance = self._get_avg_distance()
    
    def _get_avg_distance(self):
        if len(self.V) < 2:
            return 0.1
        samples = min(50, len(self.V) * (len(self.V) - 1) // 2)
        if samples <= 0:
            return 0.1
        dist_sum = 0
        for _ in range(samples):
            u, v = random.sample(self.V, 2)
            dist_sum += np.linalg.norm(self.s[u] - self.s[v])
        return max(dist_sum / samples, 0.01)


# ============================================================
# 实验
# ============================================================

print("="*60)
print("Step 1: M(d) 曲线（带误差棒）")
print("="*60)

# 固定参数
N = 100
gamma = 0.35
K = int(gamma * N)
d_values = [2, 4, 8, 16, 32]
n_runs = 5
steps = 300

results = {d: [] for d in d_values}

for d in d_values:
    print(f"\nd = {d}:")
    for run in range(n_runs):
        seed = 42 + run * 100 + d
        hg = Hypergraph(n_vertices=N, K=K, state_dim=d, seed=seed)
        hg.apply_rules(steps=steps)
        
        M = hg.get_M()
        results[d].append(M)
        
        # 验证 M 在范围内
        assert 0 <= M <= 1, f"M out of range: {M}"
        print(f"  run {run+1}: M = {M:.4f}")

# 统计
print("\n" + "="*60)
print("统计结果")
print("="*60)

means = []
stds = []
for d in d_values:
    mean_M = np.mean(results[d])
    std_M = np.std(results[d])
    means.append(mean_M)
    stds.append(std_M)
    print(f"d = {d:2d}: M* = {mean_M:.4f} ± {std_M:.4f}")

# 绘图
plt.figure(figsize=(10, 6))
plt.errorbar(d_values, means, yerr=stds, fmt='o-', capsize=5, capthick=2, 
             markersize=10, linewidth=2, color='blue')
plt.axhline(y=0.45, color='red', linestyle='--', alpha=0.5, label='M* = 0.45')
plt.xscale('log')
plt.xlabel('Dimension d', fontsize=14)
plt.ylabel('M* (max faction ratio)', fontsize=14)
plt.title(f'M(d) Curve (N={N}, γ={gamma}, {n_runs} runs each)', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(0, 1)

# 保存
plt.savefig('F:/hypergraph_bistability/figures/M_d_curve.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\n图已保存: figures/M_d_curve.png")

# 分析
print("\n" + "="*60)
print("分析")
print("="*60)

# 检查是否收敛到 0.45
if means[-1] < 0.5:
    print(f"✓ d={d_values[-1]} 时 M* = {means[-1]:.4f}，接近 0.45")
else:
    print(f"✗ d={d_values[-1]} 时 M* = {means[-1]:.4f}，未达到 0.45")

# 检查方差
print("\n方差分析：")
for d, std in zip(d_values, stds):
    cv = std / means[d_values.index(d)] * 100  # 变异系数
    print(f"  d = {d:2d}: CV = {cv:.1f}%")
    if cv > 20:
        print(f"    ⚠️ 方差较大，可能不稳定")
    else:
        print(f"    ✓ 相对稳定")

print("\n" + "="*60)
print("结论")
print("="*60)
