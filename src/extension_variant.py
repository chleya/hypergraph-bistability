"""
Rule Variant Extensions
======================
测试不同规则变体是否影响临界点 γ_c ≈ 0.35 和 M* ≈ 0.45

变体：
1. Fusion 策略：random vs preferential vs inverse-preferential
2. Split 策略：random vs size-biased vs position-based
3. Rejection 机制：degree-based vs age-based vs random
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os

os.makedirs('F:/hypergraph_bistability/figures', exist_ok=True)

# ============================================================
# 基础超图模型
# ============================================================

class Hypergraph:
    def __init__(self, N, K):
        self.N = N
        self.K = K  # 容量约束
        self.hyperedges = []  # list of sets
        self.node_degrees = defaultdict(int)
        
        # 初始化：每个节点一个超边
        for i in range(N):
            self.hyperedges.append({i})
            self.node_degrees[i] = 1
    
    def get_factions(self):
        """找连通分量（阵营）"""
        # 构建邻接
        adj = defaultdict(set)
        for he in self.hyperedges:
            nodes = list(he)
            for i in range(len(nodes)):
                for j in range(i+1, len(nodes)):
                    adj[nodes[i]].add(nodes[j])
                    adj[nodes[j]].add(nodes[i])
        
        # BFS 找连通分量
        visited = set()
        factions = []
        for start in range(self.N):
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
        """最大阵营占比"""
        factions = self.get_factions()
        if factions:
            return max(len(f) for f in factions) / self.N
        return 0
    
    def fusion_random(self):
        """随机融合：随机选择两个有交集的超边"""
        if len(self.hyperedges) < 2:
            return False
        
        # 找所有候选对
        candidates = []
        for i in range(len(self.hyperedges)):
            for j in range(i+1, len(self.hyperedges)):
                if self.hyperedges[i] & self.hyperedges[j]:
                    candidates.append((i, j))
        
        if not candidates:
            return False
        
        # 随机选择
        i, j = candidates[np.random.randint(len(candidates))]
        return self._merge(i, j)
    
    def fusion_preferential(self):
        """优先融合：按超边大小加权（大的更容易融合）"""
        if len(self.hyperedges) < 2:
            return False
        
        # 计算权重：大的超边权重高
        weights = np.array([len(he) for he in self.hyperedges], dtype=float)
        weights = weights / weights.sum()
        
        # 选择两个（按概率）
        indices = np.random.choice(len(self.hyperedges), 2, replace=False, p=weights)
        i, j = min(indices), max(indices)
        
        if self.hyperedges[i] & self.hyperedges[j]:
            return self._merge(i, j)
        return False
    
    def fusion_inverse_preferential(self):
        """反向优先融合：小的更容易融合"""
        if len(self.hyperedges) < 2:
            return False
        
        # 权重：小的超边权重高
        sizes = np.array([len(he) for he in self.hyperedges], dtype=float)
        weights = 1.0 / (sizes + 1)
        weights = weights / weights.sum()
        
        indices = np.random.choice(len(self.hyperedges), 2, replace=False, p=weights)
        i, j = min(indices), max(indices)
        
        if self.hyperedges[i] & self.hyperedges[j]:
            return self._merge(i, j)
        return False
    
    def split_random(self):
        """随机分裂：随机选择一个大超边，随机分成两部分"""
        large_indices = [i for i, he in enumerate(self.hyperedges) if len(he) >= 3]
        if not large_indices:
            return False
        
        i = np.random.choice(large_indices)
        he = list(self.hyperedges[i])
        np.random.shuffle(he)
        
        mid = len(he) // 2
        if mid == 0 or mid == len(he):
            return False
        
        he1, he2 = set(he[:mid]), set(he[mid:])
        return self._split(i, he1, he2)
    
    def split_size_biased(self):
        """按大小偏置分裂：越大的超边越容易分裂"""
        large_indices = [i for i, he in enumerate(self.hyperedges) if len(he) >= 3]
        if not large_indices:
            return False
        
        # 按大小加权
        sizes = np.array([len(self.hyperedges[i]) for i in large_indices], dtype=float)
        weights = sizes / sizes.sum()
        i = np.random.choice(large_indices, p=weights)
        
        he = list(self.hyperedges[i])
        np.random.shuffle(he)
        
        mid = len(he) // 2
        he1, he2 = set(he[:mid]), set(he[mid:])
        return self._split(i, he1, he2)
    
    def _merge(self, i, j):
        """融合超边 i 和 j"""
        new_he = self.hyperedges[i] | self.hyperedges[j]
        
        # 检查容量约束
        for node in new_he:
            if self.node_degrees[node] >= self.K:
                return False
        
        # 更新 degree
        for node in self.hyperedges[i]:
            self.node_degrees[node] -= 1
        for node in self.hyperedges[j]:
            self.node_degrees[node] -= 1
        for node in new_he:
            self.node_degrees[node] += 1
        
        self.hyperedges[i] = new_he
        self.hyperedges.pop(j)
        return True
    
    def _split(self, i, he1, he2):
        """分裂超边 i 为 he1 和 he2"""
        old_he = self.hyperedges[i]
        
        # 更新 degree
        for node in old_he:
            self.node_degrees[node] -= 1
        for node in he1:
            self.node_degrees[node] += 1
        for node in he2:
            self.node_degrees[node] += 1
        
        self.hyperedges[i] = he1
        self.hyperedges.append(he2)
        return True


def run_simulation(N=50, gamma=0.35, steps=500, seed=42,
                  fusion_rule='random', split_rule='random', 
                  fusion_prob=0.6):
    """运行模拟"""
    np.random.seed(seed)
    
    K = int(gamma * N)
    hg = Hypergraph(N, K)
    
    Ms = []
    for step in range(steps):
        Ms.append(hg.get_M())
        
        # 选择规则
        if np.random.random() < fusion_prob:
            if fusion_rule == 'random':
                hg.fusion_random()
            elif fusion_rule == 'preferential':
                hg.fusion_preferential()
            elif fusion_rule == 'inverse_preferential':
                hg.fusion_inverse_preferential()
        else:
            if split_rule == 'random':
                hg.split_random()
            elif split_rule == 'size_biased':
                hg.split_size_biased()
    
    return np.mean(Ms[-100:])


# ============================================================
# 实验 1：不同 Fusion 策略
# ============================================================

print("实验 1：不同 Fusion 策略")
print("-" * 40)

gammas = np.linspace(0.1, 1.0, 10)

results_fusion = {
    'random': [],
    'preferential': [],
    'inverse_preferential': []
}

for gamma in gammas:
    for rule in results_fusion.keys():
        M = run_simulation(N=50, gamma=gamma, steps=500, fusion_rule=rule)
        results_fusion[rule].append(M)
        print(f"  gamma={gamma:.2f}, {rule}: M={M:.3f}")

# 绘图
plt.figure(figsize=(10, 6))
for rule, Ms in results_fusion.items():
    plt.plot(gammas, Ms, 'o-', label=rule, markersize=8)
plt.axhline(y=0.45, color='gray', linestyle='--', alpha=0.5, label='M*=0.45')
plt.axvline(x=0.35, color='gray', linestyle=':', alpha=0.5, label='gamma_c=0.35')
plt.xlabel('gamma (K/N)', fontsize=12)
plt.ylabel('M', fontsize=12)
plt.title('Fusion Rule Variants', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('F:/hypergraph_bistability/figures/variant_fusion.png', dpi=150)
plt.close()

print("\n图已保存: figures/variant_fusion.png")

# ============================================================
# 实验 2：不同 Split 策略
# ============================================================

print("\n实验 2：不同 Split 策略")
print("-" * 40)

results_split = {
    'random': [],
    'size_biased': []
}

for gamma in gammas:
    for rule in results_split.keys():
        M = run_simulation(N=50, gamma=gamma, steps=500, split_rule=rule)
        results_split[rule].append(M)
        print(f"  gamma={gamma:.2f}, {rule}: M={M:.3f}")

# 绘图
plt.figure(figsize=(10, 6))
for rule, Ms in results_split.items():
    plt.plot(gammas, Ms, 's-', label=rule, markersize=8)
plt.axhline(y=0.45, color='gray', linestyle='--', alpha=0.5)
plt.axvline(x=0.35, color='gray', linestyle=':', alpha=0.5)
plt.xlabel('gamma (K/N)', fontsize=12)
plt.ylabel('M', fontsize=12)
plt.title('Split Rule Variants', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('F:/hypergraph_bistability/figures/variant_split.png', dpi=150)
plt.close()

print("\n图已保存: figures/variant_split.png")

# ============================================================
# 实验 3：综合对比
# ============================================================

print("\n实验 3：综合对比")
print("-" * 40)

# 选择几种代表性组合
configs = [
    ('random-random', 'random', 'random'),
    ('pref-size', 'preferential', 'size_biased'),
    ('inv-random', 'inverse_preferential', 'random'),
]

results_combo = {name: [] for name, _, _ in configs}

for gamma in gammas:
    for name, fusion_rule, split_rule in configs:
        M = run_simulation(N=50, gamma=gamma, steps=500, 
                         fusion_rule=fusion_rule, split_rule=split_rule)
        results_combo[name].append(M)
        print(f"  gamma={gamma:.2f}, {name}: M={M:.3f}")

# 绘图
plt.figure(figsize=(10, 6))
for name, Ms in results_combo.items():
    plt.plot(gammas, Ms, 'o-', label=name, markersize=8)
plt.axhline(y=0.45, color='gray', linestyle='--', alpha=0.5, label='M*=0.45')
plt.axvline(x=0.35, color='gray', linestyle=':', alpha=0.5, label='gamma_c=0.35')
plt.xlabel('gamma (K/N)', fontsize=12)
plt.ylabel('M', fontsize=12)
plt.title('Combined Variants', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('F:/hypergraph_bistability/figures/variant_combined.png', dpi=150)
plt.close()

print("\n图已保存: figures/variant_combined.png")

# ============================================================
# 总结
# ============================================================

print("\n" + "="*60)
print("总结")
print("="*60)

print("\n关键观察：")

for gamma_idx, gamma in enumerate(gammas):
    print(f"\ngamma = {gamma:.2f}:")
    for rule, Ms in results_fusion.items():
        print(f"  {rule}: M = {Ms[gamma_idx]:.3f}")

print("""
\n分析：
1. 不同 fusion 策略是否改变 M* 趋近值？
2. 不同 split 策略是否改变临界点位置？
3. 组合效果是否叠加？
""")
