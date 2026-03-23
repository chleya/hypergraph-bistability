"""
More Rule Variants Testing
==========================
进一步测试规则变体：
1. Fusion/Split 概率比例
2. 噪声/温度影响
3. 初始连接密度
4. 状态维度 d
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import os

os.makedirs('F:/hypergraph_bistability/figures', exist_ok=True)

# ============================================================
# 基础超图模型（修复版本：正确初始化）
# ============================================================

class HypergraphFixed:
    def __init__(self, n_vertices=15, K=5, init_density=0.3, state_dim=4):
        self.V = list(range(n_vertices))
        self.n_vertices = n_vertices
        self.K = K
        self.state_dim = state_dim
        
        # 状态向量
        self.s = {v: np.random.randn(state_dim) for v in self.V}
        
        # 超边
        self.E = []
        
        # 初始化：根据密度创建初始连接
        # 修复：确保有足够的初始连接
        n_edges = max(4, int(n_vertices * init_density))
        for _ in range(n_edges):
            size = random.randint(2, min(5, n_vertices//2))
            if size >= 2:
                e = frozenset(random.sample(self.V, size))
                self.E.append(e)
        
        # 确保至少有一些连接
        if not self.E or len(self.E) < 2:
            # 强制创建一些连接
            for i in range(n_vertices - 1):
                e = frozenset([i, i+1])
                self.E.append(e)
        
        # 阵营
        self.faction = {}
        for e in self.E:
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
        # 无惩罚版本
        return 1.0
    
    def get_factions(self):
        """找连通分量"""
        adj = {v: set() for v in self.V}
        for e in self.E:
            nodes = list(e)
            for i in range(len(nodes)):
                for j in range(i+1, len(nodes)):
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
        return factions
    
    def get_M(self):
        factions = self.get_factions()
        if factions:
            return max(len(f) for f in factions) / self.n_vertices
        return 0
    
    def enforce_K(self):
        """强制容量约束"""
        for v in self.V:
            degree = self.get_node_degree(v)
            if degree > self.K:
                edges_with_v = [e for e in self.E if v in e]
                excess = degree - self.K
                # 移除最小的超边
                edges_with_v.sort(key=lambda e: len(e))
                for _ in range(min(excess, len(edges_with_v))):
                    if edges_with_v and edges_with_v[0] in self.E:
                        e = edges_with_v[0]
                        self.E.remove(e)
                        if e in self.faction:
                            del self.faction[e]
    
    def apply_rules(self, fusion_prob=0.6, noise=0.0):
        """应用规则"""
        avg_dist = self.get_avg_distance()
        
        # 支化 (Growth)
        if random.random() < 0.35 and self.E:
            e = random.choice(self.E)
            nodes = list(e)
            x = random.choice(nodes)
            
            w = len(self.V)
            self.V.append(w)
            self.s[w] = self.s[x] + np.random.randn(self.state_dim) * (0.2 + noise * random.random())
            new_e = frozenset([x, w])
            self.E.append(new_e)
            self.faction[new_e] = self.faction.get(e, random.randint(0, 2))
        
        # 融合 (Fusion)
        if random.random() < fusion_prob and len(self.E) >= 2:
            e1 = random.choice(self.E)
            e2 = random.choice(self.E)
            
            if e1 != e2 and len(e1 & e2) >= 1:
                faction1 = self.faction.get(e1, 0)
                faction2 = self.faction.get(e2, 0)
                faction_penalty = 0.3 if faction1 != faction2 else 0.0
                
                # 状态中心
                c1 = np.mean([self.s[u] for u in e1], axis=0)
                c2 = np.mean([self.s[u] for u in e2], axis=0)
                
                distance = np.linalg.norm(c1 - c2)
                effective_threshold = 0.8 * avg_dist - faction_penalty
                
                if distance > effective_threshold:
                    # 融合
                    new_nodes = list(e1 | e2)
                    new_e = frozenset(new_nodes)
                    self.E.append(new_e)
                    self.faction[new_e] = faction1
                    
                    # 更新状态（带噪声）
                    new_center = (c1 + c2) / 2 + np.random.randn(self.state_dim) * noise
                    for u in new_nodes:
                        self.s[u] = new_center + np.random.randn(self.state_dim) * noise
        
        # 分裂 (Split)
        if random.random() < (1 - fusion_prob) and len(self.E) >= 2:
            # 选择一个大的超边
            large_edges = [e for e in self.E if len(e) >= 3]
            if large_edges:
                e = random.choice(large_edges)
                nodes = list(e)
                random.shuffle(nodes)
                mid = len(nodes) // 2
                e1 = frozenset(nodes[:mid])
                e2 = frozenset(nodes[mid:])
                
                if len(e1) >= 1 and len(e2) >= 1:
                    self.E.append(e1)
                    self.E.append(e2)
                    self.faction[e1] = self.faction.get(e, random.randint(0, 2))
                    self.faction[e2] = self.faction.get(e, random.randint(0, 2))
                    
                    if e in self.E:
                        self.E.remove(e)
                    if e in self.faction:
                        del self.faction[e]
        
        # 强制 K 约束
        self.enforce_K()


def run_fixed(N=50, gamma=0.35, steps=500, seed=42, 
              fusion_prob=0.6, noise=0.0, init_density=0.3, state_dim=4):
    """运行固定参数的模拟"""
    random.seed(seed)
    np.random.seed(seed)
    
    K = int(gamma * N)
    hg = HypergraphFixed(N, K, init_density=init_density, state_dim=state_dim)
    
    Ms = []
    for _ in range(200):  # 减少步数
        Ms.append(hg.get_M())
        hg.apply_rules(fusion_prob=fusion_prob, noise=noise)
    
    return np.mean(Ms[-50:])  # 取后50步平均


# ============================================================
# 实验 1：Fusion/Split 概率比例
# ============================================================

print("实验 1：Fusion/Split 概率比例")
print("-" * 40)

gammas = [0.2, 0.35, 0.5]  # 减少
fusion_probs = [0.3, 0.6, 0.9]  # 减少

results_ratio = {p: [] for p in fusion_probs}

for gamma in gammas:
    for fp in fusion_probs:
        M = run_fixed(N=50, gamma=gamma, steps=500, fusion_prob=fp)
        results_ratio[fp].append(M)
        print(f"  gamma={gamma:.2f}, p_fusion={fp}: M={M:.3f}")

# 绘图
plt.figure(figsize=(10, 6))
for fp, Ms in results_ratio.items():
    plt.plot(gammas, Ms, 'o-', label=f'p_fusion={fp}', markersize=8)
plt.axhline(y=0.45, color='gray', linestyle='--', alpha=0.5)
plt.xlabel('gamma (K/N)', fontsize=12)
plt.ylabel('M', fontsize=12)
plt.title('Fusion/Split Ratio Effect', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('F:/hypergraph_bistability/figures/variant_fusion_ratio.png', dpi=150)
plt.close()

print("\n图已保存: figures/variant_fusion_ratio.png")

# ============================================================
# 实验 2：噪声/温度影响
# ============================================================

print("\n实验 2：噪声/温度影响")
print("-" * 40)

noise_levels = [0.0, 0.05, 0.1, 0.2, 0.5]

results_noise = {n: [] for n in noise_levels}

for gamma in gammas:
    for noise in noise_levels:
        M = run_fixed(N=50, gamma=gamma, steps=500, noise=noise)
        results_noise[noise].append(M)
        print(f"  gamma={gamma:.2f}, noise={noise}: M={M:.3f}")

# 绘图
plt.figure(figsize=(10, 6))
for noise, Ms in results_noise.items():
    plt.plot(gammas, Ms, 's-', label=f'noise={noise}', markersize=8)
plt.axhline(y=0.45, color='gray', linestyle='--', alpha=0.5)
plt.xlabel('gamma (K/N)', fontsize=12)
plt.ylabel('M', fontsize=12)
plt.title('Noise/Temperature Effect', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('F:/hypergraph_bistability/figures/variant_noise.png', dpi=150)
plt.close()

print("\n图已保存: figures/variant_noise.png")

# ============================================================
# 实验 3：初始连接密度
# ============================================================

print("\n实验 3：初始连接密度")
print("-" * 40)

init_densities = [0.1, 0.2, 0.3, 0.5, 0.8]

results_density = {d: [] for d in init_densities}

for gamma in gammas:
    for density in init_densities:
        M = run_fixed(N=50, gamma=gamma, steps=500, init_density=density)
        results_density[density].append(M)
        print(f"  gamma={gamma:.2f}, density={density}: M={M:.3f}")

# 绘图
plt.figure(figsize=(10, 6))
for density, Ms in results_density.items():
    plt.plot(gammas, Ms, '^-', label=f'density={density}', markersize=8)
plt.axhline(y=0.45, color='gray', linestyle='--', alpha=0.5)
plt.xlabel('gamma (K/N)', fontsize=12)
plt.ylabel('M', fontsize=12)
plt.title('Initial Density Effect', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('F:/hypergraph_bistability/figures/variant_density.png', dpi=150)
plt.close()

print("\n图已保存: figures/variant_density.png")

# ============================================================
# 实验 4：状态维度 d
# ============================================================

print("\n实验 4：状态维度 d")
print("-" * 40)

state_dims = [2, 4, 8, 16]

results_dim = {d: [] for d in state_dims}

for gamma in gammas:
    for dim in state_dims:
        M = run_fixed(N=50, gamma=gamma, steps=500, state_dim=dim)
        results_dim[dim].append(M)
        print(f"  gamma={gamma:.2f}, dim={dim}: M={M:.3f}")

# 绘图
plt.figure(figsize=(10, 6))
for dim, Ms in results_dim.items():
    plt.plot(gammas, Ms, 'd-', label=f'dim={dim}', markersize=8)
plt.axhline(y=0.45, color='gray', linestyle='--', alpha=0.5)
plt.xlabel('gamma (K/N)', fontsize=12)
plt.ylabel('M', fontsize=12)
plt.title('State Dimension Effect', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('F:/hypergraph_bistability/figures/variant_dimension.png', dpi=150)
plt.close()

print("\n图已保存: figures/variant_dimension.png")

# ============================================================
# 总结
# ============================================================

print("\n" + "="*60)
print("总结")
print("="*60)

print("\n关键发现：")
print("\n1. Fusion/Split 比例：")
for fp, Ms in results_ratio.items():
    avg_M = np.mean(Ms)
    print(f"   p_fusion={fp}: avg M = {avg_M:.3f}")

print("\n2. 噪声影响：")
for noise, Ms in results_noise.items():
    avg_M = np.mean(Ms)
    print(f"   noise={noise}: avg M = {avg_M:.3f}")

print("\n3. 初始密度：")
for density, Ms in results_density.items():
    avg_M = np.mean(Ms)
    print(f"   density={density}: avg M = {avg_M:.3f}")

print("\n4. 状态维度：")
for dim, Ms in results_dim.items():
    avg_M = np.mean(Ms)
    print(f"   dim={dim}: avg M = {avg_M:.3f}")
