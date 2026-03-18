"""
Forbidden Region Test - 最后一块证据
=====================================

目标：证明"不是所有状态都能存在"

实验设计：
1. 高分辨率采样所有初始条件
2. 记录所有最终M值
3. 看分布是否有空洞
"""

import numpy as np
import random
import matplotlib.pyplot as plt
from scipy import stats
import os

os.makedirs('F:/hypergraph_bistability/figures/forbidden', exist_ok=True)


class Hypergraph:
    def __init__(self, N=50, gamma=0.35, state_dim=16, seed=42, n_initial=4):
        random.seed(seed)
        np.random.seed(seed)
        
        self.N = N
        self.gamma = gamma
        self.state_dim = state_dim
        self.K = int(gamma * N)
        self.n_initial = n_initial
        
        self.V = list(range(N))
        self.E = []
        self.s = {v: np.random.randn(state_dim) for v in self.V}
        
        for _ in range(n_initial):
            size = random.randint(2, min(5, N // 3))
            e = frozenset(random.sample(self.V, size))
            self.E.append(e)
    
    def get_node_degree(self, v):
        return sum(1 for e in self.E if v in e)
    
    def get_avg_distance(self):
        if len(self.V) < 2:
            return 0.1
        samples = min(20, len(self.V) * (len(self.V) - 1) // 2)
        if samples <= 0:
            return 0.1
        dist_sum = 0
        for _ in range(samples):
            u, v = random.sample(self.V, 2)
            dist_sum += np.linalg.norm(self.s[u] - self.s[v])
        return max(dist_sum / samples, 0.01)
    
    def apply_rules(self):
        avg_dist = self.get_avg_distance()
        
        if random.random() < 0.35 and self.E:
            e = random.choice(self.E)
            nodes = list(e)
            w = len(self.V)
            self.V.append(w)
            self.s[w] = self.s[nodes[0]] + np.random.randn(self.state_dim) * 0.2
            new_e = frozenset([nodes[0], w])
            self.E.append(new_e)
        
        if random.random() < 0.3 and len(self.E) >= 2:
            e1, e2 = random.sample(self.E, 2)
            if len(e1 & e2) >= 1:
                weights1 = np.array([1.0 / (1 + self.get_node_degree(v)) for v in e1])
                weights1 = weights1 / weights1.sum()
                c1 = np.average([self.s[u] for u in e1], axis=0, weights=weights1)
                
                weights2 = np.array([1.0 / (1 + self.get_node_degree(v)) for v in e2])
                weights2 = weights2 / weights2.sum()
                c2 = np.average([self.s[u] for u in e2], axis=0, weights=weights2)
                
                dist = np.linalg.norm(c1 - c2)
                if dist < avg_dist * 0.8:
                    new_e = frozenset(e1 | e2)
                    if new_e not in self.E:
                        self.E.append(new_e)
                        self.E.remove(e1)
                        self.E.remove(e2)
        
        if random.random() < 0.15 and len(self.E) > 1:
            large_edges = [e for e in self.E if len(e) > 2]
            if large_edges:
                e = random.choice(large_edges)
                nodes = list(e)
                random.shuffle(nodes)
                split = len(nodes) // 2
                self.E.append(frozenset(nodes[:split]))
                self.E.append(frozenset(nodes[split:]))
                self.E.remove(e)
        
        for v in self.V:
            degree = self.get_node_degree(v)
            if degree > self.K:
                excess = degree - self.K
                v_edges = [e for e in self.E if v in e]
                for e in v_edges[:excess]:
                    if len(e) > 2:
                        new_e = e - {v}
                        if len(new_e) >= 2:
                            self.E.remove(e)
                            self.E.append(new_e)
    
    def get_M(self):
        if not self.V:
            return 0
        
        visited = set()
        max_cluster = 0
        
        for start in self.V:
            if start in visited:
                continue
            stack = [start]
            cluster = set()
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                cluster.add(node)
                for e in self.E:
                    if node in e:
                        for neighbor in e:
                            if neighbor not in visited:
                                stack.append(neighbor)
            max_cluster = max(max_cluster, len(cluster))
        
        return max_cluster / len(self.V) if self.V else 0
    
    def run(self, steps=80):
        for _ in range(steps):
            self.apply_rules()
        return self.get_M()


def forbidden_region_test():
    """核心实验：Forbidden Region Test"""
    print("=" * 60)
    print("Forbidden Region Test")
    print("=" * 60)
    
    # 大规模采样：不同 n_init × 不同 seed
    all_M = []
    
    print("采样中...")
    # 关键区域密集采样
    for n_init in range(3, 35):
        for seed in range(50):  # 每个n_init 50次
            sys = Hypergraph(N=50, gamma=0.35, seed=seed, n_initial=n_init)
            M = sys.run(steps=60)
            all_M.append({
                'n_init': n_init,
                'seed': seed,
                'M': M
            })
    
    M_values = [x['M'] for x in all_M]
    
    print(f"\n总采样数: {len(M_values)}")
    print(f"M 范围: [{min(M_values):.3f}, {max(M_values):.3f}]")
    print(f"M 均值: {np.mean(M_values):.3f}")
    print(f"M 标准差: {np.std(M_values):.3f}")
    
    # 分析1: 整体分布
    plt.figure(figsize=(14, 10))
    
    plt.subplot(2, 2, 1)
    hist, bins, patches = plt.hist(M_values, bins=40, edgecolor='black', alpha=0.7)
    plt.xlabel('Final M')
    plt.ylabel('Count')
    plt.title('Overall M Distribution (All Initial Conditions)')
    
    # 分析2: 找空洞/低谷
    print("\n=== 分析分布形态 ===")
    
    # 找低谷点（count接近0的区域）
    low_points = []
    for i in range(len(hist)):
        if hist[i] < 5:  # 计数<5认为是低谷
            low_points.append((bins[i], bins[i+1], hist[i]))
    
    print(f"\n低谷区域 (count < 5):")
    for start, end, count in low_points[:10]:
        print(f"  M ∈ [{start:.2f}, {end:.2f}): count = {count}")
    
    # 分析3: 双峰检验
    from scipy.signal import find_peaks
    
    peaks, _ = find_peaks(hist, height=10, distance=3)
    print(f"\n峰值位置:")
    for p in peaks:
        print(f"  M ≈ {(bins[p] + bins[p+1])/2:.2f}: count = {hist[p]}")
    
    # 分析4: 密度估计
    plt.subplot(2, 2, 2)
    
    # KDE
    kde = stats.gaussian_kde(M_values)
    x_range = np.linspace(0, 1, 200)
    density = kde(x_range)
    
    plt.plot(x_range, density, linewidth=2)
    plt.fill_between(x_range, density, alpha=0.3)
    plt.xlabel('Final M')
    plt.ylabel('Density')
    plt.title('Kernel Density Estimation')
    
    # 标记低谷
    for start, end, _ in low_points[:5]:
        plt.axvspan(start, end, alpha=0.2, color='red')
    
    # 分析5: 按n_init分组看分布演变
    plt.subplot(2, 2, 3)
    
    for n_init in [5, 10, 15, 20, 25, 30]:
        subset = [x['M'] for x in all_M if x['n_init'] == n_init]
        if len(subset) > 5:
            kde_sub = stats.gaussian_kde(subset)
            density_sub = kde_sub(x_range)
            plt.plot(x_range, density_sub, label=f'n_init={n_init}', alpha=0.7)
    
    plt.xlabel('Final M')
    plt.ylabel('Density')
    plt.title('M Distribution by n_init')
    plt.legend()
    
    # 分析6: 2D heatmap
    plt.subplot(2, 2, 4)
    
    # 创建 2D 数据
    n_init_range = list(range(3, 35))
    M_bins = np.linspace(0, 1, 20)
    
    heatmap = np.zeros((len(n_init_range), len(M_bins)-1))
    
    for x in all_M:
        n_idx = x['n_init'] - 3
        m_idx = np.searchsorted(M_bins, x['M']) - 1
        if 0 <= n_idx < len(n_init_range) and 0 <= m_idx < len(M_bins)-1:
            heatmap[n_idx, m_idx] += 1
    
    plt.imshow(heatmap, aspect='auto', origin='lower', 
               extent=[0, 1, 3, 34], cmap='YlOrRd')
    plt.xlabel('Final M')
    plt.ylabel('n_init')
    plt.title('2D: n_init vs M')
    plt.colorbar(label='Count')
    
    plt.tight_layout()
    plt.savefig('F:/hypergraph_bistability/figures/forbidden/forbidden_analysis.png', dpi=150)
    plt.close()
    
    # 统计分析：检验双峰性
    print("\n=== 统计分析 ===")
    
    # 计算峰谷比
    if len(peaks) >= 2:
        peak_heights = [hist[p] for p in peaks]
        min_between = []
        for i in range(len(peaks) - 1):
            start_idx = peaks[i]
            end_idx = peaks[i+1]
            min_val = min(hist[start_idx:end_idx+1])
            min_between.append(min_val)
        
        if min_between:
            valley_ratio = min(peak_heights) / (min(min_between) + 1)
            print(f"峰谷比: {valley_ratio:.2f}")
            
            if valley_ratio > 3:
                print("→ 存在明显 Forbidden Region!")
            elif valley_ratio > 1.5:
                print("→ 存在弱 Forbidden Region")
            else:
                print("→ 无明显 Forbidden Region")
    
    # 分析中间区域
    middle_low = sum(1 for m in M_values if 0.3 <= m <= 0.5)
    total = len(M_values)
    middle_ratio = middle_low / total
    
    print(f"\n中间区域 M ∈ [0.3, 0.5]: {middle_low}/{total} = {middle_ratio:.1%}")
    
    if middle_ratio < 0.1:
        print("→ 中间区域极少！存在强 Forbidden Region!")
    elif middle_ratio < 0.2:
        print("→ 中间区域较少")
    else:
        print("→ 中间区域正常分布")
    
    # 最终判断
    print("\n=== 最终判断 ===")
    
    # 三种情况
    has_valley = len(low_points) > 3
    middle_sparse = middle_ratio < 0.2
    is_bimodal = len(peaks) >= 2
    
    if is_bimodal and middle_sparse:
        print("情况A: 双峰 + 中间空洞 → 存在 Forbidden Region!")
        conclusion = "A"
    elif is_bimodal:
        print("情况B: 双峰 + 中间较少")
        conclusion = "B"
    else:
        print("情况C: 连续分布")
        conclusion = "C"
    
    return {
        'all_M': M_values,
        'peaks': peaks,
        'low_points': low_points,
        'middle_ratio': middle_ratio,
        'conclusion': conclusion
    }


def detailed_boundary_analysis():
    """更详细的边界分析"""
    print("\n" + "=" * 60)
    print("Detailed Boundary Analysis")
    print("=" * 60)
    
    # 只看临界区附近
    critical_results = []
    
    for n_init in range(10, 22):
        M_values = []
        
        for seed in range(100):
            sys = Hypergraph(N=50, gamma=0.35, seed=seed, n_initial=n_init)
            M = sys.run(steps=60)
            M_values.append(M)
        
        critical_results.append({
            'n_init': n_init,
            'M_values': M_values,
            'mean': np.mean(M_values),
            'std': np.std(M_values)
        })
        
        # 统计LOW/HIGH
        low = sum(1 for m in M_values if m < 0.35)
        high = sum(1 for m in M_values if m > 0.55)
        mid = 100 - low - high
        
        print(f"n_init = {n_init}: LOW={low}, MID={mid}, HIGH={high}")
    
    # 绘图
    plt.figure(figsize=(12, 8))
    
    n_inits = [r['n_init'] for r in critical_results]
    means = [r['mean'] for r in critical_results]
    stds = [r['std'] for r in critical_results]
    
    plt.errorbar(n_inits, means, yerr=stds, marker='o', capsize=5)
    plt.axhline(y=0.35, color='r', linestyle='--', alpha=0.5, label='LOW/HIGH boundary')
    plt.axhline(y=0.55, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('n_init')
    plt.ylabel('Final M')
    plt.title('Critical Zone: n_init vs M')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('F:/hypergraph_bistability/figures/forbidden/boundary_detail.png', dpi=150)
    plt.close()


if __name__ == "__main__":
    result = forbidden_region_test()
    detailed_boundary_analysis()
    print("\n完成!")
