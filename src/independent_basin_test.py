"""
Independent Basin Test - 核心实验
=================================

目标：验证"稳态不是分裂，而是独立重建解并存"

核心问题：
- 不同稳态之间是否共享生成路径？
- 如果不共享 → "独立重建解并存"

实验设计：
1. 固定系统（单层，k=3，固定规则）
2. 随机初始化 N≥1000 次
3. 记录收敛位置 + 轨迹
4. 构建 basin clustering
5. 判断是否独立 basin
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import os
import json
from collections import defaultdict

os.makedirs('F:/hypergraph_bistability/figures/basin_test', exist_ok=True)


class Hypergraph:
    """原项目核心超图系统"""
    
    def __init__(self, N=50, gamma=0.35, state_dim=16, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        
        self.N = N
        self.gamma = gamma
        self.state_dim = state_dim
        self.K = int(gamma * N)
        
        self.V = list(range(N))
        self.E = []
        self.s = {v: np.random.randn(state_dim) for v in self.V}
        
        # 初始化超边
        n_initial = max(4, N // 4)
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
        
        # 规则1: 生长
        if random.random() < 0.35 and self.E:
            e = random.choice(self.E)
            nodes = list(e)
            w = len(self.V)
            self.V.append(w)
            self.s[w] = self.s[nodes[0]] + np.random.randn(self.state_dim) * 0.2
            new_e = frozenset([nodes[0], w])
            self.E.append(new_e)
        
        # 规则2: 融合
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
        
        # 规则3: 分裂
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
        
        # 规则4: 容量约束
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
        """阶参数：最大连通分量/总节点数"""
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
    
    def get_state_signature(self):
        """获取系统状态的特征签名"""
        # 简化的状态签名：度分布
        degrees = sorted([self.get_node_degree(v) for v in self.V])
        return tuple(degrees[-10:])  # 取最大的10个度
    
    def run(self, steps=80, record_trajectory=False):
        """运行动力学"""
        M_history = [self.get_M()]
        
        for _ in range(steps):
            self.apply_rules()
            M_history.append(self.get_M())
        
        return M_history


def independent_basin_test(n_runs=1000, gamma=0.35):
    """
    独立Basin测试
    """
    print("=" * 60)
    print("Independent Basin Test")
    print("=" * 60)
    print(f"运行 {n_runs} 次随机初始化...")
    
    # 收集所有结果
    results = []
    
    for i in range(n_runs):
        if i % 100 == 0:
            print(f"  进度: {i}/{n_runs}")
        
        # 每次不同的随机种子
        sys = Hypergraph(N=50, gamma=gamma, seed=i*1000)
        M_history = sys.run(steps=60)
        final_M = M_history[-1]
        
        results.append({
            'seed': i * 1000,
            'final_M': final_M,
            'trajectory': M_history
        })
    
    # 分析M的分布
    M_values = [r['final_M'] for r in results]
    
    print(f"\n结果分析:")
    print(f"  M 范围: [{min(M_values):.3f}, {max(M_values):.3f}]")
    print(f"  M 均值: {np.mean(M_values):.3f}")
    print(f"  M 标准差: {np.std(M_values):.3f}")
    
    # 聚类：找不同的稳定态
    # 使用简单的阈值方法
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    clusters = defaultdict(list)
    for r in results:
        M = r['final_M']
        # 找到最近的阈值
        cluster_id = min(thresholds, key=lambda t: abs(M - t))
        clusters[cluster_id].append(r)
    
    print(f"\n聚类结果 (阈值聚类):")
    for th, items in sorted(clusters.items()):
        print(f"  M ≈ {th}: {len(items)} 次 ({len(items)/n_runs*100:.1f}%)")
    
    # 使用k-means更精确地聚类
    from sklearn.cluster import KMeans
    
    M_array = np.array(M_values).reshape(-1, 1)
    
    # 尝试不同的k
    best_k = 2
    best_score = -1
    
    for k in [2, 3, 4]:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(M_array)
        
        # 计算轮廓系数
        from sklearn.metrics import silhouette_score
        score = silhouette_score(M_array, labels)
        
        if score > best_score:
            best_score = score
            best_k = k
    
    print(f"\n最优聚类数: {best_k} (轮廓系数: {best_score:.3f})")
    
    # 使用最优k
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(M_array)
    
    # 分析每个cluster
    print(f"\n聚类分析:")
    for i in range(best_k):
        cluster_M = [M_values[j] for j in range(len(M_values)) if labels[j] == i]
        print(f"  Cluster {i}: M ∈ [{min(cluster_M):.3f}, {max(cluster_M):.3f}], 均值={np.mean(cluster_M):.3f}")
    
    # 绘制M分布
    plt.figure(figsize=(12, 8))
    
    # 子图1: M分布直方图
    plt.subplot(2, 2, 1)
    plt.hist(M_values, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Final M')
    plt.ylabel('Count')
    plt.title('Distribution of Final M')
    
    # 子图2: 聚类结果
    plt.subplot(2, 2, 2)
    colors = ['red', 'blue', 'green', 'orange']
    for i in range(best_k):
        cluster_M = [M_values[j] for j in range(len(M_values)) if labels[j] == i]
        plt.hist(cluster_M, bins=15, alpha=0.5, label=f'Cluster {i}', color=colors[i])
    plt.xlabel('Final M')
    plt.ylabel('Count')
    plt.title(f'Clustered M Distribution (k={best_k})')
    plt.legend()
    
    # 子图3: 轨迹示例
    plt.subplot(2, 2, 3)
    for i in range(best_k):
        # 从每个cluster随机选一个轨迹
        cluster_idx = [j for j in range(len(results)) if labels[j] == i]
        if cluster_idx:
            example_idx = random.choice(cluster_idx)
            plt.plot(results[example_idx]['trajectory'], 
                     alpha=0.7, label=f'Cluster {i}', color=colors[i])
    plt.xlabel('Time Step')
    plt.ylabel('M')
    plt.title('Example Trajectories')
    plt.legend()
    
    # 子图4: 收敛时间分布
    plt.subplot(2, 2, 4)
    convergence_times = []
    for r in results:
        traj = r['trajectory']
        final_M = r['final_M']
        # 找到收敛时间（首次接近最终值）
        for t in range(len(traj)):
            if abs(traj[t] - final_M) < 0.05:
                convergence_times.append(t)
                break
    
    plt.hist(convergence_times, bins=20, edgecolor='black', alpha=0.7)
    plt.xlabel('Convergence Time')
    plt.ylabel('Count')
    plt.title('Convergence Time Distribution')
    
    plt.tight_layout()
    plt.savefig('F:/hypergraph_bistability/figures/basin_test/basin_distribution.png', dpi=150)
    plt.close()
    
    return {
        'n_runs': n_runs,
        'M_values': M_values,
        'labels': labels,
        'best_k': best_k,
        'best_score': best_score,
        'clusters': {i: [M_values[j] for j in range(len(M_values)) if labels[j] == i] 
                     for i in range(best_k)}
    }


def trajectory_correlation_test(n_runs=500, gamma=0.35):
    """
    轨迹相关性测试：验证不同稳态是否有不同的轨迹模式
    """
    print("\n" + "=" * 60)
    print("Trajectory Correlation Test")
    print("=" * 60)
    
    # 收集轨迹
    trajectories = []
    
    for i in range(n_runs):
        sys = Hypergraph(N=50, gamma=gamma, seed=i*1000)
        traj = sys.run(steps=60)
        trajectories.append(traj)
    
    # 对轨迹进行聚类
    traj_array = np.array(trajectories)
    
    # 使用动态时间规整或简单相关
    from sklearn.cluster import KMeans
    
    # 对最终状态聚类
    final_M = traj_array[:, -1]
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = kmeans.fit_predict(final_M.reshape(-1, 1))
    
    # 计算每个cluster的平均轨迹
    cluster0_traj = traj_array[labels == 0].mean(axis=0)
    cluster1_traj = traj_array[labels == 1].mean(axis=0)
    
    # 计算轨迹相似度
    correlation = np.corrcoef(cluster0_traj, cluster1_traj)[0, 1]
    
    print(f"两个稳态的平均轨迹相关性: {correlation:.3f}")
    
    # 如果相关性低 → 轨迹模式不同 → 独立basin
    if abs(correlation) < 0.5:
        print("→ 轨迹模式显著不同，支持'独立basin'假设")
    else:
        print("→ 轨迹模式相似，可能来自相同basin")
    
    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(cluster0_traj, label='Cluster 0 (Low M)', linewidth=2)
    plt.plot(cluster1_traj, label='Cluster 1 (High M)', linewidth=2)
    plt.xlabel('Time Step')
    plt.ylabel('M')
    plt.title(f'Average Trajectories (Correlation: {correlation:.3f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('F:/hypergraph_bistability/figures/basin_test/trajectory_comparison.png', dpi=150)
    plt.close()
    
    return correlation


def basin_boundary_test():
    """
    测试basin边界：初始条件如何影响最终稳态
    """
    print("\n" + "=" * 60)
    print("Basin Boundary Test")
    print("=" * 60)
    
    # 测试不同的初始超边数量
    results = []
    
    for n_initial in [2, 4, 8, 16, 32]:
        M_values = []
        
        for seed in range(50):
            random.seed(seed)
            np.random.seed(seed)
            
            # 创建系统
            sys = Hypergraph(N=50, gamma=0.35, seed=seed)
            
            # 修改初始超边数量
            sys.E = []
            for _ in range(n_initial):
                size = random.randint(2, min(5, sys.N // 3))
                e = frozenset(random.sample(sys.V, size))
                sys.E.append(e)
            
            traj = sys.run(steps=60)
            M_values.append(traj[-1])
        
        results.append({
            'n_initial': n_initial,
            'M_mean': np.mean(M_values),
            'M_std': np.std(M_values)
        })
        
        print(f"初始超边数 {n_initial}: M = {np.mean(M_values):.3f} ± {np.std(M_values):.3f}")
    
    # 绘图
    plt.figure(figsize=(10, 6))
    x = [r['n_initial'] for r in results]
    y = [r['M_mean'] for r in results]
    yerr = [r['M_std'] for r in results]
    
    plt.errorbar(x, y, yerr=yerr, marker='o', capsize=5)
    plt.xlabel('Number of Initial Hyperedges')
    plt.ylabel('Final M')
    plt.title('Initial Condition Effect on Final State')
    plt.grid(True, alpha=0.3)
    plt.savefig('F:/hypergraph_bistability/figures/basin_test/basin_boundary.png', dpi=150)
    plt.close()


def main():
    print("=" * 60)
    print("Independent Basin Test - Start")
    print("=" * 60)
    
    # 实验1: 独立Basin测试
    result1 = independent_basin_test(n_runs=1000, gamma=0.35)
    
    # 实验2: 轨迹相关性测试
    result2 = trajectory_correlation_test(n_runs=500, gamma=0.35)
    
    # 实验3: Basin边界测试
    result3 = basin_boundary_test()
    
    print("\n" + "=" * 60)
    print("所有实验完成!")
    print("=" * 60)
    
    # 总结
    print("\n=== 结论总结 ===")
    print(f"1. 发现的稳态数量: {result1['best_k']}")
    print(f"2. 聚类质量(轮廓系数): {result1['best_score']:.3f}")
    print(f"3. 轨迹相关性: {result2:.3f}")


if __name__ == "__main__":
    main()
