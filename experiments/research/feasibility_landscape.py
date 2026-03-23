"""
Feasibility Landscape Mapping - 可行性 landscape 映射
==================================================

目标：找到"不可达区间"和"跳跃点"

核心问题：
- 扫描初始超边数 1 → 50
- 记录最终M、收敛时间
- 寻找：
  1. 跳跃点 (critical threshold)
  2. 不稳定区 (slow convergence)
  3. 不可达区间 (unreachable M intervals)
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import os

os.makedirs('F:/hypergraph_bistability/figures/feasibility', exist_ok=True)


class Hypergraph:
    """原项目核心超图系统"""
    
    def __init__(self, N=50, gamma=0.35, state_dim=16, seed=42, n_initial=4):
        random.seed(seed)
        np.random.seed(seed)
        
        self.N = N
        self.gamma = gamma
        self.state_dim = state_dim
        self.K = int(gamma * N)
        self.n_initial = n_initial  # 可变的初始超边数
        
        self.V = list(range(N))
        self.E = []
        self.s = {v: np.random.randn(state_dim) for v in self.V}
        
        # 初始化指定数量的超边
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
    
    def run_with_trajectory(self, steps=80, record_every=1):
        """运行并记录完整轨迹"""
        trajectory = [self.get_M()]
        
        for t in range(steps):
            self.apply_rules()
            trajectory.append(self.get_M())
        
        return trajectory


def feasibility_landscape():
    """核心实验：可行 landscape 映射"""
    print("=" * 60)
    print("Feasibility Landscape Mapping")
    print("=" * 60)
    
    # 扫描初始超边数 1 → 50
    results = []
    
    print("扫描初始条件...")
    for n_init in range(1, 51):
        if n_init % 10 == 0:
            print(f"  进度: {n_init}/50")
        
        M_values = []
        convergence_times = []
        
        for seed in range(30):  # 每次运行30次取平均
            sys = Hypergraph(N=50, gamma=0.35, seed=seed, n_initial=n_init)
            traj = sys.run_with_trajectory(steps=60)
            
            final_M = traj[-1]
            M_values.append(final_M)
            
            # 收敛时间：首次接近最终值5%以内
            converged = False
            for t in range(len(traj)):
                if abs(traj[t] - final_M) < final_M * 0.05 + 0.01:
                    convergence_times.append(t)
                    converged = True
                    break
            if not converged:
                convergence_times.append(len(traj))
        
        results.append({
            'n_init': n_init,
            'mean_M': np.mean(M_values),
            'std_M': np.std(M_values),
            'mean_conv_time': np.mean(convergence_times),
            'all_M': M_values
        })
    
    # 分析结果
    print("\n=== 分析结果 ===")
    
    # 提取数据
    n_inits = [r['n_init'] for r in results]
    mean_Ms = [r['mean_M'] for r in results]
    std_Ms = [r['std_M'] for r in results]
    conv_times = [r['mean_conv_time'] for r in results]
    
    # 1. 找跳跃点
    jumps = []
    for i in range(1, len(mean_Ms)):
        dM = mean_Ms[i] - mean_Ms[i-1]
        if abs(dM) > 0.1:  # 显著跳跃
            jumps.append({
                'n_init': n_inits[i],
                'from': mean_Ms[i-1],
                'to': mean_Ms[i],
                'delta': dM
            })
    
    print(f"\n1. 跳跃点 (|ΔM| > 0.1):")
    for j in jumps:
        print(f"   n_init = {j['n_init']}: M {j['from']:.2f} → {j['to']:.2f} (Δ={j['delta']:.2f})")
    
    # 2. 找不可达区间
    all_M_flat = []
    for r in results:
        all_M_flat.extend(r['all_M'])
    
    # 统计每个M区间的频率
    M_hist, bins = np.histogram(all_M_flat, bins=50)
    zero_bins = []
    for i in range(len(M_hist)):
        if M_hist[i] == 0:
            zero_bins.append((bins[i], bins[i+1]))
    
    print(f"\n2. 不可达区间 (出现次数=0):")
    if zero_bins:
        # 合并连续区间
        merged = []
        for start, end in zero_bins:
            if merged and abs(start - merged[-1][1]) < 0.05:
                merged[-1] = (merged[-1][0], end)
            else:
                merged.append((start, end))
        for start, end in merged[:5]:  # 只显示前5个
            print(f"   M ∈ [{start:.2f}, {end:.2f})")
    else:
        print("   无明显不可达区间")
    
    # 3. 不稳定区（收敛时间长的）
    slow_conv = []
    for r in results:
        if r['mean_conv_time'] > 40:  # 收敛慢
            slow_conv.append((r['n_init'], r['mean_conv_time']))
    
    print(f"\n3. 不稳定区 (收敛时间 > 40):")
    if slow_conv:
        for n, t in slow_conv[:5]:
            print(f"   n_init = {n}: 收敛时间 = {t:.1f}")
    else:
        print("   无明显不稳定区")
    
    # 绘图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 图1: M vs n_init
    ax1 = axes[0, 0]
    ax1.errorbar(n_inits, mean_Ms, yerr=std_Ms, marker='o', capsize=3, markersize=4)
    ax1.set_xlabel('Initial Hyperedge Count')
    ax1.set_ylabel('Final M')
    ax1.set_title('Feasibility Landscape: Initial Size → Final M')
    ax1.grid(True, alpha=0.3)
    
    # 标记跳跃点
    for j in jumps:
        ax1.axvline(x=j['n_init'], color='r', linestyle='--', alpha=0.5)
    
    # 图2: M分布直方图
    ax2 = axes[0, 1]
    ax2.hist(all_M_flat, bins=30, edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Final M')
    ax2.set_ylabel('Count')
    ax2.set_title('M Distribution (All Initial Conditions)')
    
    # 图3: 收敛时间
    ax3 = axes[1, 0]
    ax3.plot(n_inits, conv_times, marker='.', linewidth=1)
    ax3.set_xlabel('Initial Hyperedge Count')
    ax3.set_ylabel('Convergence Time')
    ax3.set_title('Convergence Time vs Initial Size')
    ax3.grid(True, alpha=0.3)
    
    # 图4: 2D phase space
    ax4 = axes[1, 1]
    ax4.scatter(n_inits, mean_Ms, c=conv_times, cmap='viridis', s=50)
    ax4.set_xlabel('Initial Hyperedge Count')
    ax4.set_ylabel('Final M')
    ax4.set_title('Phase Space (color = convergence time)')
    plt.colorbar(ax4.collections[0], ax=ax4, label='Conv. Time')
    
    plt.tight_layout()
    plt.savefig('F:/hypergraph_bistability/figures/feasibility/landscape.png', dpi=150)
    plt.close()
    
    # 详细分析：找临界点
    print("\n=== 详细分析 ===")
    
    # 计算一阶导数
    dMdn = np.gradient(mean_Ms, n_inits)
    
    # 找临界点（导数绝对值最大的点）
    critical_points = []
    for i in range(len(dMdn)):
        if abs(dMdn[i]) > 0.02:  # 显著变化
            critical_points.append((n_inits[i], mean_Ms[i], dMdn[i]))
    
    print("\n临界点 (|dM/dn| > 0.02):")
    for n, m, d in critical_points[:10]:
        print(f"   n_init = {n}: M = {m:.3f}, dM/dn = {d:.3f}")
    
    # 精细扫描关键区域
    print("\n=== 精细扫描 (n_init = 10-20) ===")
    fine_results = []
    for n_init in range(10, 21):
        M_values = []
        for seed in range(50):
            sys = Hypergraph(N=50, gamma=0.35, seed=seed, n_initial=n_init)
            traj = sys.run_with_trajectory(steps=80)
            M_values.append(traj[-1])
        
        fine_results.append({
            'n_init': n_init,
            'mean_M': np.mean(M_values),
            'std_M': np.std(M_values),
            'M_values': M_values
        })
        
        # 统计分布
        low_count = sum(1 for m in M_values if m < 0.3)
        high_count = sum(1 for m in M_values if m > 0.5)
        
        print(f"n_init = {n_init}: M = {np.mean(M_values):.3f} ± {np.std(M_values):.3f}, LOW:{low_count}/HIGH:{high_count}")
    
    # 绘制精细扫描
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    n_inits_fine = [r['n_init'] for r in fine_results]
    mean_Ms_fine = [r['mean_M'] for r in fine_results]
    std_Ms_fine = [r['std_M'] for r in fine_results]
    plt.errorbar(n_inits_fine, mean_Ms_fine, yerr=std_Ms_fine, marker='o', capsize=5)
    plt.xlabel('Initial Hyperedge Count')
    plt.ylabel('Final M')
    plt.title('Fine Scan: n_init = 10-20')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    # 绘制分布变化
    for r in fine_results:
        M_vals = r['M_values']
        plt.hist(M_vals, bins=10, alpha=0.3, label=f'n={r["n_init"]}')
    plt.xlabel('Final M')
    plt.ylabel('Count')
    plt.title('M Distribution Evolution')
    plt.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig('F:/hypergraph_bistability/figures/feasibility/fine_scan.png', dpi=150)
    plt.close()
    
    return {
        'jumps': jumps,
        'zero_bins': zero_bins,
        'critical_points': critical_points,
        'results': results
    }


def self_consistency_test():
    """测试自洽性：不同M的constraint violation"""
    print("\n" + "=" * 60)
    print("Self-Consistency Gap Test")
    print("=" * 60)
    
    # 对不同最终M值，计算其"自洽程度"
    # 方法：检查超边的平均大小 vs M的关系
    
    results = []
    
    for n_init in [5, 10, 15, 20, 25, 30, 40]:
        M_values = []
        avg_edge_sizes = []
        
        for seed in range(30):
            sys = Hypergraph(N=50, gamma=0.35, seed=seed, n_initial=n_init)
            sys.run_with_trajectory(steps=60)
            
            M = sys.get_M()
            M_values.append(M)
            
            # 平均超边大小
            if sys.E:
                avg_size = np.mean([len(e) for e in sys.E])
            else:
                avg_size = 0
            avg_edge_sizes.append(avg_size)
        
        results.append({
            'n_init': n_init,
            'mean_M': np.mean(M_values),
            'mean_edge_size': np.mean(avg_edge_sizes),
            'correlation': np.corrcoef(M_values, avg_edge_sizes)[0, 1]
        })
        
        print(f"n_init = {n_init}: M = {np.mean(M_values):.3f}, avg_edge = {np.mean(avg_edge_sizes):.3f}, corr = {results[-1]['correlation']:.3f}")
    
    # 绘图
    plt.figure(figsize=(10, 6))
    
    n_inits = [r['n_init'] for r in results]
    mean_Ms = [r['mean_M'] for r in results]
    edge_sizes = [r['mean_edge_size'] for r in results]
    
    plt.plot(n_inits, mean_Ms, marker='o', label='Final M', linewidth=2)
    plt.plot(n_inits, edge_sizes, marker='s', label='Avg Edge Size', linewidth=2)
    
    plt.xlabel('Initial Hyperedge Count')
    plt.ylabel('Value')
    plt.title('M vs Edge Size Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('F:/hypergraph_bistability/figures/feasibility/self_consistency.png', dpi=150)
    plt.close()


if __name__ == "__main__":
    result = feasibility_landscape()
    self_consistency_test()
    print("\n完成!")
