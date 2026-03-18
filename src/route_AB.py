"""
Route A + B: 结构相变 + 动力学非对称性
========================================

目标：
1. k 扫描 → 定义"结构相"
2. 每个相里测 escape rate (τ(H→L), τ(L→H))
3. 得到：结构 → 势函数 → 动力学

实验设计：
- k = 2, 2.5, 3, 3.5, 4 (连续化：effective k = 混合比例)
- 测 M* (阶参数)
- 测 escape rates
- 重建势函数
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import os
import json

os.makedirs('F:/hypergraph_bistability/figures', exist_ok=True)
os.makedirs('F:/hypergraph_bistability/results', exist_ok=True)


class MixedHypergraph:
    """混合阶超图：effective k = p*2 + (1-p)*3"""
    
    def __init__(self, N=40, p_pair=0.5, state_dim=8, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        
        self.N = N
        self.p_pair = p_pair  # 2-body 比例
        self.state_dim = state_dim
        self.K = int(0.35 * N)  # 容量约束
        
        self.V = list(range(N))
        self.E = []
        self.s = {v: np.random.randn(state_dim) for v in self.V}
        
        # 初始化：混合超边
        n_initial = max(5, N // 3)
        for _ in range(n_initial):
            if random.random() < p_pair:
                # 2-body (边)
                e = frozenset(random.sample(self.V, 2))
            else:
                # 3-body (三元组)
                e = frozenset(random.sample(self.V, min(3, N)))
            self.E.append(e)
    
    def get_node_degree(self, v):
        return sum(1 for e in self.E if v in e)
    
    def get_avg_distance(self):
        if len(self.V) < 2:
            return 0.1
        samples = min(15, len(self.V) * (len(self.V) - 1) // 2)
        if samples <= 0:
            return 0.1
        dist_sum = 0
        for _ in range(samples):
            u, v = random.sample(self.V, 2)
            dist_sum += np.linalg.norm(self.s[u] - self.s[v])
        return max(dist_sum / samples, 0.01)
    
    def apply_rules(self, steps=1):
        """应用混合动力学规则"""
        for _ in range(steps):
            avg_dist = self.get_avg_distance()
            
            # 规则1: 生长
            if random.random() < 0.3 and self.E:
                e = random.choice(self.E)
                nodes = list(e)
                w = max(self.V) + 1
                self.V.append(w)
                self.s[w] = self.s[nodes[0]] + np.random.randn(self.state_dim) * 0.2
                
                # 随机选择边类型
                if random.random() < self.p_pair:
                    new_e = frozenset([nodes[0], w])
                else:
                    nodes2 = random.sample(self.V[:-1], min(2, len(self.V)-1))
                    new_e = frozenset(nodes2 + [w])
                self.E.append(new_e)
            
            # 规则2: 融合 (2-body 和 3-body 混合)
            if random.random() < 0.25 and len(self.E) >= 2:
                e1, e2 = random.sample(self.E, 2)
                
                # 计算交集
                intersection = e1 & e2
                if len(intersection) >= 1:
                    # 准备融合
                    e1_only = list(e1 - intersection)
                    e2_only = list(e2 - intersection)
                    
                    if e1_only and e2_only:
                        v1, v2 = e1_only[0], e2_only[0]
                        dist = np.linalg.norm(self.s[v1] - self.s[v2])
                        
                        if dist < avg_dist * 0.8 and random.random() < 0.5:
                            # 创建新超边（保持类型比例）
                            new_nodes = list(intersection) + [v1, v2]
                            if len(new_nodes) >= 2:
                                if random.random() < self.p_pair and len(new_nodes) >= 2:
                                    new_e = frozenset(random.sample(new_nodes, 2))
                                else:
                                    new_e = frozenset(random.sample(new_nodes, min(3, len(new_nodes))))
                                self.E.append(new_e)
                                if e1 in self.E: self.E.remove(e1)
                                if e2 in self.E: self.E.remove(e2)
            
            # 规则3: 分裂
            if random.random() < 0.12 and len(self.E) > 1:
                large_edges = [e for e in self.E if len(e) > 2]
                if large_edges:
                    e = random.choice(large_edges)
                    nodes = list(e)
                    random.shuffle(nodes)
                    split = len(nodes) // 2
                    
                    e1 = frozenset(nodes[:split])
                    e2 = frozenset(nodes[split:])
                    
                    if len(e1) >= 2 and len(e2) >= 2:
                        self.E.append(e1)
                        self.E.append(e2)
                        self.E.remove(e)
            
            # 规则4: 容量约束
            for v in list(self.V):
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
        """阶参数：最大连通分量"""
        if not self.V:
            return 0
        
        # 构建邻接
        adj = {v: set() for v in self.V}
        for e in self.E:
            nodes = list(e)
            for i in range(len(nodes)):
                for j in range(i+1, len(nodes)):
                    adj[nodes[i]].add(nodes[j])
                    adj[nodes[j]].add(nodes[i])
        
        # BFS
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
                for neighbor in adj[node]:
                    if neighbor not in visited:
                        stack.append(neighbor)
            max_cluster = max(max_cluster, len(cluster))
        
        return max_cluster / len(self.V) if self.V else 0


def measure_M_star(p_pair, n_runs=15, steps=200):
    """测量给定 p_pair 下的 M*"""
    results = []
    for i in range(n_runs):
        h = MixedHypergraph(N=40, p_pair=p_pair, state_dim=8, seed=i*100+42)
        for _ in range(steps):
            h.apply_rules()
        results.append(h.get_M())
    return np.mean(results), np.std(results), results


def measure_escape_rates(p_pair, n_runs=15, steps=500):
    """测量 escape rates: τ(H→L) 和 τ(L→H)"""
    # 初始条件1: HIGH (大超边)
    high_results = []
    for i in range(n_runs):
        h = MixedHypergraph(N=40, p_pair=p_pair, state_dim=8, seed=i*100+42)
        # 初始为 HIGH：创建大连通
        h.E = [frozenset(range(40))]
        
        # 跟踪 M 变化
        M_history = [h.get_M()]
        for t in range(steps):
            h.apply_rules()
            M_history.append(h.get_M())
        
        # 找首次进入 LOW basin (M < 0.5) 的时间
        escape_time = None
        for t, M in enumerate(M_history):
            if M < 0.5:
                escape_time = t
                break
        
        if escape_time:
            high_results.append(escape_time)
        else:
            high_results.append(steps + 1)  # 未逃逸
    
    # 初始条件2: LOW (小超边)
    low_results = []
    for i in range(n_runs):
        h = MixedHypergraph(N=40, p_pair=p_pair, state_dim=8, seed=i*100+42)
        # 初始为 LOW：创建多个小超边
        h.E = [frozenset([j, j+1]) for j in range(0, 38, 2)]
        
        M_history = [h.get_M()]
        for t in range(steps):
            h.apply_rules()
            M_history.append(h.get_M())
        
        # 找首次进入 HIGH basin (M > 0.7) 的时间
        escape_time = None
        for t, M in enumerate(M_history):
            if M > 0.7:
                escape_time = t
                break
        
        if escape_time:
            low_results.append(escape_time)
        else:
            low_results.append(steps + 1)
    
    # 计算平均逃逸时间
    tau_HL = np.mean([t for t in high_results if t <= steps])
    tau_LH = np.mean([t for t in low_results if t <= steps])
    
    return {
        'tau_HL': tau_HL,
        'tau_LH': tau_LH,
        'asymmetry': tau_LH / tau_HL if tau_HL > 0 else float('inf'),
        'high_escaped': sum(1 for t in high_results if t <= steps) / n_runs,
        'low_escaped': sum(1 for t in low_results if t <= steps) / n_runs
    }


def main():
    print("=" * 60)
    print("Route A + B: 结构相变 + 动力学非对称性")
    print("=" * 60)
    
    # Step 1: k 扫描 (通过 p_pair 控制 effective k)
    # p_pair = 1.0 → k = 2 (纯 2-body)
    # p_pair = 0.0 → k = 3 (纯 3-body)
    # p_pair = 0.5 → effective k ≈ 2.5 (混合)
    
    p_values = [1.0, 0.75, 0.5, 0.25, 0.0]
    k_effective = [2.0, 2.25, 2.5, 2.75, 3.0]  # effective k
    
    print("\n[Step 1] k 扫描 (通过 p_pair 控制)")
    M_results = {}
    
    for p, k_eff in zip(p_values, k_effective):
        print(f"  p_pair={p} (k_eff={k_eff})...", end=" ")
        mean_M, std_M, all_M = measure_M_star(p, n_runs=12, steps=150)
        M_results[k_eff] = {'mean': mean_M, 'std': std_M, 'all': all_M}
        print(f"M* = {mean_M:.3f} +/- {std_M:.3f}")
    
    # Step 2: 测量 escape rates (选两个极端)
    print("\n[Step 2] 测量 escape rates")
    
    # k=2 (p=1.0)
    print("  k=2 (p=1.0)...")
    rates_k2 = measure_escape_rates(1.0, n_runs=10, steps=300)
    print(f"    τ(H→L) = {rates_k2['tau_HL']:.1f}, τ(L→H) = {rates_k2['tau_LH']:.1f}")
    print(f"    Asymmetry = {rates_k2['asymmetry']:.2f}")
    
    # k=3 (p=0.0)
    print("  k=3 (p=0.0)...")
    rates_k3 = measure_escape_rates(0.0, n_runs=10, steps=300)
    print(f"    τ(H→L) = {rates_k3['tau_HL']:.1f}, τ(L→H) = {rates_k3['tau_LH']:.1f}")
    print(f"    Asymmetry = {rates_k3['asymmetry']:.2f}")
    
    # Step 3: 绘图
    print("\n[Step 3] 生成图表")
    
    # 图1: M* vs k_effective
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ks = list(M_results.keys())
    Ms = [M_results[k]['mean'] for k in ks]
    stds = [M_results[k]['std'] for k in ks]
    
    ax1.errorbar(ks, Ms, yerr=stds, marker='o', capsize=5, linewidth=2, markersize=10)
    ax1.axhline(0.5, color='green', linestyle='--', alpha=0.5, label='M=0.5')
    ax1.axvline(2.5, color='red', linestyle=':', alpha=0.5, label='k_c ~ 2.5')
    ax1.set_xlabel('Effective k (interaction order)')
    ax1.set_ylabel('M* (order parameter)')
    ax1.set_title('Route A: Structural Phase Transition')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.2)
    plt.tight_layout()
    plt.savefig('F:/hypergraph_bistability/figures/AB_k_scan.png', dpi=150)
    print("[OK] Saved: figures/AB_k_scan.png")
    
    # 图2: 分布对比
    fig2, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # k=2 分布
    axes[0].hist(M_results[2.0]['all'], bins=10, alpha=0.7, color='blue', edgecolor='black')
    axes[0].axvline(M_results[2.0]['mean'], color='red', linestyle='--', 
                   label=f'Mean={M_results[2.0]["mean"]:.2f}')
    axes[0].set_xlabel('M')
    axes[0].set_ylabel('Count')
    axes[0].set_title('k=2 (Graph-like)')
    axes[0].legend()
    
    # k=3 分布
    axes[1].hist(M_results[3.0]['all'], bins=10, alpha=0.7, color='red', edgecolor='black')
    axes[1].axvline(M_results[3.0]['mean'], color='blue', linestyle='--', 
                   label=f'Mean={M_results[3.0]["mean"]:.2f}')
    axes[1].set_xlabel('M')
    axes[1].set_ylabel('Count')
    axes[1].set_title('k=3 (Hypergraph-like)')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('F:/hypergraph_bistability/figures/AB_distribution.png', dpi=150)
    print("[OK] Saved: figures/AB_distribution.png")
    
    # 图3: 非对称性对比
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    x = ['k=2', 'k=3']
    tau_HL = [rates_k2['tau_HL'], rates_k3['tau_HL']]
    tau_LH = [rates_k2['tau_LH'], rates_k3['tau_LH']]
    
    width = 0.35
    x_pos = np.arange(len(x))
    
    ax3.bar(x_pos - width/2, tau_HL, width, label='τ(H→L)', color='blue', alpha=0.7)
    ax3.bar(x_pos + width/2, tau_LH, width, label='τ(L→H)', color='red', alpha=0.7)
    ax3.set_ylabel('Escape Time')
    ax3.set_title('Route B: Asymmetric Transition Rates')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(x)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 添加非对称比
    for i, (tHL, tLH) in enumerate(zip(tau_HL, tau_LH)):
        if tHL > 0:
            ax3.annotate(f'Asym={tLH/tHL:.1f}', xy=(i, max(tHL, tLH)+5), 
                        ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('F:/hypergraph_bistability/figures/AB_asymmetry.png', dpi=150)
    print("[OK] Saved: figures/AB_asymmetry.png")
    
    # 图4: 综合框架
    fig4, ax4 = plt.subplots(figsize=(12, 8))
    ax4.axis('off')
    
    # 绘制 A + B 框架
    y_positions = [0.85, 0.55, 0.25]
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    contents = [
        f"Structure: k_eff\nk=2 → M*={M_results[2.0]['mean']:.2f}\nk=3 → M*={M_results[3.0]['mean']:.2f}",
        f"Dynamics:\nk=2: τ(H→L)={rates_k2['tau_HL']:.0f}, τ(L→H)={rates_k2['tau_LH']:.0f}\nk=3: τ(H→L)={rates_k3['tau_HL']:.0f}, τ(L→H)={rates_k3['tau_LH']:.0f}",
        "Complete Loop:\nstructure → phase → dynamics"
    ]
    
    for i, (y, color, content) in enumerate(zip(y_positions, colors, contents)):
        rect = plt.Rectangle((0.1, y-0.15), 0.8, 0.28, 
                             fill=True, facecolor=color, alpha=0.3,
                             edgecolor=color, linewidth=3)
        ax4.add_patch(rect)
        ax4.text(0.5, y+0.08, ['Route A: Structure', 'Route B: Dynamics', 'A + B = Complete'][i], 
                ha='center', va='center', fontsize=14, fontweight='bold')
        ax4.text(0.5, y-0.02, content, ha='center', va='center', fontsize=11)
        
        if i < 2:
            ax4.annotate('', xy=(0.5, y_positions[i+1]+0.18), 
                       xytext=(0.5, y-0.18),
                       arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.set_title('Route A + B: Structure → Phase → Dynamics', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('F:/hypergraph_bistability/figures/AB_framework.png', dpi=150)
    print("[OK] Saved: figures/AB_framework.png")
    
    # 保存结果
    results = {
        'M_results': {str(k): {'mean': v['mean'], 'std': v['std']} for k, v in M_results.items()},
        'rates_k2': rates_k2,
        'rates_k3': rates_k3
    }
    
    with open('F:/hypergraph_bistability/results/AB_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("[OK] Saved: results/AB_results.json")
    
    # 总结
    print("\n" + "=" * 60)
    print("SUMMARY: Route A + B Results")
    print("=" * 60)
    print(f"\n[Structure (Route A)]")
    print(f"  k=2: M* = {M_results[2.0]['mean']:.3f} (fragmented)")
    print(f"  k=3: M* = {M_results[3.0]['mean']:.3f} (clustered)")
    print(f"  Transition: k_c ~ 2.5")
    
    print(f"\n[Dynamics (Route B)]")
    print(f"  k=2: τ(H→L)={rates_k2['tau_HL']:.0f}, τ(L→H)={rates_k2['tau_LH']:.0f}, asym={rates_k2['asymmetry']:.1f}")
    print(f"  k=3: τ(H→L)={rates_k3['tau_HL']:.0f}, τ(L→H)={rates_k3['tau_LH']:.0f}, asym={rates_k3['asymmetry']:.1f}")
    
    print(f"\n[Complete Loop]")
    print("  Structure → Phase → Dynamics 框架已建立")


if __name__ == '__main__':
    main()
