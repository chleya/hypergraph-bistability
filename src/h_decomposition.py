"""
H 结构分解：Core vs Periphery
=============================

目标：验证"复杂性来自外围"假设
- H_core(t): 最大 cluster 内部
- H_periphery(t): 其余部分
- H_total(t): 整体
"""

import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

os.makedirs('F:/hypergraph_bistability/figures', exist_ok=True)


class HypergraphDecompose:
    def __init__(self, N=50, p_pair=0.5, seed=42):
        random.seed(seed)
        
        self.N = N
        self.p_pair = p_pair
        self.K = int(0.35 * N)
        self.V = list(range(N))
        self.E = []
        
        for _ in range(15):
            if random.random() < p_pair:
                e = frozenset(random.sample(self.V, 2))
            else:
                e = frozenset(random.sample(self.V, min(3, N)))
            self.E.append(e)
    
    def get_node_degree(self, v):
        return sum(1 for e in self.E if v in e)
    
    def get_clusters(self):
        """获取所有 cluster 和最大 cluster"""
        if not self.V:
            return [], set()
        
        adj = {v: set() for v in self.V}
        for e in self.E:
            nodes = list(e)
            for i in range(len(nodes)):
                for j in range(i+1, len(nodes)):
                    adj[nodes[i]].add(nodes[j])
                    adj[nodes[j]].add(nodes[i])
        
        visited = set()
        clusters = []
        
        for s in self.V:
            if s in visited:
                continue
            stack = [s]
            c = set()
            while stack:
                n = stack.pop()
                if n in visited:
                    continue
                visited.add(n)
                c.add(n)
                for nb in adj[n]:
                    if nb not in visited:
                        stack.append(nb)
            clusters.append(c)
        
        if clusters:
            max_cluster = max(clusters, key=len)
        else:
            max_cluster = set()
        
        return clusters, max_cluster
    
    def apply_rules(self, steps=1):
        for _ in range(steps):
            # Growth
            if random.random() < 0.3 and self.E:
                e = random.choice(self.E)
                v = random.choice(list(e))
                w = max(self.V) + 1
                self.V.append(w)
                
                if random.random() < self.p_pair:
                    new_e = frozenset([v, w])
                else:
                    new_e = frozenset([v, w, random.choice(self.V[:-1])])
                self.E.append(new_e)
            
            # Fusion
            if random.random() < 0.25 and len(self.E) >= 2:
                e1, e2 = random.sample(self.E, 2)
                if len(e1 & e2) >= 1 and random.random() < 0.5:
                    new_e = frozenset(e1 | e2)
                    if len(new_e) >= 2:
                        self.E.append(new_e)
                        if e1 in self.E: self.E.remove(e1)
                        if e2 in self.E: self.E.remove(e2)
            
            # Split
            if random.random() < 0.12:
                large = [e for e in self.E if len(e) > 2]
                if large:
                    e = random.choice(large)
                    nodes = list(e)
                    if len(nodes) >= 4:
                        split = len(nodes) // 2
                        self.E.append(frozenset(nodes[:split]))
                        self.E.append(frozenset(nodes[split:]))
                        self.E.remove(e)
            
            # K cap
            for v in list(self.V):
                d = self.get_node_degree(v)
                if d > self.K:
                    excess = d - self.K
                    v_edges = [e for e in self.E if v in e]
                    for e in v_edges[:excess]:
                        if len(e) > 2:
                            new_e = e - {v}
                            if len(new_e) >= 2:
                                self.E.remove(e)
                                self.E.append(new_e)
    
    def get_entropy_for_nodes(self, node_set):
        """计算指定节点集合的超边大小分布熵"""
        # 找出涉及这些节点的 hyperedges
        relevant_edges = [e for e in self.E if len(e & node_set) > 0]
        
        if not relevant_edges:
            return 0
        
        # 超边大小分布
        sizes = [len(e) for e in relevant_edges]
        counts = {}
        for s in sizes:
            counts[s] = counts.get(s, 0) + 1
        total = sum(counts.values())
        
        # Shannon 熵
        H = 0
        for c in counts.values():
            p = c / total
            if p > 0:
                H -= p * np.log2(p)
        
        return H
    
    def get_M(self):
        """Order parameter"""
        if not self.V:
            return 0
        
        adj = {v: set() for v in self.V}
        for e in self.E:
            nodes = list(e)
            for i in range(len(nodes)):
                for j in range(i+1, len(nodes)):
                    adj[nodes[i]].add(nodes[j])
                    adj[nodes[j]].add(nodes[i])
        
        visited = set()
        max_c = 0
        
        for s in self.V:
            if s in visited:
                continue
            stack = [s]
            c = set()
            while stack:
                n = stack.pop()
                if n in visited:
                    continue
                visited.add(n)
                c.add(n)
                for nb in adj[n]:
                    if nb not in visited:
                        stack.append(nb)
            max_c = max(max_c, len(c))
        
        return max_c / len(self.V)
    
    def get_total_entropy(self):
        """整体 H"""
        return self.get_entropy_for_nodes(set(self.V))


def run_decomposition(p_pair=0.5, n_steps=100, seed=42):
    """运行轨迹，分解 H"""
    h = HypergraphDecompose(N=50, p_pair=p_pair, seed=seed)
    
    H_total_history = []
    H_core_history = []
    H_periphery_history = []
    M_history = []
    core_size_history = []
    
    for t in range(n_steps):
        h.apply_rules()
        
        # 获取 cluster 信息
        clusters, max_cluster = h.get_clusters()
        
        # 计算各个 H
        H_total = h.get_total_entropy()
        
        if max_cluster:
            H_core = h.get_entropy_for_nodes(max_cluster)
            periphery = set(h.V) - max_cluster
            H_periphery = h.get_entropy_for_nodes(periphery) if periphery else 0
            core_size = len(max_cluster) / len(h.V)
        else:
            H_core = 0
            H_periphery = 0
            core_size = 0
        
        M = h.get_M()
        
        H_total_history.append(H_total)
        H_core_history.append(H_core)
        H_periphery_history.append(H_periphery)
        M_history.append(M)
        core_size_history.append(core_size)
    
    return {
        'H_total': H_total_history,
        'H_core': H_core_history,
        'H_periphery': H_periphery_history,
        'M': M_history,
        'core_size': core_size_history
    }


def main():
    print("=" * 60)
    print("H Decomposition: Core vs Periphery")
    print("=" * 60)
    
    # 运行多条轨迹
    n_runs = 10
    n_steps = 100
    
    print(f"\n[Running] {n_runs} trajectories")
    
    results = []
    for i in range(n_runs):
        r = run_decomposition(p_pair=0.5, n_steps=n_steps, seed=i*100+42)
        results.append(r)
        print(f"  Run {i+1}/{n_runs} done")
    
    # 分析
    print("\n[Analysis]")
    
    # 平均曲线
    H_total_mean = np.mean([r['H_total'] for r in results], axis=0)
    H_total_std = np.std([r['H_total'] for r in results], axis=0)
    
    H_core_mean = np.mean([r['H_core'] for r in results], axis=0)
    H_core_std = np.std([r['H_core'] for r in results], axis=0)
    
    H_periphery_mean = np.mean([r['H_periphery'] for r in results], axis=0)
    H_periphery_std = np.std([r['H_periphery'] for r in results], axis=0)
    
    M_mean = np.mean([r['M'] for r in results], axis=0)
    core_size_mean = np.mean([r['core_size'] for r in results], axis=0)
    
    # 打印关键数值
    print(f"\nH_total: {H_total_mean[0]:.3f} → {H_total_mean[-1]:.3f} (Δ={H_total_mean[-1]-H_total_mean[0]:+.3f})")
    print(f"H_core:  {H_core_mean[0]:.3f} → {H_core_mean[-1]:.3f} (Δ={H_core_mean[-1]-H_core_mean[0]:+.3f})")
    print(f"H_peri:  {H_periphery_mean[0]:.3f} → {H_periphery_mean[-1]:.3f} (Δ={H_periphery_mean[-1]-H_periphery_mean[0]:+.3f})")
    print(f"M:       {M_mean[0]:.3f} → {M_mean[-1]:.3f} (Δ={M_mean[-1]-M_mean[0]:+.3f})")
    
    # 绘图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    t = np.arange(n_steps)
    
    # 1. H 分解
    ax1 = axes[0, 0]
    ax1.plot(t, H_total_mean, 'b-', linewidth=2, label='H_total')
    ax1.fill_between(t, H_total_mean - H_total_std, H_total_mean + H_total_std, alpha=0.2, color='blue')
    ax1.plot(t, H_core_mean, 'r-', linewidth=2, label='H_core')
    ax1.fill_between(t, H_core_mean - H_core_std, H_core_mean + H_core_std, alpha=0.2, color='red')
    ax1.plot(t, H_periphery_mean, 'g-', linewidth=2, label='H_periphery')
    ax1.fill_between(t, H_periphery_mean - H_periphery_std, H_periphery_mean + H_periphery_std, alpha=0.2, color='green')
    ax1.set_xlabel('Time step')
    ax1.set_ylabel('Entropy H')
    ax1.set_title('H Decomposition: Core vs Periphery')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. H 变化量
    ax2 = axes[0, 1]
    dH_total = H_total_mean[-1] - H_total_mean[0]
    dH_core = H_core_mean[-1] - H_core_mean[0]
    dH_peri = H_periphery_mean[-1] - H_periphery_mean[0]
    
    labels = ['H_total', 'H_core', 'H_periphery']
    values = [dH_total, dH_core, dH_peri]
    colors = ['blue', 'red', 'green']
    bars = ax2.bar(labels, values, color=colors, alpha=0.7)
    ax2.axhline(0, color='black', linestyle='-', linewidth=0.5)
    for bar, v in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2, v + 0.01, f'{v:+.3f}', 
                ha='center', va='bottom', fontsize=12)
    ax2.set_ylabel('Delta H (t=99 - t=0)')
    ax2.set_title('H Change Over Time')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. M 和 core size
    ax3 = axes[1, 0]
    ax3.plot(t, M_mean, 'b-', linewidth=2, label='M (order)')
    ax3.plot(t, core_size_mean, 'r--', linewidth=2, label='Core size')
    ax3.set_xlabel('Time step')
    ax3.set_ylabel('Value')
    ax3.set_title('Order Parameter M and Core Size')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 相图 M vs H
    ax4 = axes[1, 1]
    for i, r in enumerate(results):
        ax4.plot(r['H_total'], r['M'], 'o-', alpha=0.2, markersize=2)
    ax4.plot(H_total_mean, M_mean, 'b-', linewidth=2, label='Mean trajectory')
    ax4.scatter([H_total_mean[0]], [M_mean[0]], c='green', s=100, label='Start', zorder=5)
    ax4.scatter([H_total_mean[-1]], [M_mean[-1]], c='red', s=100, label='End', zorder=5)
    ax4.set_xlabel('H_total')
    ax4.set_ylabel('M')
    ax4.set_title('M-H Phase Diagram')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('F:/hypergraph_bistability/figures/H_decomposition.png', dpi=150)
    print("\n[OK] Saved: figures/H_decomposition.png")
    
    # 结论
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    
    if dH_peri > 0 and dH_core < 0:
        print("\n✅ CASE A: 理想结果!")
        print(f"   H_periphery ↑ ({dH_peri:+.3f})")
        print(f"   H_core ↓ ({dH_core:+.3f})")
        print("   → 复杂性来自外围，核心在收缩!")
    elif dH_peri > 0:
        print("\n🟡 CASE B: 部分支持")
        print(f"   H_periphery ↑ ({dH_peri:+.3f})")
        print(f"   H_core {'↑' if dH_core > 0 else '≈'}")
    else:
        print("\n🔴 CASE C: 需要重新分析")
        print(f"   H_total: {dH_total:+.3f}")
        print(f"   H_core: {dH_core:+.3f}")
        print(f"   H_periphery: {dH_peri:+.3f}")


if __name__ == '__main__':
    main()
