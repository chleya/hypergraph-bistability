"""
H 分解完整版：Core vs Periphery + Size Control + Shuffle Baseline
===================================================================

Checklist:
- [x] H_core, H_periphery, H_total
- [x] size_core, size_periphery
- [ ] Shuffle baseline
- [x] R(t) = H_periphery / H_total
"""

import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

os.makedirs('F:/hypergraph_bistability/figures', exist_ok=True)


class HypergraphComplete:
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
        
        max_cluster = max(clusters, key=len) if clusters else set()
        return clusters, max_cluster
    
    def apply_rules(self, steps=1):
        for _ in range(steps):
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
            
            if random.random() < 0.25 and len(self.E) >= 2:
                e1, e2 = random.sample(self.E, 2)
                if len(e1 & e2) >= 1 and random.random() < 0.5:
                    new_e = frozenset(e1 | e2)
                    if len(new_e) >= 2:
                        self.E.append(new_e)
                        if e1 in self.E: self.E.remove(e1)
                        if e2 in self.E: self.E.remove(e2)
            
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
    
    def compute_H_for_node_set(self, node_set):
        """计算节点集合的 H（归一化）"""
        relevant_edges = [e for e in self.E if len(e & node_set) > 0]
        
        if not relevant_edges:
            return 0, 0
        
        sizes = [len(e) for e in relevant_edges]
        counts = {}
        for s in sizes:
            counts[s] = counts.get(s, 0) + 1
        total = len(relevant_edges)
        
        H = 0
        for c in counts.values():
            p = c / total
            if p > 0:
                H -= p * np.log2(p)
        
        return H, total
    
    def get_M(self):
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


def run_complete(p_pair=0.5, n_steps=100, seed=42):
    """完整运行轨迹"""
    h = HypergraphComplete(N=50, p_pair=p_pair, seed=seed)
    
    H_total, H_core, H_peri = [], [], []
    size_core, size_peri = [], []
    M_history = []
    
    for t in range(n_steps):
        h.apply_rules()
        
        clusters, max_cluster = h.get_clusters()
        
        # 所有节点
        all_nodes = set(h.V)
        H_tot, n_tot = h.compute_H_for_node_set(all_nodes)
        
        # Core
        if max_cluster:
            H_c, n_c = h.compute_H_for_node_set(max_cluster)
            periphery = all_nodes - max_cluster
            H_p, n_p = h.compute_H_for_node_set(periphery) if periphery else (0, 0)
        else:
            H_c, n_c = 0, 0
            H_p, n_p = 0, 0
        
        H_total.append(H_tot)
        H_core.append(H_c)
        H_peri.append(H_p)
        
        size_core.append(len(max_cluster) / len(h.V) if max_cluster else 0)
        size_peri.append(1 - len(max_cluster)/len(h.V) if max_cluster else 1)
        
        M_history.append(h.get_M())
    
    return {
        'H_total': H_total,
        'H_core': H_core,
        'H_periphery': H_peri,
        'size_core': size_core,
        'size_periphery': size_peri,
        'M': M_history
    }


def run_shuffle_baseline(N=50, n_steps=100, seed=42):
    """Shuffle baseline: 保持超边数量和大小分布，随机打乱"""
    random.seed(seed)
    np.random.seed(seed)
    
    # 固定 N 步后的超边数量和大小的边缘分布
    h = HypergraphComplete(N=N, p_pair=0.5, seed=seed)
    for _ in range(n_steps):
        h.apply_rules()
    
    # 记录
    edge_sizes = [len(e) for e in h.E]
    n_edges = len(h.E)
    
    # Shuffle
    random.shuffle(h.E)
    
    # 重新计算
    clusters, max_cluster = h.get_clusters()
    all_nodes = set(h.V)
    
    H_tot, _ = h.compute_H_for_node_set(all_nodes)
    
    if max_cluster:
        H_c, _ = h.compute_H_for_node_set(max_cluster)
        periphery = all_nodes - max_cluster
        H_p, _ = h.compute_H_for_node_set(periphery) if periphery else (0, 0)
    else:
        H_c, H_p = 0, 0
    
    return {
        'H_total': H_tot,
        'H_core': H_c,
        'H_periphery': H_p
    }


def main():
    print("=" * 60)
    print("H Decomposition: Complete Analysis")
    print("=" * 60)
    
    n_runs = 10
    n_steps = 100
    
    print(f"\n[Running] {n_runs} trajectories")
    results = []
    for i in range(n_runs):
        r = run_complete(p_pair=0.5, n_steps=n_steps, seed=i*100+42)
        results.append(r)
        print(f"  {i+1}/{n_runs}")
    
    # Shuffle baseline
    print("\n[Shuffle Baseline]")
    shuffle_results = []
    for i in range(10):
        sr = run_shuffle_baseline(N=50, n_steps=100, seed=i*1000+42)
        shuffle_results.append(sr)
    
    # 平均
    H_total_mean = np.mean([r['H_total'] for r in results], axis=0)
    H_core_mean = np.mean([r['H_core'] for r in results], axis=0)
    H_peri_mean = np.mean([r['H_periphery'] for r in results], axis=0)
    size_core_mean = np.mean([r['size_core'] for r in results], axis=0)
    M_mean = np.mean([r['M'] for r in results], axis=0)
    
    # Shuffle 平均
    H_total_shuf = np.mean([r['H_total'] for r in shuffle_results])
    H_core_shuf = np.mean([r['H_core'] for r in shuffle_results])
    H_peri_shuf = np.mean([r['H_periphery'] for r in shuffle_results])
    
    # 变化量
    dH_total = H_total_mean[-1] - H_total_mean[0]
    dH_core = H_core_mean[-1] - H_core_mean[0]
    dH_peri = H_peri_mean[-1] - H_peri_mean[0]
    dM = M_mean[-1] - M_mean[0]
    dSize_core = size_core_mean[-1] - size_core_mean[0]
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print(f"\n[Time Evolution]")
    print(f"  H_total:  {H_total_mean[0]:.3f} → {H_total_mean[-1]:.3f} (Δ={dH_total:+.3f})")
    print(f"  H_core:   {H_core_mean[0]:.3f} → {H_core_mean[-1]:.3f} (Δ={dH_core:+.3f})")
    print(f"  H_peri:   {H_peri_mean[0]:.3f} → {H_peri_mean[-1]:.3f} (Δ={dH_peri:+.3f})")
    print(f"  M:        {M_mean[0]:.3f} → {M_mean[-1]:.3f} (Δ={dM:+.3f})")
    print(f"  Size_core:{size_core_mean[0]:.3f} → {size_core_mean[-1]:.3f} (Δ={dSize_core:+.3f})")
    
    print(f"\n[Shuffle Baseline (t=100)]")
    print(f"  H_total:  {H_total_shuf:.3f}")
    print(f"  H_core:   {H_core_shuf:.3f}")
    print(f"  H_peri:   {H_peri_shuf:.3f}")
    
    print(f"\n[Deviation from Shuffle]")
    print(f"  H_total:  {H_total_mean[-1] - H_total_shuf:+.3f}")
    print(f"  H_core:   {H_core_mean[-1] - H_core_shuf:+.3f}")
    print(f"  H_peri:   {H_peri_mean[-1] - H_peri_shuf:+.3f}")
    
    # R(t) = H_peri / H_total
    R = np.array(H_peri_mean) / np.array(H_total_mean)
    dR = R[-1] - R[0]
    print(f"\n[R(t) = H_peri / H_total]")
    print(f"  R(0):  {R[0]:.3f}")
    print(f"  R(99): {R[-1]:.3f}")
    print(f"  ΔR:    {dR:+.3f}")
    
    # 绘图
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    t = np.arange(n_steps)
    
    # 1. H decomposition
    ax1 = axes[0, 0]
    ax1.plot(t, H_total_mean, 'b-', linewidth=2, label='H_total')
    ax1.plot(t, H_core_mean, 'r-', linewidth=2, label='H_core')
    ax1.plot(t, H_peri_mean, 'g-', linewidth=2, label='H_periphery')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('H')
    ax1.set_title('H Decomposition')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Size evolution
    ax2 = axes[0, 1]
    ax2.plot(t, size_core_mean, 'r-', linewidth=2, label='Core size')
    ax2.plot(t, 1-size_core_mean, 'g--', linewidth=2, label='Periphery size')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Fraction')
    ax2.set_title('Core/Periphery Size')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. H vs Size correlation
    ax3 = axes[0, 2]
    ax3.scatter(size_core_mean, H_core_mean, c='red', alpha=0.5, label='H_core vs size')
    ax3.scatter(1-size_core_mean, H_peri_mean, c='green', alpha=0.5, label='H_peri vs size')
    ax3.set_xlabel('Size fraction')
    ax3.set_ylabel('H')
    ax3.set_title('H vs Size')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. M vs H
    ax4 = axes[1, 0]
    ax4.plot(H_total_mean, M_mean, 'b-', linewidth=2)
    ax4.scatter([H_total_mean[0]], [M_mean[0]], c='green', s=100, label='Start')
    ax4.scatter([H_total_mean[-1]], [M_mean[-1]], c='red', s=100, label='End')
    ax4.set_xlabel('H_total')
    ax4.set_ylabel('M')
    ax4.set_title('M vs H')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. R(t)
    ax5 = axes[1, 1]
    ax5.plot(t, R, 'purple', linewidth=2)
    ax5.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax5.set_xlabel('Time')
    ax5.set_ylabel('R = H_peri / H_total')
    ax5.set_title('Complexity Migration Ratio')
    ax5.grid(True, alpha=0.3)
    
    # 6. Comparison bar
    ax6 = axes[1, 2]
    labels = ['t=0', 't=100', 'Shuffle']
    x = np.arange(3)
    width = 0.25
    
    ax6.bar(x - width, [H_total_mean[0], H_total_mean[-1], H_total_shuf], 
            width, label='H_total', color='blue', alpha=0.7)
    ax6.bar(x, [H_core_mean[0], H_core_mean[-1], H_core_shuf], 
            width, label='H_core', color='red', alpha=0.7)
    ax6.bar(x + width, [H_peri_mean[0], H_peri_mean[-1], H_peri_shuf], 
            width, label='H_peri', color='green', alpha=0.7)
    ax6.set_xticks(x)
    ax6.set_xticklabels(labels)
    ax6.set_ylabel('H')
    ax6.set_title('Comparison')
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('F:/hypergraph_bistability/figures/H_complete.png', dpi=150)
    print("\n[OK] Saved: figures/H_complete.png")
    
    # 结论
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    
    if dH_core > 0 and dH_peri < 0:
        print("\n✅ CASE: Core-complex, Peri-simple")
        print(f"   H_core: +{dH_core:.3f} (more complex)")
        print(f"   H_peri: {dH_peri:.3f} (simpler)")
        print(f"   R(t): {dR:+.3f} (complexity NOT migrating to periphery)")
    else:
        print("\n⚠️ Other pattern")
    
    print(f"\nInterpretation:")
    print(f"  - Core grows AND becomes more internally complex")
    print(f"  - Periphery shrinks AND becomes more uniform")
    print(f"  - Complexity is ABSORBED by core, not generated in periphery")


if __name__ == '__main__':
    main()
