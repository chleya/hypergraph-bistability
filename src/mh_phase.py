"""
M-H 相图：轨迹分析
===================

目标：画出 (M, H_total), (M, H_core), (M, H_peri) 三条轨迹
验证是否存在"结构化凝聚路径"
"""

import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

os.makedirs('F:/hypergraph_bistability/figures', exist_ok=True)


class HypergraphMH:
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
    
    def compute_H(self, node_set):
        relevant_edges = [e for e in self.E if len(e & node_set) > 0]
        
        if not relevant_edges:
            return 0
        
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
        
        return H
    
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


def run_trajectory(p_pair=0.5, n_steps=100, seed=42):
    h = HypergraphMH(N=50, p_pair=p_pair, seed=seed)
    
    M_history = []
    H_total_history = []
    H_core_history = []
    H_peri_history = []
    
    for t in range(n_steps):
        h.apply_rules()
        
        clusters, max_cluster = h.get_clusters()
        all_nodes = set(h.V)
        
        M = h.get_M()
        H_total = h.compute_H(all_nodes)
        
        if max_cluster:
            H_core = h.compute_H(max_cluster)
            periphery = all_nodes - max_cluster
            H_peri = h.compute_H(periphery) if periphery else 0
        else:
            H_core = 0
            H_peri = 0
        
        M_history.append(M)
        H_total_history.append(H_total)
        H_core_history.append(H_core)
        H_peri_history.append(H_peri)
    
    return {
        'M': M_history,
        'H_total': H_total_history,
        'H_core': H_core_history,
        'H_periphery': H_peri_history
    }


def main():
    print("=" * 60)
    print("M-H Phase Diagram")
    print("=" * 60)
    
    n_runs = 15
    n_steps = 100
    
    print(f"\n[Running] {n_runs} trajectories")
    results = []
    for i in range(n_runs):
        r = run_trajectory(p_pair=0.5, n_steps=n_steps, seed=i*100+42)
        results.append(r)
        print(f"  {i+1}/{n_runs}")
    
    # 平均
    M_mean = np.mean([r['M'] for r in results], axis=0)
    H_total_mean = np.mean([r['H_total'] for r in results], axis=0)
    H_core_mean = np.mean([r['H_core'] for r in results], axis=0)
    H_peri_mean = np.mean([r['H_periphery'] for r in results], axis=0)
    
    t = np.arange(n_steps)
    
    # 绘图
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. M-H_total 相图
    ax1 = axes[0, 0]
    for i, r in enumerate(results):
        ax1.plot(r['M'], r['H_total'], '-', alpha=0.15, linewidth=0.5, color='blue')
    ax1.plot(M_mean, H_total_mean, 'b-', linewidth=2.5, label='Mean trajectory')
    ax1.scatter([M_mean[0]], [H_total_mean[0]], c='green', s=150, marker='o', 
                label='Start', zorder=5, edgecolors='black')
    ax1.scatter([M_mean[-1]], [H_total_mean[-1]], c='red', s=150, marker='s', 
                label='End', zorder=5, edgecolors='black')
    
    # 箭头
    ax1.annotate('', xy=(M_mean[-1], H_total_mean[-1]), 
                xytext=(M_mean[0], H_total_mean[0]),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    ax1.set_xlabel('M (Order Parameter)', fontsize=12)
    ax1.set_ylabel('H_total', fontsize=12)
    ax1.set_title('M vs H_total Phase Diagram', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. M-H_core 相图
    ax2 = axes[0, 1]
    for i, r in enumerate(results):
        ax2.plot(r['M'], r['H_core'], '-', alpha=0.15, linewidth=0.5, color='red')
    ax2.plot(M_mean, H_core_mean, 'r-', linewidth=2.5, label='Mean trajectory')
    ax2.scatter([M_mean[0]], [H_core_mean[0]], c='green', s=150, marker='o', 
                label='Start', zorder=5, edgecolors='black')
    ax2.scatter([M_mean[-1]], [H_core_mean[-1]], c='red', s=150, marker='s', 
                label='End', zorder=5, edgecolors='black')
    ax2.annotate('', xy=(M_mean[-1], H_core_mean[-1]), 
                xytext=(M_mean[0], H_core_mean[0]),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax2.set_xlabel('M (Order Parameter)', fontsize=12)
    ax2.set_ylabel('H_core', fontsize=12)
    ax2.set_title('M vs H_core Phase Diagram', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 3. M-H_periphery 相图
    ax3 = axes[1, 0]
    for i, r in enumerate(results):
        ax3.plot(r['M'], r['H_periphery'], '-', alpha=0.15, linewidth=0.5, color='green')
    ax3.plot(M_mean, H_peri_mean, 'g-', linewidth=2.5, label='Mean trajectory')
    ax3.scatter([M_mean[0]], [H_peri_mean[0]], c='green', s=150, marker='o', 
                label='Start', zorder=5, edgecolors='black')
    ax3.scatter([M_mean[-1]], [H_peri_mean[-1]], c='red', s=150, marker='s', 
                label='End', zorder=5, edgecolors='black')
    ax3.annotate('', xy=(M_mean[-1], H_peri_mean[-1]), 
                xytext=(M_mean[0], H_peri_mean[0]),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax3.set_xlabel('M (Order Parameter)', fontsize=12)
    ax3.set_ylabel('H_periphery', fontsize=12)
    ax3.set_title('M vs H_periphery Phase Diagram', fontsize=14)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 4. 三条线叠加
    ax4 = axes[1, 1]
    ax4.plot(M_mean, H_total_mean, 'b-', linewidth=2.5, label='H_total')
    ax4.plot(M_mean, H_core_mean, 'r-', linewidth=2.5, label='H_core')
    ax4.plot(M_mean, H_peri_mean, 'g-', linewidth=2.5, label='H_periphery')
    ax4.set_xlabel('M (Order Parameter)', fontsize=12)
    ax4.set_ylabel('H', fontsize=12)
    ax4.set_title('All H Components vs M', fontsize=14)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('F:/hypergraph_bistability/figures/MH_phase_diagram.png', dpi=150)
    print("\n[OK] Saved: figures/MH_phase_diagram.png")
    
    # 分析轨迹特征
    print("\n" + "=" * 60)
    print("TRAJECTORY ANALYSIS")
    print("=" * 60)
    
    # 计算相关性
    corr_total = np.corrcoef(M_mean, H_total_mean)[0, 1]
    corr_core = np.corrcoef(M_mean, H_core_mean)[0, 1]
    corr_peri = np.corrcoef(M_mean, H_peri_mean)[0, 1]
    
    print(f"\n[M-H Correlations]")
    print(f"  M vs H_total:   {corr_total:+.3f}")
    print(f"  M vs H_core:    {corr_core:+.3f}")
    print(f"  M vs H_peri:    {corr_peri:+.3f}")
    
    print(f"\n[Trajectory Pattern]")
    print(f"  H_total:  {H_total_mean[0]:.3f} → {H_total_mean[-1]:.3f} ({H_total_mean[-1]-H_total_mean[0]:+.3f})")
    print(f"  H_core:   {H_core_mean[0]:.3f} → {H_core_mean[-1]:.3f} ({H_core_mean[-1]-H_core_mean[0]:+.3f})")
    print(f"  H_peri:   {H_peri_mean[0]:.3f} → {H_peri_mean[-1]:.3f} ({H_peri_mean[-1]-H_peri_mean[0]:+.3f})")
    
    # 判断
    if corr_core > 0 and corr_peri < 0:
        print(f"\n✅ ORDER-COMPLEXITY SEPARATION CONFIRMED!")
        print(f"   M increases → H_core increases (complexity into core)")
        print(f"   M increases → H_periphery decreases (simplicity in periphery)")
    else:
        print(f"\n⚠️ Different pattern")


if __name__ == '__main__':
    main()
