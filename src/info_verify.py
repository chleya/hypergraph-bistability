"""
信息论验证实验：H(t) vs M(t) 轨迹
==================================

目标：验证系统是否沿"压缩路径"演化
- H 单调下降？
- M 单调上升？
- H-M 轨迹稳定？
"""

import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

os.makedirs('F:/hypergraph_bistability/figures', exist_ok=True)


class HypergraphEntropy:
    def __init__(self, N=50, p_pair=0.5, seed=42):
        random.seed(seed)
        
        self.N = N
        self.p_pair = p_pair
        self.K = int(0.35 * N)  # k ≈ N * 0.35
        self.V = list(range(N))
        self.E = []
        
        # 初始化：稀疏连接
        for _ in range(15):
            if random.random() < p_pair:
                e = frozenset(random.sample(self.V, 2))
            else:
                e = frozenset(random.sample(self.V, min(3, N)))
            self.E.append(e)
    
    def get_node_degree(self, v):
        return sum(1 for e in self.E if v in e)
    
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
    
    def get_M(self):
        """Order parameter: max faction size / N"""
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
    
    def get_entropy_H(self):
        """计算超图的熵 H"""
        if not self.E:
            return 0
        
        # 超边大小分布
        sizes = [len(e) for e in self.E]
        if not sizes:
            return 0
        
        # 归一化
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
    
    def get_coverage(self):
        """节点覆盖率"""
        if not self.V:
            return 0
        covered = set()
        for e in self.E:
            covered.update(e)
        return len(covered) / len(self.V)
    
    def get_avg_degree(self):
        """平均度数"""
        if not self.V:
            return 0
        degrees = [self.get_node_degree(v) for v in self.V]
        return np.mean(degrees)


def run_trajectory(p_pair=0.5, n_steps=100, seed=42):
    """运行单条轨迹，记录 H(t), M(t)"""
    h = HypergraphEntropy(N=50, p_pair=p_pair, seed=seed)
    
    H_history = []
    M_history = []
    coverage_history = []
    degree_history = []
    
    for t in range(n_steps):
        h.apply_rules()
        H_history.append(h.get_entropy_H())
        M_history.append(h.get_M())
        coverage_history.append(h.get_coverage())
        degree_history.append(h.get_avg_degree())
    
    return {
        'H': H_history,
        'M': M_history,
        'coverage': coverage_history,
        'degree': degree_history
    }


def main():
    print("=" * 60)
    print("Information Theory Verification: H(t) vs M(t)")
    print("=" * 60)
    
    # 运行多条轨迹
    n_runs = 10
    n_steps = 100
    
    all_H = []
    all_M = []
    all_dH = []
    
    results = []
    
    print(f"\n[Running] {n_runs} trajectories, {n_steps} steps each")
    
    for i in range(n_runs):
        r = run_trajectory(p_pair=0.5, n_steps=n_steps, seed=i*100+42)
        results.append(r)
        
        # 收集所有数据点
        all_H.extend(r['H'])
        all_M.extend(r['M'])
        
        # 计算 dH = H(t+1) - H(t)
        for t in range(len(r['H']) - 1):
            all_dH.append(r['H'][t+1] - r['H'][t])
    
    all_H = np.array(all_H)
    all_M = np.array(all_M)
    all_dH = np.array(all_dH)
    
    # 分析
    print("\n[Analysis]")
    
    # 1. H 趋势
    H_mean_by_time = np.mean([r['H'] for r in results], axis=0)
    H_std_by_time = np.std([r['H'] for r in results], axis=0)
    
    print(f"\nH(t):")
    print(f"  t=0: {H_mean_by_time[0]:.3f} ± {H_std_by_time[0]:.3f}")
    print(f"  t=50: {H_mean_by_time[50]:.3f} ± {H_std_by_time[50]:.3f}")
    print(f"  t=99: {H_mean_by_time[-1]:.3f} ± {H_std_by_time[-1]:.3f}")
    print(f"  Change: {H_mean_by_time[-1] - H_mean_by_time[0]:.3f}")
    
    # 2. M 趋势
    M_mean_by_time = np.mean([r['M'] for r in results], axis=0)
    M_std_by_time = np.std([r['M'] for r in results], axis=0)
    
    print(f"\nM(t):")
    print(f"  t=0: {M_mean_by_time[0]:.3f} ± {M_std_by_time[0]:.3f}")
    print(f"  t=50: {M_mean_by_time[50]:.3f} ± {M_std_by_time[50]:.3f}")
    print(f"  t=99: {M_mean_by_time[-1]:.3f} ± {M_std_by_time[-1]:.3f}")
    print(f"  Change: {M_mean_by_time[-1] - M_mean_by_time[0]:.3f}")
    
    # 3. dH 分析
    print(f"\ndH = H(t+1) - H(t):")
    print(f"  Mean dH: {np.mean(all_dH):.4f}")
    print(f"  Std dH: {np.std(all_dH):.4f}")
    print(f"  P(dH < 0): {np.mean(all_dH < 0):.2%}")
    
    # 4. H-M 相关性
    correlation = np.corrcoef(all_H, all_M)[0, 1]
    print(f"\nH-M correlation: {correlation:.3f}")
    
    # 5. 轨迹终点分析
    final_H = [r['H'][-1] for r in results]
    final_M = [r['M'][-1] for r in results]
    print(f"\nFinal state (t=99):")
    print(f"  H: {np.mean(final_H):.3f} ± {np.std(final_H):.3f}")
    print(f"  M: {np.mean(final_M):.3f} ± {np.std(final_M):.3f}")
    
    # 绘图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. H(t) 轨迹
    ax1 = axes[0, 0]
    t = np.arange(n_steps)
    for i, r in enumerate(results):
        ax1.plot(t, r['H'], alpha=0.3, linewidth=0.5)
    ax1.plot(t, H_mean_by_time, 'b-', linewidth=2, label='Mean')
    ax1.fill_between(t, H_mean_by_time - H_std_by_time, 
                     H_mean_by_time + H_std_by_time, alpha=0.2)
    ax1.set_xlabel('Time step')
    ax1.set_ylabel('Entropy H')
    ax1.set_title('H(t) Trajectory')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. M(t) 轨迹
    ax2 = axes[0, 1]
    for i, r in enumerate(results):
        ax2.plot(t, r['M'], alpha=0.3, linewidth=0.5)
    ax2.plot(t, M_mean_by_time, 'r-', linewidth=2, label='Mean')
    ax2.fill_between(t, M_mean_by_time - M_std_by_time,
                     M_mean_by_time + M_std_by_time, alpha=0.2)
    ax2.set_xlabel('Time step')
    ax2.set_ylabel('Order M')
    ax2.set_title('M(t) Trajectory')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. H-M 相图
    ax3 = axes[1, 0]
    for i, r in enumerate(results):
        ax3.plot(r['H'], r['M'], 'o-', alpha=0.3, markersize=2)
    ax3.set_xlabel('Entropy H')
    ax3.set_ylabel('Order M')
    ax3.set_title(f'H-M Phase Diagram (r={correlation:.2f})')
    ax3.grid(True, alpha=0.3)
    
    # 标注起点和终点
    ax3.annotate('Start', (H_mean_by_time[0], M_mean_by_time[0]), 
                fontsize=12, color='green')
    ax3.annotate('End', (H_mean_by_time[-1], M_mean_by_time[-1]),
                fontsize=12, color='red')
    
    # 4. dH 分布
    ax4 = axes[1, 1]
    ax4.hist(all_dH, bins=50, alpha=0.7, edgecolor='black')
    ax4.axvline(0, color='red', linestyle='--', linewidth=2)
    ax4.axvline(np.mean(all_dH), color='blue', linestyle='-', linewidth=2,
               label=f'Mean={np.mean(all_dH):.4f}')
    ax4.set_xlabel('dH = H(t+1) - H(t)')
    ax4.set_ylabel('Count')
    ax4.set_title(f'dH Distribution (P(dH<0)={np.mean(all_dH<0):.1%})')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('F:/hypergraph_bistability/figures/H_M_trajectory.png', dpi=150)
    print("\n[OK] Saved: figures/H_M_trajectory.png")
    
    # 结论
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    
    if np.mean(all_dH) < 0 and np.mean(all_dH < 0) > 0.5:
        print("\n✅ E[dH] < 0: H is monotonically decreasing (Lyapunov-like)")
    else:
        print("\n⚠️ E[dH] is not clearly negative")
    
    if correlation < -0.5:
        print(f"✅ H-M correlation = {correlation:.2f}: Strong negative correlation")
    else:
        print(f"⚠️ H-M correlation = {correlation:.2f}: Weak correlation")
    
    if M_mean_by_time[-1] > M_mean_by_time[0]:
        print(f"✅ M increases over time: {M_mean_by_time[0]:.2f} → {M_mean_by_time[-1]:.2f}")
    else:
        print(f"⚠️ M does not increase")


if __name__ == '__main__':
    main()
