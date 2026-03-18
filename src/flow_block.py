"""
Flow-level Intervention: Block peri → core 流动
===============================================

目标：阻断外围节点并入核心的流动
- 验证：如果切断复杂性流入通道，H_core 是否还增长

方案 B：
- 禁止外围节点并入核心
- 或大幅降低 merge probability when peri → core
"""

import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

os.makedirs('F:/hypergraph_bistability/figures', exist_ok=True)


class HypergraphBlockFlow:
    """阻断 peri → core 流动的超图"""
    def __init__(self, N=50, p_pair=0.5, seed=42, block_prob=0.95):
        random.seed(seed)
        
        self.N = N
        self.p_pair = p_pair
        self.K = int(0.35 * N)
        self.block_prob = block_prob  # 阻断概率
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
            
            # Fusion (with flow blocking!)
            if random.random() < 0.25 and len(self.E) >= 2:
                e1, e2 = random.sample(self.E, 2)
                if len(e1 & e2) >= 1 and random.random() < 0.5:
                    # 检查是否会 peri → core
                    clusters, max_cluster = self.get_clusters()
                    
                    # e1 和 e2 的节点
                    nodes1 = set(e1)
                    nodes2 = set(e2)
                    new_nodes = nodes1 | nodes2
                    
                    # 检查是否有外围节点并入核心
                    is_peri_to_core = False
                    if max_cluster:
                        # 如果一个超边包含核心节点，另一个包含外围节点，且合并后外围节点进入核心
                        e1_in_core = bool(nodes1 & max_cluster)
                        e2_in_core = bool(nodes2 & max_cluster)
                        e1_in_peri = bool(nodes1 - max_cluster)
                        e2_in_peri = bool(nodes2 - max_cluster)
                        
                        # 如果是 peri → core 合并
                        if (e1_in_core and e2_in_peri) or (e1_in_peri and e2_in_core):
                            is_peri_to_core = True
                    
                    # 如果是 peri → core，可能阻断
                    if is_peri_to_core and random.random() < self.block_prob:
                        # 阻断！不执行融合，或者执行但不并入核心
                        pass
                    else:
                        # 正常融合
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


class HypergraphNormal:
    """正常系统"""
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


def run_block_experiment(n_runs=8, n_steps=80):
    """对比：正常 vs 阻断流动"""
    
    print("=" * 60)
    print("Flow-Level Intervention: Block peri -> core")
    print("=" * 60)
    
    # 正常系统
    print("\n[Running] Normal system")
    normal_results = []
    for i in range(n_runs):
        h = HypergraphNormal(N=50, p_pair=0.5, seed=i*100+42)
        M_hist, H_core_hist, H_peri_hist = [], [], []
        for t in range(n_steps):
            h.apply_rules()
            clusters, max_cluster = h.get_clusters()
            all_nodes = set(h.V)
            
            M = h.get_M()
            if max_cluster:
                H_core = h.compute_H(max_cluster)
                periphery = all_nodes - max_cluster
                H_peri = h.compute_H(periphery) if periphery else 0
            else:
                H_core, H_peri = 0, 0
            
            M_hist.append(M)
            H_core_hist.append(H_core)
            H_peri_hist.append(H_peri)
        normal_results.append({'M': M_hist, 'H_core': H_core_hist, 'H_peri': H_peri_hist})
        print(f"  {i+1}/{n_runs}")
    
    # 阻断流动系统
    print("\n[Running] Blocked flow system (block_prob=0.95)")
    blocked_results = []
    for i in range(n_runs):
        h = HypergraphBlockFlow(N=50, p_pair=0.5, seed=i*100+42, block_prob=0.95)
        M_hist, H_core_hist, H_peri_hist = [], [], []
        for t in range(n_steps):
            h.apply_rules()
            clusters, max_cluster = h.get_clusters()
            all_nodes = set(h.V)
            
            M = h.get_M()
            if max_cluster:
                H_core = h.compute_H(max_cluster)
                periphery = all_nodes - max_cluster
                H_peri = h.compute_H(periphery) if periphery else 0
            else:
                H_core, H_peri = 0, 0
            
            M_hist.append(M)
            H_core_hist.append(H_core)
            H_peri_hist.append(H_peri)
        blocked_results.append({'M': M_hist, 'H_core': H_core_hist, 'H_peri': H_peri_hist})
        print(f"  {i+1}/{n_runs}")
    
    # 分析
    print("\n[Analysis]")
    
    # 正常
    M_normal = np.mean([r['M'] for r in normal_results], axis=0)
    H_core_normal = np.mean([r['H_core'] for r in normal_results], axis=0)
    H_peri_normal = np.mean([r['H_peri'] for r in normal_results], axis=0)
    
    # 阻断
    M_blocked = np.mean([r['M'] for r in blocked_results], axis=0)
    H_core_blocked = np.mean([r['H_core'] for r in blocked_results], axis=0)
    H_peri_blocked = np.mean([r['H_peri'] for r in blocked_results], axis=0)
    
    print(f"\n[Normal System]")
    print(f"  M:       {M_normal[0]:.3f} → {M_normal[-1]:.3f} (Δ={M_normal[-1]-M_normal[0]:+.3f})")
    print(f"  H_core:  {H_core_normal[0]:.3f} → {H_core_normal[-1]:.3f} (Δ={H_core_normal[-1]-H_core_normal[0]:+.3f})")
    print(f"  H_peri:  {H_peri_normal[0]:.3f} → {H_peri_normal[-1]:.3f} (Δ={H_peri_normal[-1]-H_peri_normal[0]:+.3f})")
    
    print(f"\n[Blocked Flow System]")
    print(f"  M:       {M_blocked[0]:.3f} → {M_blocked[-1]:.3f} (Δ={M_blocked[-1]-M_blocked[0]:+.3f})")
    print(f"  H_core:  {H_core_blocked[0]:.3f} → {H_core_blocked[-1]:.3f} (Δ={H_core_blocked[-1]-H_core_blocked[0]:+.3f})")
    print(f"  H_peri:  {H_peri_blocked[0]:.3f} → {H_peri_blocked[-1]:.3f} (Δ={H_peri_blocked[-1]-H_peri_blocked[0]:+.3f})")
    
    # 绘图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    t = np.arange(n_steps)
    
    # M 对比
    ax1 = axes[0, 0]
    ax1.plot(t, M_normal, 'b-', linewidth=2, label='Normal')
    ax1.plot(t, M_blocked, 'r--', linewidth=2, label='Blocked')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('M')
    ax1.set_title('M: Normal vs Blocked')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # H_core 对比
    ax2 = axes[0, 1]
    ax2.plot(t, H_core_normal, 'b-', linewidth=2, label='Normal')
    ax2.plot(t, H_core_blocked, 'r--', linewidth=2, label='Blocked')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('H_core')
    ax2.set_title('H_core: Normal vs Blocked')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # H_peri 对比
    ax3 = axes[0, 2]
    ax3.plot(t, H_peri_normal, 'b-', linewidth=2, label='Normal')
    ax3.plot(t, H_peri_blocked, 'r--', linewidth=2, label='Blocked')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('H_periphery')
    ax3.set_title('H_peri: Normal vs Blocked')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # M vs H_core 散点
    ax4 = axes[1, 0]
    ax4.scatter(M_normal, H_core_normal, c='blue', alpha=0.5, label='Normal')
    ax4.scatter(M_blocked, H_core_blocked, c='red', alpha=0.5, label='Blocked')
    ax4.set_xlabel('M')
    ax4.set_ylabel('H_core')
    ax4.set_title('M vs H_core')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 变化量对比
    ax5 = axes[1, 1]
    labels = ['M', 'H_core', 'H_peri']
    normal_deltas = [M_normal[-1]-M_normal[0], H_core_normal[-1]-H_core_normal[0], H_peri_normal[-1]-H_peri_normal[0]]
    blocked_deltas = [M_blocked[-1]-M_blocked[0], H_core_blocked[-1]-H_core_blocked[0], H_peri_blocked[-1]-H_peri_blocked[0]]
    
    x = np.arange(len(labels))
    width = 0.35
    ax5.bar(x - width/2, normal_deltas, width, label='Normal', color='blue', alpha=0.7)
    ax5.bar(x + width/2, blocked_deltas, width, label='Blocked', color='red', alpha=0.7)
    ax5.axhline(0, color='black', linewidth=0.5)
    ax5.set_xticks(x)
    ax5.set_xticklabels(labels)
    ax5.set_ylabel('Delta (t=end - t=0)')
    ax5.set_title('Change Comparison')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 相关性对比
    ax6 = axes[1, 2]
    corr_normal = np.corrcoef(M_normal, H_core_normal)[0, 1]
    corr_blocked = np.corrcoef(M_blocked, H_core_blocked)[0, 1]
    ax6.bar(['Normal', 'Blocked'], [corr_normal, corr_blocked], color=['blue', 'red'], alpha=0.7)
    ax6.set_ylabel('r(M, H_core)')
    ax6.set_title('Correlation: M vs H_core')
    ax6.set_ylim(0, 1)
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('F:/hypergraph_bistability/figures/flow_block.png', dpi=150)
    print("\n[OK] Saved: figures/flow_block.png")
    
    # 结论
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    
    delta_M_norm = M_normal[-1] - M_normal[0]
    delta_M_block = M_blocked[-1] - M_blocked[0]
    delta_H_core_norm = H_core_normal[-1] - H_core_normal[0]
    delta_H_core_block = H_core_blocked[-1] - H_core_blocked[0]
    
    print(f"\nDelta M:")
    print(f"  Normal: {delta_M_norm:+.3f}")
    print(f"  Blocked: {delta_M_block:+.3f}")
    print(f"  Reduction: {(1 - delta_M_block/delta_M_norm)*100:.1f}%")
    
    print(f"\nDelta H_core:")
    print(f"  Normal: {delta_H_core_norm:+.3f}")
    print(f"  Blocked: {delta_H_core_block:+.3f}")
    print(f"  Reduction: {(1 - delta_H_core_block/delta_H_core_norm)*100:.1f}%")
    
    if delta_M_block < delta_M_norm * 0.5 and delta_H_core_block < delta_H_core_norm * 0.5:
        print("\n✅ CAUSAL CONFIRMED!")
        print("   Blocking flow reduces BOTH M and H_core")
    elif delta_H_core_block < delta_H_core_norm * 0.5:
        print("\n✅ H_core reduced even with M maintained")
        print("   Complexity flow is causal!")
    else:
        print("\n⚠️ Need stronger intervention")


if __name__ == '__main__':
    run_block_experiment(n_runs=8, n_steps=80)
