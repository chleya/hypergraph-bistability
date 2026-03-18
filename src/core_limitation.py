"""
Core 限制实验：因果验证
======================

目标：限制 core growth，验证 H_core 是否还增长
- 如果 M 受限后 H_core 也停止 → 因果成立
- 如果 M 受限后 H_core 仍增长 → 相关性可能是伪的
"""

import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

os.makedirs('F:/hypergraph_bistability/figures', exist_ok=True)


class HypergraphCapped:
    """带 core size cap 的超图"""
    def __init__(self, N=50, p_pair=0.5, seed=42, max_core_frac=0.3):
        random.seed(seed)
        
        self.N = N
        self.p_pair = p_pair
        self.K = int(0.35 * N)
        self.max_core_frac = max_core_frac  # 限制 core 最大比例
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
        
        # 限制 core size
        max_size = int(self.max_core_frac * len(self.V))
        if len(max_cluster) > max_size:
            # 截断
            max_cluster = set(list(max_cluster)[:max_size])
        
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
                        # 检查是否超过 core cap
                        clusters, max_cluster = self.get_clusters()
                        if len(max_cluster) + len(new_e) <= self.max_core_frac * len(self.V):
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
    """正常超图（无限制）"""
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


def run_comparison(n_runs=8, n_steps=80):
    """对比：正常 vs Core 限制"""
    
    print("=" * 60)
    print("Core Limitation Experiment")
    print("=" * 60)
    
    # 正常系统
    print("\n[Running] Normal system")
    normal_results = []
    for i in range(n_runs):
        h = HypergraphNormal(N=50, p_pair=0.5, seed=i*100+42)
        M_hist, H_core_hist = [], []
        for t in range(n_steps):
            h.apply_rules()
            clusters, max_cluster = h.get_clusters()
            M = h.get_M()
            if max_cluster:
                H_core = h.compute_H(max_cluster)
            else:
                H_core = 0
            M_hist.append(M)
            H_core_hist.append(H_core)
        normal_results.append({'M': M_hist, 'H_core': H_core_hist})
        print(f"  {i+1}/{n_runs}")
    
    # Core 限制系统
    print("\n[Running] Capped system (max_core=0.3)")
    capped_results = []
    for i in range(n_runs):
        h = HypergraphCapped(N=50, p_pair=0.5, seed=i*100+42, max_core_frac=0.3)
        M_hist, H_core_hist = [], []
        for t in range(n_steps):
            h.apply_rules()
            clusters, max_cluster = h.get_clusters()
            M = h.get_M()
            if max_cluster:
                H_core = h.compute_H(max_cluster)
            else:
                H_core = 0
            M_hist.append(M)
            H_core_hist.append(H_core)
        capped_results.append({'M': M_hist, 'H_core': H_core_hist})
        print(f"  {i+1}/{n_runs}")
    
    # 分析
    print("\n[Analysis]")
    
    # 正常
    M_normal = np.mean([r['M'] for r in normal_results], axis=0)
    H_normal = np.mean([r['H_core'] for r in normal_results], axis=0)
    
    # 限制
    M_capped = np.mean([r['M'] for r in capped_results], axis=0)
    H_capped = np.mean([r['H_core'] for r in capped_results], axis=0)
    
    print(f"\n[Normal System]")
    print(f"  M: {M_normal[0]:.3f} → {M_normal[-1]:.3f} (Δ={M_normal[-1]-M_normal[0]:+.3f})")
    print(f"  H_core: {H_normal[0]:.3f} → {H_normal[-1]:.3f} (Δ={H_normal[-1]-H_normal[0]:+.3f})")
    
    print(f"\n[Capped System (max=0.3)]")
    print(f"  M: {M_capped[0]:.3f} → {M_capped[-1]:.3f} (Δ={M_capped[-1]-M_capped[0]:+.3f})")
    print(f"  H_core: {H_capped[0]:.3f} → {H_capped[-1]:.3f} (Δ={H_capped[-1]-H_capped[0]:+.3f})")
    
    # 绘图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    t = np.arange(n_steps)
    
    # M 对比
    ax1 = axes[0]
    ax1.plot(t, M_normal, 'b-', linewidth=2, label='Normal')
    ax1.plot(t, M_capped, 'r--', linewidth=2, label='Capped')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('M')
    ax1.set_title('M: Normal vs Capped')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0.3, color='gray', linestyle=':', alpha=0.5)
    
    # H_core 对比
    ax2 = axes[1]
    ax2.plot(t, H_normal, 'b-', linewidth=2, label='Normal')
    ax2.plot(t, H_capped, 'r--', linewidth=2, label='Capped')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('H_core')
    ax2.set_title('H_core: Normal vs Capped')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 散点图：M vs H_core
    ax3 = axes[2]
    ax3.scatter(M_normal, H_normal, c='blue', alpha=0.5, label='Normal')
    ax3.scatter(M_capped, H_capped, c='red', alpha=0.5, label='Capped')
    ax3.set_xlabel('M')
    ax3.set_ylabel('H_core')
    ax3.set_title('M vs H_core')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('F:/hypergraph_bistability/figures/core_limitation.png', dpi=150)
    print("\n[OK] Saved: figures/core_limitation.png")
    
    # 结论
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    
    if M_capped[-1] < M_normal[-1] * 0.8:
        print("\n✅ Core cap effective: M is significantly reduced")
    else:
        print("\n⚠️ Core cap not very effective")
    
    if H_capped[-1] < H_normal[-1] * 0.8:
        print("✅ H_core also reduced when M is capped")
        print("   → CAUSAL relationship confirmed!")
    else:
        print("⚠️ H_core still grows even with M capped")
        print("   → Correlation might be spurious")


if __name__ == '__main__':
    run_comparison(n_runs=8, n_steps=80)
