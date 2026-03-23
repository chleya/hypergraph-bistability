"""
Identity Tracking v2: 修复版
============================
"""

import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

os.makedirs('F:/hypergraph_bistability/figures', exist_ok=True)


class HypergraphIdentity:
    """带身份追踪的超图"""
    def __init__(self, N=50, p_pair=0.5, seed=42):
        random.seed(seed)
        
        self.N = N
        self.p_pair = p_pair
        self.K = int(0.35 * N)
        self.V = list(range(N))
        self.E = {}  # id -> frozenset
        self.next_id = 0
        
        # 初始超边
        for _ in range(15):
            if random.random() < p_pair:
                e = frozenset(random.sample(self.V, 2))
            else:
                e = frozenset(random.sample(self.V, min(3, N)))
            self.E[self.next_id] = e
            self.next_id += 1
    
    def get_node_degree(self, v):
        return sum(1 for e in self.E.values() if v in e)
    
    def get_clusters_and_edges(self):
        """返回 (clusters, max_cluster, edge_ids_in_max_cluster)"""
        if not self.V or not self.E:
            return [], set(), set()
        
        # 构建邻接表
        adj = {v: set() for v in self.V}
        for eid, e in self.E.items():
            nodes = list(e)
            for i in range(len(nodes)):
                for j in range(i+1, len(nodes)):
                    if nodes[i] in adj and nodes[j] in adj:
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
                if n in adj:
                    for nb in adj[n]:
                        if nb not in visited:
                            stack.append(nb)
            clusters.append(c)
        
        if not clusters:
            return [], set(), set()
        
        # 找最大 cluster
        max_cluster = max(clusters, key=len)
        
        # 找最大 cluster 中的边
        edge_ids_in_max = set()
        for eid, e in self.E.items():
            if e & max_cluster:  # 有交集就算
                edge_ids_in_max.add(eid)
        
        return clusters, max_cluster, edge_ids_in_max
    
    def apply_rules(self, steps=1):
        for _ in range(steps):
            # Growth
            if random.random() < 0.3 and self.E:
                eid, e = random.choice(list(self.E.items()))
                v = random.choice(list(e))
                w = max(self.V) + 1
                self.V.append(w)
                
                if random.random() < self.p_pair:
                    new_e = frozenset([v, w])
                else:
                    new_e = frozenset([v, w, random.choice(self.V[:-1])])
                self.E[self.next_id] = new_e
                self.next_id += 1
            
            # Fusion
            if random.random() < 0.25 and len(self.E) >= 2:
                eids = list(self.E.keys())
                eid1, eid2 = random.sample(eids, 2)
                e1, e2 = self.E[eid1], self.E[eid2]
                
                if len(e1 & e2) >= 1 and random.random() < 0.5:
                    new_e = frozenset(e1 | e2)
                    if len(new_e) >= 2:
                        self.E[self.next_id] = new_e
                        self.next_id += 1
                        if eid1 in self.E:
                            del self.E[eid1]
                        if eid2 in self.E:
                            del self.E[eid2]
            
            # Split
            if random.random() < 0.12:
                large = [(eid, e) for eid, e in self.E.items() if len(e) > 2]
                if large:
                    eid, e = random.choice(large)
                    nodes = list(e)
                    if len(nodes) >= 4:
                        split = len(nodes) // 2
                        e_a = frozenset(nodes[:split])
                        e_b = frozenset(nodes[split:])
                        self.E[self.next_id] = e_a
                        self.next_id += 1
                        self.E[self.next_id] = e_b
                        self.next_id += 1
                        if eid in self.E:
                            del self.E[eid]
            
            # K cap
            for v in list(self.V):
                d = self.get_node_degree(v)
                if d > self.K:
                    excess = d - self.K
                    v_edges = [(eid, e) for eid, e in self.E.items() if v in e]
                    for eid, e in v_edges[:excess]:
                        if len(e) > 2:
                            new_e = e - {v}
                            if len(new_e) >= 2:
                                del self.E[eid]
                                self.E[self.next_id] = new_e
                                self.next_id += 1
    
    def get_M(self):
        if not self.V:
            return 0
        
        _, max_cluster, _ = self.get_clusters_and_edges()
        return len(max_cluster) / len(self.V) if max_cluster else 0
    
    def analyze_core_composition(self):
        """分析核心组成"""
        clusters, max_cluster, edge_ids_in_max = self.get_clusters_and_edges()
        
        if not max_cluster:
            return {
                'M': 0,
                'core_edges': 0,
                'new_ratio': 0,
                'early_ratio': 0
            }
        
        total_edges = len(self.E)
        core_edge_count = len(edge_ids_in_max)
        
        # 早期边界：t < 25 为"早期"（假设前25%是早期）
        early_threshold = int(self.next_id * 0.25)
        early_edges_in_core = sum(1 for eid in edge_ids_in_max if eid < early_threshold)
        
        new_ratio = 1.0 - (early_edges_in_core / max(core_edge_count, 1))
        early_ratio = early_edges_in_core / max(core_edge_count, 1)
        
        return {
            'M': len(max_cluster) / len(self.V),
            'core_edges': core_edge_count,
            'total_edges': total_edges,
            'new_ratio': new_ratio,
            'early_ratio': early_ratio,
            'early_threshold': early_threshold
        }


def run_tracking(n_runs=10, n_steps=80):
    print("=" * 60)
    print("Identity Tracking v2")
    print("=" * 60)
    
    results = []
    
    for run in range(n_runs):
        h = HypergraphIdentity(N=50, p_pair=0.5, seed=run*100+42)
        
        history = []
        
        for t in range(n_steps):
            h.apply_rules()
            
            if t % 10 == 9 or t == n_steps - 1:
                analysis = h.analyze_core_composition()
                history.append({
                    't': t,
                    'M': analysis['M'],
                    'new_ratio': analysis['new_ratio'],
                    'early_ratio': analysis['early_ratio']
                })
        
        results.append(history)
        print(f"Run {run+1}/{n_runs}")
    
    # 分析最终状态
    final_M = [r[-1]['M'] for r in results]
    final_new = [r[-1]['new_ratio'] for r in results]
    final_early = [r[-1]['early_ratio'] for r in results]
    
    print(f"\n[Final State]")
    print(f"  M: {np.mean(final_M):.3f} ± {np.std(final_M):.3f}")
    print(f"  New ratio: {np.mean(final_new):.1%} ± {np.std(final_new):.1%}")
    print(f"  Early ratio: {np.mean(final_early):.1%} ± {np.std(final_early):.1%}")
    
    # 绘图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    times = [r['t'] for r in results[0]]
    M_avg = np.mean([[h['M'] for h in r] for r in results], axis=0)
    new_avg = np.mean([[h['new_ratio'] for h in r] for r in results], axis=0)
    early_avg = np.mean([[h['early_ratio'] for h in r] for r in results], axis=0)
    
    ax1 = axes[0]
    ax1.plot(times, M_avg, 'b-', linewidth=2)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('M')
    ax1.set_title('Order Parameter M')
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    ax2.plot(times, new_avg, 'g-', linewidth=2, label='New')
    ax2.plot(times, early_avg, 'r-', linewidth=2, label='Early')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Ratio')
    ax2.set_title('Core Composition')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[2]
    ax3.hist(final_new, bins=15, alpha=0.7, color='green')
    ax3.axvline(np.mean(final_new), color='green', linestyle='--', linewidth=2)
    ax3.set_xlabel('New Ratio')
    ax3.set_ylabel('Count')
    ax3.set_title('Final New Ratio')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('F:/hypergraph_bistability/figures/identity_v2.png', dpi=150)
    print("\n[OK] Saved: figures/identity_v2.png")
    
    # 结论
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    
    avg_new = np.mean(final_new)
    
    if avg_new > 0.7:
        print(f"\n✅ Complexity Reconstruction: {avg_new:.0%} NEW")
    elif avg_new > 0.5:
        print(f"\n🟡 Mixed: {avg_new:.0%} new, {1-avg_new:.0%} inherited")
    else:
        print(f"\n⚠️ Mostly inherited: {avg_new:.0%} new")


if __name__ == '__main__':
    run_tracking(n_runs=10, n_steps=80)
