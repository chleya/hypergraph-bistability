"""
Identity Tracking: 核心复杂性是"继承"还是"新生"?
=================================================

目标：追踪每个 hyperedge 的"出身"
- new_ratio: 核心中新生元素的比例
- inherited_ratio: 核心中继承自外围的元素比例
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
    
    def get_clusters_with_id(self):
        """返回 (clusters, max_cluster_ids, all_edge_ids_in_core)"""
        if not self.V:
            return [], set(), set()
        
        adj = {v: set() for v in self.V}
        for eid, e in self.E.items():
            nodes = list(e)
            for i in range(len(nodes)):
                for j in range(i+1, len(nodes)):
                    adj[nodes[i]].add(nodes[j])
                    adj[nodes[j]].add(nodes[i])
        
        visited = set()
        clusters = []
        cluster_edge_ids = []
        
        for s in self.V:
            if s in visited:
                continue
            stack = [s]
            c = set()
            edge_ids_in_cluster = set()
            while stack:
                n = stack.pop()
                if n in visited:
                    continue
                visited.add(n)
                c.add(n)
                # 找出连接到这个节点的所有边
                for eid, e in self.E.items():
                    if n in e:
                        edge_ids_in_cluster.add(eid)
            
            clusters.append(c)
            cluster_edge_ids.append(edge_ids_in_cluster)
        
        if clusters:
            max_cluster_idx = max(range(len(clusters)), key=lambda i: len(clusters[i]))
            max_cluster = clusters[max_cluster_idx]
            max_cluster_edge_ids = cluster_edge_ids[max_cluster_idx]
        else:
            max_cluster = set()
            max_cluster_edge_ids = set()
        
        return clusters, max_cluster, max_cluster_edge_ids
    
    def apply_rules(self, steps=1):
        for _ in range(steps):
            # Growth - 新 hyperedge
            if random.random() < 0.3 and self.E:
                # 选择一个现有 hyperedge 作为种子
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
            
            # Fusion - 合并两个超边
            if random.random() < 0.25 and len(self.E) >= 2:
                eid1, e1 = random.choice(list(self.E.items()))
                eid2, e2 = random.choice(list(self.E.items()))
                
                if eid1 != eid2 and len(e1 & e2) >= 1 and random.random() < 0.5:
                    # 合并产生新 hyperedge
                    new_e = frozenset(e1 | e2)
                    if len(new_e) >= 2:
                        # 新边有新 ID（新生）
                        self.E[self.next_id] = new_e
                        self.next_id += 1
                        # 删除旧边
                        if eid1 in self.E:
                            del self.E[eid1]
                        if eid2 in self.E:
                            del self.E[eid2]
            
            # Split - 分裂
            if random.random() < 0.12:
                large = [(eid, e) for eid, e in self.E.items() if len(e) > 2]
                if large:
                    eid, e = random.choice(large)
                    nodes = list(e)
                    if len(nodes) >= 4:
                        split = len(nodes) // 2
                        e_a = frozenset(nodes[:split])
                        e_b = frozenset(nodes[split:])
                        # 两条新边
                        self.E[self.next_id] = e_a
                        self.next_id += 1
                        self.E[self.next_id] = e_b
                        self.next_id += 1
                        # 删除旧边
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
                                # 删除旧边，添加新边
                                del self.E[eid]
                                self.E[self.next_id] = new_e
                                self.next_id += 1
    
    def analyze_core_at(self, time_step):
        """分析某时刻的核心组成"""
        clusters, max_cluster, max_cluster_edge_ids = self.get_clusters_with_id()
        
        if not max_cluster:
            return {
                'core_size': 0,
                'total_edges': len(self.E),
                'new_ratio': 0,
                'early_ratio': 0,
            }
        
        total_edges = len(self.E)
        core_edge_count = len(max_cluster_edge_ids)
        
        # 早期边界：t < 20 为"早期"
        early_threshold = 20
        early_edges_in_core = sum(1 for eid in max_cluster_edge_ids if eid < early_threshold)
        
        new_ratio = 1.0 - (early_edges_in_core / max(core_edge_count, 1))
        early_ratio = early_edges_in_core / max(core_edge_count, 1)
        
        return {
            'core_size': len(max_cluster) / len(self.V),
            'total_edges': total_edges,
            'core_edges': core_edge_count,
            'new_ratio': new_ratio,
            'early_ratio': early_ratio,
            'early_edges_in_core': early_edges_in_core
        }


def run_identity_tracking(n_runs=10, n_steps=100):
    """运行 identity tracking"""
    
    print("=" * 60)
    print("Identity Tracking: New vs Inherited")
    print("=" * 60)
    
    results = []
    
    for run in range(n_runs):
        h = HypergraphIdentity(N=50, p_pair=0.5, seed=run*100+42)
        
        # 记录每个时间点的核心组成
        history = []
        
        for t in range(n_steps):
            h.apply_rules()
            
            if t % 10 == 0 or t == n_steps - 1:
                analysis = h.analyze_core_at(t)
                history.append({
                    't': t,
                    'core_size': analysis['core_size'],
                    'total_edges': analysis['total_edges'],
                    'core_edges': analysis['core_edges'],
                    'new_ratio': analysis['new_ratio'],
                    'early_ratio': analysis['early_ratio']
                })
        
        results.append(history)
        print(f"Run {run+1}/{n_runs} done")
    
    # 分析
    print("\n[Analysis]")
    
    # 平均
    times = [r['t'] for r in results[0]]
    core_size_avg = np.mean([[h['core_size'] for h in r] for r in results], axis=0)
    new_ratio_avg = np.mean([[h['new_ratio'] for h in r] for r in results], axis=0)
    early_ratio_avg = np.mean([[h['early_ratio'] for h in r] for r in results], axis=0)
    
    # 最终状态
    final_new_ratio = [r[-1]['new_ratio'] for r in results]
    final_early_ratio = [r[-1]['early_ratio'] for r in results]
    final_core_size = [r[-1]['core_size'] for r in results]
    
    print(f"\n[Final State Statistics]")
    print(f"  Core size: {np.mean(final_core_size):.3f} ± {np.std(final_core_size):.3f}")
    print(f"  New ratio: {np.mean(final_new_ratio):.3f} ± {np.std(final_new_ratio):.3f}")
    print(f"  Early ratio: {np.mean(final_early_ratio):.3f} ± {np.std(final_early_ratio):.3f}")
    
    # 绘图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Core size over time
    ax1 = axes[0, 0]
    ax1.plot(times, core_size_avg, 'b-', linewidth=2)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Core Size (M)')
    ax1.set_title('Core Growth')
    ax1.grid(True, alpha=0.3)
    
    # 2. New ratio over time
    ax2 = axes[0, 1]
    ax2.plot(times, new_ratio_avg, 'g-', linewidth=2, label='New ratio')
    ax2.plot(times, early_ratio_avg, 'r-', linewidth=2, label='Early ratio')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Ratio')
    ax2.set_title('Core Composition: New vs Early')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Final distribution
    ax3 = axes[1, 0]
    ax3.hist(final_new_ratio, bins=20, alpha=0.7, color='green', label='New ratio')
    ax3.axvline(np.mean(final_new_ratio), color='green', linestyle='--', linewidth=2)
    ax3.set_xlabel('New Ratio')
    ax3.set_ylabel('Count')
    ax3.set_title('Final New Ratio Distribution')
    ax3.grid(True, alpha=0.3)
    
    # 4. Scatter: M vs new_ratio
    ax4 = axes[1, 1]
    ax4.scatter(final_core_size, final_new_ratio, c='blue', alpha=0.6)
    ax4.set_xlabel('Core Size (M)')
    ax4.set_ylabel('New Ratio')
    ax4.set_title('M vs New Ratio')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('F:/hypergraph_bistability/figures/identity_tracking.png', dpi=150)
    print("\n[OK] Saved: figures/identity_tracking.png")
    
    # 结论
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    
    avg_new = np.mean(final_new_ratio)
    avg_early = np.mean(final_early_ratio)
    
    print(f"\n  New ratio: {avg_new:.1%}")
    print(f"  Early ratio: {avg_early:.1%}")
    
    if avg_new > 0.7:
        print("\n✅ COMPLEXITY RECONSTRUCTION CONFIRMED!")
        print(f"   {avg_new:.0%} of core is NEWLY GENERATED")
        print("   → Complexity is NOT transported, but REBUILT in place!")
    elif avg_early > 0.5:
        print("\n⚠️ Complexity might be inherited")
        print(f"   {avg_early:.0%} of core comes from early structures")
    else:
        print("\n⚠️ Mixed - need more analysis")


if __name__ == '__main__':
    run_identity_tracking(n_runs=10, n_steps=100)
