"""
Flow Blocking with Diagnostics: 验证干预是否生效
================================================
"""

import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

os.makedirs('F:/hypergraph_bistability/figures', exist_ok=True)


class HypergraphBlockFlowDiag:
    """带诊断的阻断流动超图"""
    def __init__(self, N=50, p_pair=0.5, seed=42, block_prob=0.95):
        random.seed(seed)
        
        self.N = N
        self.p_pair = p_pair
        self.K = int(0.35 * N)
        self.block_prob = block_prob
        self.V = list(range(N))
        self.E = []
        
        # 诊断统计
        self.total_fusions = 0
        self.peri_to_core_attempts = 0
        self.blocks = 0
        
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
            
            # Fusion (with flow blocking + diagnostics!)
            if random.random() < 0.25 and len(self.E) >= 2:
                e1, e2 = random.sample(self.E, 2)
                if len(e1 & e2) >= 1 and random.random() < 0.5:
                    self.total_fusions += 1
                    
                    clusters, max_cluster = self.get_clusters()
                    
                    nodes1 = set(e1)
                    nodes2 = set(e2)
                    
                    # 检测是否是 peri → core
                    is_peri_to_core = False
                    if max_cluster:
                        e1_in_core = bool(nodes1 & max_cluster)
                        e2_in_core = bool(nodes2 & max_cluster)
                        e1_in_peri = bool(nodes1 - max_cluster)
                        e2_in_peri = bool(nodes2 - max_cluster)
                        
                        # 如果一个是 core，一个是 peri
                        if (e1_in_core and e2_in_peri) or (e1_in_peri and e2_in_core):
                            is_peri_to_core = True
                            self.peri_to_core_attempts += 1
                    
                    if is_peri_to_core:
                        if random.random() < self.block_prob:
                            self.blocks += 1
                        else:
                            new_e = frozenset(e1 | e2)
                            if len(new_e) >= 2:
                                self.E.append(new_e)
                                if e1 in self.E: self.E.remove(e1)
                                if e2 in self.E: self.E.remove(e2)
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


def run_diagnostic(n_runs=5, n_steps=80):
    """运行诊断"""
    
    print("=" * 60)
    print("Flow Blocking Diagnostic")
    print("=" * 60)
    
    total_fusions = 0
    total_peri_to_core = 0
    total_blocks = 0
    
    for i in range(n_runs):
        h = HypergraphBlockFlowDiag(N=50, p_pair=0.5, seed=i*100+42, block_prob=0.95)
        for t in range(n_steps):
            h.apply_rules()
        
        total_fusions += h.total_fusions
        total_peri_to_core += h.peri_to_core_attempts
        total_blocks += h.blocks
        
        print(f"Run {i+1}: fusions={h.total_fusions}, peri_to_core={h.peri_to_core_attempts}, blocks={h.blocks}")
    
    print(f"\n[Total Statistics]")
    print(f"  Total fusions: {total_fusions}")
    print(f"  Total peri->core attempts: {total_peri_to_core}")
    print(f"  Total blocks: {total_blocks}")
    print(f"  Block rate: {total_blocks/max(total_peri_to_core,1)*100:.1f}%")
    print(f"  Peri->core rate: {total_peri_to_core/max(total_fusions,1)*100:.1f}%")
    
    # 结论
    print("\n" + "=" * 60)
    print("DIAGNOSIS")
    print("=" * 60)
    
    if total_peri_to_core == 0:
        print("\n🔴 IMPLEMENTATION BUG: peri->core never detected!")
        print("   → The condition is never triggered")
    elif total_blocks == 0:
        print("\n🔴 IMPLEMENTATION BUG: blocks never executed!")
        print("   → block_prob might not be working")
    else:
        print("\n✅ Implementation is working")
        print(f"   → {total_peri_to_core} peri->core events detected")
        print(f"   → {total_blocks} were blocked")


if __name__ == '__main__':
    run_diagnostic(n_runs=5, n_steps=80)
