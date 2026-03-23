"""
Diagnostic: Understand system structure and fusion behavior
"""
import numpy as np
import random
from scipy import stats

class DiagnosticSystem:
    def __init__(self, N=50, p_pair=0.5, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        
        self.N = N
        self.p_pair = p_pair
        self.K = int(0.35 * N)
        self.V = list(range(N))
        self.E = []
        
        self.total_fusions = 0
        self.peri_to_core_detected = 0
        self.fusion_attempts = 0

        for _ in range(15):
            if random.random() < p_pair:
                e = frozenset(random.sample(self.V, 2))
            else:
                e = frozenset(random.sample(self.V, min(3, N)))
            self.E.append(e)
    
    def get_node_degree(self, v):
        return sum(1 for e in self.E if v in e)
    
    def get_clusters_and_edges(self):
        if not self.V or not self.E:
            return [], set(), set(), set()
        
        adj = {v: set() for v in self.V}
        for idx, e in enumerate(self.E):
            nodes = list(e)
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
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
            return [], set(), set(), set()
        
        max_cluster = max(clusters, key=len)
        
        edge_ids_in_max = set()
        edge_ids_in_peri = set()
        for idx, e in enumerate(self.E):
            if e & max_cluster:
                edge_ids_in_max.add(idx)
            else:
                edge_ids_in_peri.add(idx)
        
        return clusters, max_cluster, edge_ids_in_max, edge_ids_in_peri
    
    def compute_H(self, node_set):
        relevant_edges = [e for e in self.E if len(e & node_set) > 0]
        if not relevant_edges:
            return 0.0
        sizes = [len(e) for e in relevant_edges]
        counts = {}
        for s in sizes:
            counts[s] = counts.get(s, 0) + 1
        total = len(relevant_edges)
        H = 0.0
        for c in counts.values():
            p = c / total
            if p > 0:
                H -= p * np.log2(p)
        return H
    
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
                    self.fusion_attempts += 1
                    self.total_fusions += 1
                    
                    _, max_cluster, edge_ids_max, edge_ids_peri = self.get_clusters_and_edges()
                    
                    nodes1 = set(e1)
                    nodes2 = set(e2)
                    
                    is_peri_to_core = False
                    if max_cluster:
                        e1_in_core = bool(nodes1 & max_cluster)
                        e2_in_core = bool(nodes2 & max_cluster)
                        e1_in_peri = bool(nodes1 - max_cluster)
                        e2_in_peri = bool(nodes2 - max_cluster)
                        
                        if (e1_in_core and e2_in_peri) or (e1_in_peri and e2_in_core):
                            is_peri_to_core = True
                            self.peri_to_core_detected += 1
                    
                    if True:
                        new_e = frozenset(e1 | e2)
                        if len(new_e) >= 2:
                            self.E.append(new_e)
                            if e1 in self.E:
                                self.E.remove(e1)
                            if e2 in self.E:
                                self.E.remove(e2)
            
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


def run_diagnostic(n_runs=20, T=80):
    print("=" * 70)
    print("System Structure Diagnostic")
    print("=" * 70)
    
    fusion_attempts_list = []
    total_fusions_list = []
    peri_to_core_list = []
    final_M_list = []
    final_H_core_list = []
    
    for i in range(n_runs):
        h = DiagnosticSystem(N=50, p_pair=0.5, seed=i * 100 + 42)
        
        for t in range(T):
            h.apply_rules()
        
        fusion_attempts_list.append(h.fusion_attempts)
        total_fusions_list.append(h.total_fusions)
        peri_to_core_list.append(h.peri_to_core_detected)
        
        _, max_cluster, _, _ = h.get_clusters_and_edges()
        M = len(max_cluster) / len(h.V) if max_cluster else 0.0
        H_core = h.compute_H(max_cluster) if max_cluster else 0.0
        
        final_M_list.append(M)
        final_H_core_list.append(H_core)
        
        print(f"Run {i+1}: fusion_attempts={h.fusion_attempts}, total_fusions={h.total_fusions}, "
              f"peri_to_core={h.peri_to_core_detected}, M={M:.3f}, H_core={H_core:.3f}")
    
    print("\n" + "=" * 70)
    print("SUMMARY (across {} runs)".format(n_runs))
    print("=" * 70)
    print(f"  Fusion attempts: {np.mean(fusion_attempts_list):.1f} ± {np.std(fusion_attempts_list):.1f}")
    print(f"  Total fusions:  {np.mean(total_fusions_list):.1f} ± {np.std(total_fusions_list):.1f}")
    print(f"  Peri->core:     {np.mean(peri_to_core_list):.1f} ± {np.std(peri_to_core_list):.1f}")
    print(f"  Final M:        {np.mean(final_M_list):.3f} ± {np.std(final_M_list):.3f}")
    print(f"  Final H_core:   {np.mean(final_H_core_list):.3f} ± {np.std(final_H_core_list):.3f}")
    
    if np.mean(peri_to_core_list) < 0.1:
        print("\n" + "=" * 70)
        print("CRITICAL FINDING:")
        print("  peri->core fusion is essentially ZERO in this system")
        print("  ")
        print("  IMPLICATIONS:")
        print("  1. Transport via fusion is NOT a significant mechanism")
        print("  2. Growth + Split must be the source of core complexity")
        print("  3. In-situ reconstruction is confirmed by default")
        print("=" * 70)
    
    return {
        'fusion_attempts': fusion_attempts_list,
        'total_fusions': total_fusions_list,
        'peri_to_core': peri_to_core_list,
        'final_M': final_M_list,
        'final_H_core': final_H_core_list
    }


if __name__ == '__main__':
    results = run_diagnostic(n_runs=20, T=80)
