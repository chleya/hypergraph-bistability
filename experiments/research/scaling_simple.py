"""
Scaling Law 实验（简化版）
==========================
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import os
import json

os.makedirs('F:/hypergraph_bistability/figures', exist_ok=True)


class HypergraphSimple:
    """简化版超图"""
    
    def __init__(self, N=40, p_pair=0.5, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        
        self.N = N
        self.p_pair = p_pair
        self.K = 14  # 固定容量约束
        
        self.V = list(range(N))
        self.E = []
        
        # 初始化
        for _ in range(10):
            if random.random() < p_pair:
                e = frozenset(random.sample(self.V, 2))
            else:
                e = frozenset(random.sample(self.V, min(3, N)))
            self.E.append(e)
    
    def get_node_degree(self, v):
        return sum(1 for e in self.E if v in e)
    
    def apply_rules(self, steps=1):
        for _ in range(steps):
            # 生长
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
            
            # 融合
            if random.random() < 0.25 and len(self.E) >= 2:
                e1, e2 = random.sample(self.E, 2)
                if len(e1 & e2) >= 1 and random.random() < 0.5:
                    new_e = frozenset(e1 | e2)
                    if len(new_e) >= 2:
                        self.E.append(new_e)
                        if e1 in self.E: self.E.remove(e1)
                        if e2 in self.E: self.E.remove(e2)
            
            # 分裂
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
            
            # 容量约束
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


def main():
    print("=" * 50)
    print("Scaling Law (简化版)")
    print("=" * 50)
    
    # k 值
    k_values = [2.0, 2.5, 3.0]
    p_values = [1.0, 0.5, 0.0]
    
    M_results = {}
    rate_results = {}
    
    # 测量 M* 和 rates
    for p, k in zip(p_values, k_values):
        print(f"\nk={k}:")
        
        # M*
        Ms = []
        for i in range(8):
            h = HypergraphSimple(N=40, p_pair=p, seed=i*10+42)
            for _ in range(100):
                h.apply_rules()
            Ms.append(h.get_M())
        
        M_results[k] = {'mean': np.mean(Ms), 'std': np.std(Ms)}
        print(f"  M* = {np.mean(Ms):.3f} +/- {np.std(Ms):.3f}")
        
        # HIGH → LOW
        HL_times = []
        for i in range(8):
            h = HypergraphSimple(N=40, p_pair=p, seed=i*10+42)
            h.E = [frozenset(range(40))]
            
            for t in range(200):
                h.apply_rules()
                if h.get_M() < 0.5:
                    HL_times.append(t)
                    break
            else:
                HL_times.append(201)
        
        # LOW → HIGH
        LH_times = []
        for i in range(8):
            h = HypergraphSimple(N=40, p_pair=p, seed=i*10+42)
            h.E = [frozenset([j, j+1]) for j in range(0, 38, 2)]
            
            for t in range(200):
                h.apply_rules()
                if h.get_M() > 0.7:
                    LH_times.append(t)
                    break
            else:
                LH_times.append(201)
        
        tau_HL = np.mean([t for t in HL_times if t <= 200])
        tau_LH = np.mean([t for t in LH_times if t <= 200])
        ratio = tau_LH / tau_HL if tau_HL > 0 and tau_LH <= 200 else 0
        
        rate_results[k] = {
            'tau_HL': tau_HL,
            'tau_LH': tau_LH,
            'ratio': ratio,
            'dV': np.log(ratio) if ratio > 0 else 0
        }
        
        print(f"  τ(H→L) = {tau_HL:.0f}, τ(L→H) = {tau_LH:.0f}")
        print(f"  ratio = {ratio:.1f}, ΔV = {np.log(ratio) if ratio > 0 else 0:.2f}")
    
    # 绘图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # M* vs k
    ax1 = axes[0]
    ks = list(M_results.keys())
    Ms = [M_results[k]['mean'] for k in ks]
    ax1.plot(ks, Ms, 'bo-', markersize=10, linewidth=2)
    ax1.axhline(0.5, color='green', linestyle='--', alpha=0.5)
    ax1.set_xlabel('k')
    ax1.set_ylabel('M*')
    ax1.set_title('M*(k)')
    ax1.grid(True, alpha=0.3)
    
    # Barrier
    ax2 = axes[1]
    ratios = [rate_results[k]['ratio'] for k in ks]
    dVs = [rate_results[k]['dV'] for k in ks]
    ax2.bar(range(len(ks)), dVs, color='red', alpha=0.7)
    ax2.set_xticks(range(len(ks)))
    ax2.set_xticklabels([f'k={k}' for k in ks])
    ax2.set_ylabel('ΔV = log(τ_ratio)')
    ax2.set_title('Barrier Height')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # τ comparison
    ax3 = axes[2]
    tau_HL = [rate_results[k]['tau_HL'] for k in ks]
    tau_LH = [rate_results[k]['tau_LH'] for k in ks]
    x = np.arange(len(ks))
    width = 0.35
    ax3.bar(x - width/2, tau_HL, width, label='τ(H→L)', color='blue', alpha=0.7)
    ax3.bar(x + width/2, tau_LH, width, label='τ(L→H)', color='red', alpha=0.7)
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'k={k}' for k in ks])
    ax3.set_ylabel('τ')
    ax3.set_title('Escape Times')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('F:/hypergraph_bistability/figures/scaling_simple.png', dpi=150)
    print("\n[OK] Saved: figures/scaling_simple.png")
    
    # 总结
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for k in ks:
        r = rate_results[k]
        print(f"k={k}: M*={M_results[k]['mean']:.2f}, ΔV={r['dV']:.2f}")
    
    # 保存
    with open('F:/hypergraph_bistability/results/scaling_simple.json', 'w') as f:
        json.dump({
            'M': {str(k): v for k, v in M_results.items()},
            'rates': {str(k): v for k, v in rate_results.items()}
        }, f, indent=2)


if __name__ == '__main__':
    main()
