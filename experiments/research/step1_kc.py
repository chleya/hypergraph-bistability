"""
严格版 Scaling Law 实验
=======================

目标：
1. 精确锁定 k_c（ΔV 从 0 变正的点）
2. 判定 scaling 类型（线性 / 幂律 / 指数）
3. 验证 ΔV 定义（log(τ) ∝ 1/noise?）

Step 1: 窄扫描 k = 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6
"""

import numpy as np
import random
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
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
        self.K = 14
        
        self.V = list(range(N))
        self.E = []
        
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


def measure_k_c(n_runs=10, steps=150, time_limit=150):
    """测量不同 k 下的 ΔV"""
    # p_pair 到 k 的映射: p=1.0 -> k=2, p=0.0 -> k=3
    # k = 2 + p_pair
    k_values = [2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6]
    p_values = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    
    results = {}
    
    for p, k in zip(p_values, k_values):
        print(f"\nk={k} (p={p}):")
        
        # 先测 M*
        Ms = []
        for i in range(n_runs):
            h = HypergraphSimple(N=40, p_pair=p, seed=i*10+42)
            for _ in range(steps):
                h.apply_rules()
            Ms.append(h.get_M())
        
        M_mean = np.mean(Ms)
        print(f"  M* = {M_mean:.3f}")
        
        # 测 escape rates
        # HIGH -> LOW
        HL_times = []
        for i in range(n_runs):
            h = HypergraphSimple(N=40, p_pair=p, seed=i*10+42)
            h.E = [frozenset(range(40))]
            
            for t in range(time_limit):
                h.apply_rules()
                if h.get_M() < 0.5:
                    HL_times.append(t)
                    break
            else:
                HL_times.append(time_limit + 1)
        
        # LOW -> HIGH
        LH_times = []
        for i in range(n_runs):
            h = HypergraphSimple(N=40, p_pair=p, seed=i*10+42)
            h.E = [frozenset([j, j+1]) for j in range(0, 38, 2)]
            
            for t in range(time_limit):
                h.apply_rules()
                if h.get_M() > 0.7:
                    LH_times.append(t)
                    break
            else:
                LH_times.append(time_limit + 1)
        
        tau_HL = np.mean([t for t in HL_times if t <= time_limit])
        tau_LH = np.mean([t for t in LH_times if t <= time_limit])
        
        # 计算 ΔV
        if tau_HL > 0 and tau_LH <= time_limit and tau_LH > 0:
            ratio = tau_LH / tau_HL
            dV = np.log(ratio) if ratio > 1 else 0
        else:
            ratio = 0
            dV = 0
        
        print(f"  τ(H→L)={tau_HL:.0f}, τ(L→H)={tau_LH:.0f}, ratio={ratio:.1f}, ΔV={dV:.2f}")
        
        results[k] = {
            'M': M_mean,
            'tau_HL': tau_HL,
            'tau_LH': tau_LH,
            'ratio': ratio,
            'dV': dV,
            'has_bistability': dV > 0.1
        }
    
    return results


def main():
    print("=" * 60)
    print("Step 1: 精确锁定 k_c")
    print("=" * 60)
    
    results = measure_k_c(n_runs=8, steps=120, time_limit=120)
    
    # 分析 k_c
    print("\n" + "=" * 60)
    print("k_c 分析")
    print("=" * 60)
    
    for k, v in results.items():
        status = "BISTABLE" if v['has_bistability'] else "NO BISTABILITY"
        print(f"k={k}: dV={v['dV']:.2f}, {status}")
    
    # 找 k_c
    bistable_k = [k for k, v in results.items() if v['has_bistability']]
    if bistable_k:
        k_c = min(bistable_k)
        print(f"\nk_c ≈ {k_c} (双稳态出现的临界点)")
    else:
        k_c = None
        print("\nk_c 未确定，需要更多数据")
    
    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # M* vs k
    ax1 = axes[0]
    ks = list(results.keys())
    Ms = [results[k]['M'] for k in ks]
    ax1.plot(ks, Ms, 'bo-', markersize=10, linewidth=2)
    ax1.axhline(0.5, color='green', linestyle='--', alpha=0.5, label='M=0.5')
    if k_c:
        ax1.axvline(k_c, color='red', linestyle=':', alpha=0.5, label=f'k_c≈{k_c}')
    ax1.set_xlabel('k')
    ax1.set_ylabel('M*')
    ax1.set_title('Order Parameter M*(k)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ΔV vs k
    ax2 = axes[1]
    dVs = [results[k]['dV'] for k in ks]
    colors = ['green' if dV > 0.1 else 'red' for dV in dVs]
    ax2.bar(ks, dVs, color=colors, alpha=0.7)
    ax2.axhline(0, color='black', linestyle='-', alpha=0.3)
    if k_c:
        ax2.axvline(k_c, color='red', linestyle=':', alpha=0.5, label=f'k_c≈{k_c}')
    ax2.set_xlabel('k')
    ax2.set_ylabel('ΔV')
    ax2.set_title('Barrier Height ΔV(k)')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('F:/hypergraph_bistability/figures/k_c_lock.png', dpi=150)
    print("\n[OK] Saved: figures/k_c_lock.png")
    
    # 保存
    with open('F:/hypergraph_bistability/results/k_c_results.json', 'w') as f:
        json.dump({
            'k_c': k_c,
            'results': {str(k): v for k, v in results.items()}
        }, f, indent=2)
    print("[OK] Saved: results/k_c_results.json")
    
    # 总结
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"k_c ≈ {k_c}")
    print(f"k < k_c: 无双稳态 (ΔV ≈ 0)")
    print(f"k > k_c: 有双稳态 (ΔV > 0)")


if __name__ == '__main__':
    main()
