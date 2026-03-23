"""
快速 β 拟合
==========
"""

import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json
from scipy.optimize import curve_fit

os.makedirs('F:/hypergraph_bistability/figures', exist_ok=True)


class HypergraphSimple:
    def __init__(self, N=40, p_pair=0.5, seed=42):
        random.seed(seed)
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


def measure_dV(p_pair, n_runs=20, time_limit=120):
    """测量 ΔV"""
    HL_escaped = 0
    for i in range(n_runs):
        h = HypergraphSimple(N=40, p_pair=p_pair, seed=i*100+42)
        h.E = [frozenset(range(40))]
        for t in range(time_limit):
            h.apply_rules()
            if h.get_M() < 0.5:
                HL_escaped += 1
                break
    
    LH_escaped = 0
    for i in range(n_runs):
        h = HypergraphSimple(N=40, p_pair=p_pair, seed=i*100+42)
        h.E = [frozenset([j, j+1]) for j in range(0, 38, 2)]
        for t in range(time_limit):
            h.apply_rules()
            if h.get_M() > 0.7:
                LH_escaped += 1
                break
    
    P_HL = HL_escaped / n_runs
    P_LH = LH_escaped / n_runs
    
    if P_HL > 0 and P_LH > 0:
        dV = np.log(P_LH / P_HL)
    else:
        dV = 0
    
    return dV


def main():
    print("=" * 50)
    print("Quick Beta Fitting")
    print("=" * 50)
    
    # k values: 2.35 to 2.55
    k_values = [2.35, 2.40, 2.45, 2.50, 2.55]
    p_values = [0.65, 0.60, 0.55, 0.50, 0.45]
    
    results = {}
    
    for p, k in zip(p_values, k_values):
        print(f"k={k}:", end=" ")
        dV = measure_dV(p, n_runs=20, time_limit=120)
        print(f"dV={dV:.2f}")
        results[k] = dV
    
    # 过滤有效点（排除 k=2.35 太接近 k_c）
    valid = [(k, dV) for k, dV in results.items() if dV > 0.1 and k > 2.36]
    valid_k = np.array([x[0] for x in valid])
    valid_dV = np.array([x[1] for x in valid])
    
    # 固定 k_c = 2.35 拟合
    k_c = 2.35
    dk = valid_k - k_c
    
    # 线性拟合
    z_lin = np.polyfit(dk, valid_dV, 1)
    dV_lin = np.poly1d(z_lin)(dk)
    ss_res_lin = np.sum((valid_dV - dV_lin)**2)
    ss_tot = np.sum((valid_dV - np.mean(valid_dV))**2)
    r2_lin = 1 - ss_res_lin / ss_tot
    
    # 幂律拟合
    z_pow = np.polyfit(np.log(dk), np.log(valid_dV), 1)
    beta = z_pow[0]
    log_A = z_pow[1]
    A = np.exp(log_A)
    dV_pow = A * dk**beta
    ss_res_pow = np.sum((valid_dV - dV_pow)**2)
    r2_pow = 1 - ss_res_pow / ss_tot
    
    print(f"\n[Results]")
    print(f"Linear: slope={z_lin[0]:.2f}, R2={r2_lin:.3f}")
    print(f"Power: A={A:.2f}, beta={beta:.2f}, R2={r2_pow:.3f}")
    
    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Linear
    ax1 = axes[0]
    k_fit = np.linspace(2.35, 2.55, 50)
    dk_fit = k_fit - k_c
    ax1.scatter(valid_k, valid_dV, s=100, c='blue')
    ax1.plot(k_fit, z_lin[0]*dk_fit + z_lin[1], 'r--', 
             label=f'Linear (R2={r2_lin:.3f})')
    ax1.axvline(k_c, color='red', linestyle=':', alpha=0.5)
    ax1.set_xlabel('k')
    ax1.set_ylabel('dV')
    ax1.set_title('Linear Fit')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Log-log
    ax2 = axes[1]
    ax2.scatter(np.log(dk), np.log(valid_dV), s=100, c='blue')
    ax2.plot(np.log(dk_fit), z_pow[0]*np.log(dk_fit) + z_pow[1], 'r--',
             label=f'Power (beta={beta:.2f}, R2={r2_pow:.3f})')
    ax2.set_xlabel('log(k - k_c)')
    ax2.set_ylabel('log(dV)')
    ax2.set_title('Log-Log (Power Law)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('F:/hypergraph_bistability/figures/beta_fit.png', dpi=150)
    print("\n[OK] Saved: figures/beta_fit.png")
    
    # 保存
    with open('F:/hypergraph_bistability/results/beta_fit.json', 'w') as f:
        json.dump({
            'results': results,
            'linear': {'slope': float(z_lin[0]), 'R2': float(r2_lin)},
            'power': {'A': float(A), 'beta': float(beta), 'R2': float(r2_pow)}
        }, f, indent=2)
    
    print(f"\n*** FINAL: beta = {beta:.2f} ***")


if __name__ == '__main__':
    main()
