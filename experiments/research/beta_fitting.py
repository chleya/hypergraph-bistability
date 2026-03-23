"""
精细 β 拟合实验
==============

目标：在 k_c 附近做更密的扫描，精确拟合 β

k 范围：2.32 → 2.55（步长 0.02-0.03）
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


def measure_with_probability(p_pair, n_runs=30, time_limit=150):
    """用概率方法测量 escape"""
    steps = 100
    
    # HIGH -> LOW
    HL_escaped = 0
    for i in range(n_runs):
        h = HypergraphSimple(N=40, p_pair=p_pair, seed=i*100+42)
        h.E = [frozenset(range(40))]
        
        for t in range(time_limit):
            h.apply_rules()
            if h.get_M() < 0.5:
                HL_escaped += 1
                break
    
    # LOW -> HIGH
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
    
    return P_HL, P_LH, dV


def power_law(x, A, beta, k_c):
    """ΔV = A * (x - k_c)^beta"""
    return A * np.power(np.maximum(x - k_c, 0.001), beta)


def main():
    print("=" * 60)
    print("精细 β 拟合实验")
    print("=" * 60)
    
    # 精细 k 扫描
    # p_pair 到 k: k = 2 + p_pair
    k_values = [2.32, 2.34, 2.36, 2.38, 2.40, 2.42, 2.44, 2.46, 2.48, 2.50, 2.52, 2.54]
    p_values = [0.68, 0.66, 0.64, 0.62, 0.60, 0.58, 0.56, 0.54, 0.52, 0.50, 0.48, 0.46]
    
    results = {}
    
    print("\n[测量] 精细扫描 k_c 附近")
    for p, k in zip(p_values, k_values):
        print(f"k={k}:", end=" ")
        
        # M* 测量
        Ms = []
        for i in range(15):
            h = HypergraphSimple(N=40, p_pair=p, seed=i*10+42)
            for _ in range(100):
                h.apply_rules()
            Ms.append(h.get_M())
        M_mean = np.mean(Ms)
        
        # 概率测量
        P_HL, P_LH, dV = measure_with_probability(p, n_runs=30, time_limit=150)
        
        print(f"M*={M_mean:.2f}, dV={dV:.2f}")
        
        results[k] = {
            'M': M_mean,
            'P_HL': P_HL,
            'P_LH': P_LH,
            'dV': dV,
            'has_bistability': dV > 0.1
        }
    
    # 过滤有效数据点
    valid_data = [(k, v['dV']) for k, v in results.items() if v['dV'] > 0.05]
    
    if len(valid_data) >= 4:
        valid_k = np.array([x[0] for x in valid_data])
        valid_dV = np.array([x[1] for x in valid_data])
        
        print("\n[拟合] Power law: ΔV = A * (k - k_c)^β")
        
        # 固定 k_c = 2.35，拟合 A 和 beta
        k_c_fixed = 2.35
        
        try:
            # 幂律拟合
            popt, pcov = curve_fit(
                lambda x, A, beta: power_law(x, A, beta, k_c_fixed),
                valid_k, valid_dV,
                p0=[1.0, 1.0],
                maxfev=5000
            )
            A_fit, beta_fit = popt
            
            print(f"A = {A_fit:.3f}")
            print(f"beta = {beta_fit:.3f}")
            
            # 计算 R²
            dV_pred = power_law(valid_k, A_fit, beta_fit, k_c_fixed)
            ss_res = np.sum((valid_dV - dV_pred)**2)
            ss_tot = np.sum((valid_dV - np.mean(valid_dV))**2)
            r2 = 1 - ss_res / ss_tot
            
            print(f"R² = {r2:.3f}")
            
            # 也尝试自由 k_c 拟合
            print("\n[自由 k_c 拟合]")
            popt2, pcov2 = curve_fit(
                power_law,
                valid_k, valid_dV,
                p0=[1.0, 1.0, 2.3],
                bounds=([0.1, 0.1, 2.0], [10, 5, 2.5]),
                maxfev=5000
            )
            A_free, beta_free, kc_free = popt2
            
            print(f"A = {A_free:.3f}")
            print(f"beta = {beta_free:.3f}")
            print(f"k_c = {kc_free:.3f}")
            
            dV_pred2 = power_law(valid_k, A_free, beta_free, kc_free)
            ss_res2 = np.sum((valid_dV - dV_pred2)**2)
            r2_free = 1 - ss_res2 / ss_tot
            
            print(f"R² = {r2_free:.3f}")
            
        except Exception as e:
            print(f"拟合失败: {e}")
            A_fit, beta_fit, k_c_fixed = None, None, 2.35
            r2 = 0
            A_free, beta_free, kc_free = None, None, None
            r2_free = 0
    else:
        print("数据点不足")
        A_fit, beta_fit = None, None
        r2 = 0
        A_free, beta_free, kc_free = None, None, None
        r2_free = 0
    
    # 绘图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. M* vs k
    ax1 = axes[0, 0]
    ks = sorted(results.keys())
    Ms = [results[k]['M'] for k in ks]
    ax1.plot(ks, Ms, 'bo-', markersize=8, linewidth=2)
    ax1.axhline(0.5, color='green', linestyle='--', alpha=0.5)
    ax1.axvline(2.35, color='red', linestyle=':', alpha=0.5, label='k_c=2.35')
    ax1.set_xlabel('k')
    ax1.set_ylabel('M*')
    ax1.set_title('Order Parameter M*(k)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. ΔV vs k
    ax2 = axes[0, 1]
    dVs = [results[k]['dV'] for k in ks]
    colors = ['green' if dV > 0 else 'red' for dV in dVs]
    ax2.bar(ks, dVs, color=colors, alpha=0.7, width=0.015)
    ax2.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax2.axvline(2.35, color='red', linestyle=':', alpha=0.5, label='k_c=2.35')
    ax2.set_xlabel('k')
    ax2.set_ylabel('ΔV')
    ax2.set_title('Barrier Height ΔV(k)')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Power law fit (fixed k_c)
    ax3 = axes[1, 0]
    if A_fit is not None:
        k_fit = np.linspace(2.35, 2.55, 100)
        dV_fit = power_law(k_fit, A_fit, beta_fit, k_c_fixed)
        ax3.scatter(valid_k, valid_dV, s=100, c='blue', label='Data')
        ax3.plot(k_fit, dV_fit, 'r-', linewidth=2, 
                label=f'Fit: ΔV = {A_fit:.2f}(k-2.35)^{beta_fit:.2f}')
        ax3.set_xlabel('k')
        ax3.set_ylabel('ΔV')
        ax3.set_title(f'Power Law Fit (R²={r2:.3f})')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, "Fitting failed", ha='center', va='center')
    
    # 4. Log-log plot
    ax4 = axes[1, 1]
    if A_fit is not None:
        dk = valid_k - k_c_fixed
        log_dk = np.log(dk)
        log_dV = np.log(valid_dV)
        
        ax4.scatter(log_dk, log_dV, s=100, c='blue', label='Data')
        
        # 理论线
        k_theory = np.linspace(min(log_dk), max(log_dk), 50)
        dV_theory = np.log(A_fit) + beta_fit * k_theory
        ax4.plot(k_theory, dV_theory, 'r-', linewidth=2,
                label=f'β = {beta_fit:.2f}')
        
        ax4.set_xlabel('log(k - k_c)')
        ax4.set_ylabel('log(ΔV)')
        ax4.set_title('Log-Log Plot (Power Law Test)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, "Fitting failed", ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig('F:/hypergraph_bistability/figures/beta_fitting.png', dpi=150)
    print("\n[OK] Saved: figures/beta_fitting.png")
    
    # 保存
    with open('F:/hypergraph_bistability/results/beta_fitting.json', 'w') as f:
        json.dump({
            'results': {str(k): v for k, v in results.items()},
            'fit_fixed_kc': {
                'A': float(A_fit) if A_fit else None,
                'beta': float(beta_fit) if beta_fit else None,
                'R2': float(r2)
            },
            'fit_free_kc': {
                'A': float(A_free) if A_free else None,
                'beta': float(beta_free) if beta_free else None,
                'k_c': float(kc_free) if kc_free else None,
                'R2': float(r2_free)
            }
        }, f, indent=2)
    print("[OK] Saved: results/beta_fitting.json")
    
    # 总结
    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)
    print(f"\nFixed k_c = 2.35:")
    print(f"  ΔV = {A_fit:.2f} * (k - 2.35)^{beta_fit:.2f}")
    print(f"  R² = {r2:.3f}")
    
    if kc_free:
        print(f"\nFree k_c:")
        print(f"  ΔV = {A_free:.2f} * (k - {kc_free:.2f})^{beta_free:.2f}")
        print(f"  k_c = {kc_free:.2f}")
        print(f"  R² = {r2_free:.3f}")
    
    print(f"\n*** Beta = {beta_fit:.2f} ***")


if __name__ == '__main__':
    main()
