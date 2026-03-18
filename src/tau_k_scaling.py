"""
τ vs k Scaling 实验
===================

目标：验证 τ 由结构 k 直接决定

固定 noise = 1.0
扫 k: 2.35 → 2.6
测量 τ_HL, τ_LH
"""

import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

os.makedirs('F:/hypergraph_bistability/figures', exist_ok=True)


class HypergraphTauK:
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


def measure_tau(p_pair, n_runs=30, max_time=250):
    """测量 escape 时间"""
    
    # HIGH -> LOW
    HL_times = []
    for i in range(n_runs):
        h = HypergraphTauK(N=40, p_pair=p_pair, seed=i*100+42)
        h.E = [frozenset(range(40))]
        
        for t in range(max_time):
            h.apply_rules()
            if h.get_M() < 0.5:
                HL_times.append(t + 1)
                break
        else:
            HL_times.append(max_time)
    
    # LOW -> HIGH
    LH_times = []
    for i in range(n_runs):
        h = HypergraphTauK(N=40, p_pair=p_pair, seed=i*100+42)
        h.E = [frozenset([j, j+1]) for j in range(0, 38, 2)]
        
        for t in range(max_time):
            h.apply_rules()
            if h.get_M() > 0.7:
                LH_times.append(t + 1)
                break
        else:
            LH_times.append(max_time)
    
    return np.mean(HL_times), np.mean(LH_times)


def main():
    print("=" * 60)
    print("Tau vs k Scaling Experiment")
    print("=" * 60)
    
    # k 到 p_pair 的映射: k = 2 + p_pair
    # 扫 k: 2.35 -> 2.60
    k_values = [2.35, 2.40, 2.45, 2.50, 2.55, 2.60]
    p_values = [0.65, 0.60, 0.55, 0.50, 0.45, 0.40]
    
    results = {}
    
    print("\n[Measuring] tau vs k")
    
    for p, k in zip(p_values, k_values):
        print(f"k={k}:", end=" ")
        
        tau_HL, tau_LH = measure_tau(p, n_runs=30, max_time=250)
        
        print(f"tau_HL={tau_HL:.1f}, tau_LH={tau_LH:.1f}")
        
        results[k] = {
            'tau_HL': tau_HL,
            'tau_LH': tau_LH,
            'p': p
        }
    
    # 分析
    ks = sorted(results.keys())
    tau_HL_list = [results[k]['tau_HL'] for k in ks]
    tau_LH_list = [results[k]['tau_LH'] for k in ks]
    
    print("\n[Analysis]")
    
    # 拟合: tau vs (k - k_c)
    k_c = 2.35
    
    # 只用 k > k_c 的数据
    valid = [(k, results[k]['tau_LH']) for k in ks if k > k_c]
    if len(valid) >= 3:
        vk = np.array([x[0] - k_c for x in valid])
        vtau = np.array([x[1] for x in valid])
        
        # 线性
        z_lin = np.polyfit(vk, vtau, 1)
        pred_lin = np.poly1d(z_lin)(vk)
        ss_res_lin = np.sum((vtau - pred_lin)**2)
        ss_tot = np.sum((vtau - np.mean(vtau))**2)
        r2_lin = 1 - ss_res_lin / ss_tot
        
        # 幂律: tau = A * (k - k_c)^gamma
        # log tau = log A + gamma * log(k - k_c)
        z_pow = np.polyfit(np.log(vk), np.log(vtau), 1)
        gamma = z_pow[0]
        log_A = z_pow[1]
        A = np.exp(log_A)
        pred_pow = A * vk**gamma
        ss_res_pow = np.sum((vtau - pred_pow)**2)
        r2_pow = 1 - ss_res_pow / ss_tot
        
        # 指数: tau = A * exp(b*k)
        z_exp = np.polyfit(ks, np.log(tau_LH_list), 1)
        b_exp = z_exp[0]
        A_exp = np.exp(z_exp[1])
        pred_exp = A_exp * np.exp(b_exp * np.array(ks))
        ss_res_exp = np.sum((np.array(tau_LH_list) - pred_exp)**2)
        r2_exp = 1 - ss_res_exp / ss_tot
        
        print(f"\nTau_LH vs (k - {k_c}):")
        print(f"  Linear: tau = {z_lin[0]:.1f}*(k-{k_c}) + {z_lin[1]:.1f}, R2={r2_lin:.3f}")
        print(f"  Power: tau = {A:.1f}*(k-{k_c})^{gamma:.2f}, R2={r2_pow:.3f}")
        print(f"  Exponential: tau = {A_exp:.1f}*exp({b_exp:.2f}*k), R2={r2_exp:.3f}")
        
        # 找最佳
        best = max([(r2_lin, 'linear', z_lin), 
                    (r2_pow, 'power', (A, gamma)), 
                    (r2_exp, 'exp', (A_exp, b_exp))], 
                   key=lambda x: x[0])
        
        print(f"\nBest fit: {best[1]} with R2={best[0]:.3f}")
        
        # 绘图
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 1. Raw data
        ax1 = axes[0]
        ax1.plot(ks, tau_HL_list, 'bo-', label='tau_HL', markersize=8)
        ax1.plot(ks, tau_LH_list, 'rs-', label='tau_LH', markersize=8)
        ax1.axvline(k_c, color='gray', linestyle='--', alpha=0.5, label=f'k_c={k_c}')
        ax1.set_xlabel('k')
        ax1.set_ylabel('tau')
        ax1.set_title('Tau vs k')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Power law fit
        ax2 = axes[1]
        ax2.scatter(vk, vtau, s=100, c='blue', label='Data')
        k_fit = np.linspace(0.01, max(vk)*1.1, 50)
        ax2.plot(k_fit, A * k_fit**gamma, 'r-', linewidth=2,
                label=f'Power: tau ~ (k-{k_c})^{gamma:.2f}')
        ax2.set_xlabel(f'k - {k_c}')
        ax2.set_ylabel('tau_LH')
        ax2.set_title(f'Power Law Fit (R2={r2_pow:.3f})')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Log-log
        ax3 = axes[2]
        ax3.scatter(np.log(vk), np.log(vtau), s=100, c='blue', label='Data')
        ax3.plot(np.log(k_fit), np.log(A) + gamma*np.log(k_fit), 'r-', linewidth=2,
                label=f'gamma = {gamma:.2f}')
        ax3.set_xlabel(f'log(k - {k_c})')
        ax3.set_ylabel('log(tau_LH)')
        ax3.set_title('Log-Log Plot')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('F:/hypergraph_bistability/figures/tau_k_scaling.png', dpi=150)
        print("\n[OK] Saved: figures/tau_k_scaling.png")
        
        # 保存
        import json
        with open('F:/hypergraph_bistability/results/tau_k_scaling.json', 'w') as f:
            json.dump({
                'results': {str(k): v for k, v in results.items()},
                'fits': {
                    'linear': {'slope': float(z_lin[0]), 'intercept': float(z_lin[1]), 'R2': float(r2_lin)},
                    'power': {'A': float(A), 'gamma': float(gamma), 'R2': float(r2_pow)},
                    'exp': {'A': float(A_exp), 'b': float(b_exp), 'R2': float(r2_exp)}
                },
                'best': {'type': best[1], 'R2': float(best[0])}
            }, f, indent=2)
        
        print("[OK] Saved: results/tau_k_scaling.json")
        
        # 最终结论
        print("\n" + "=" * 60)
        print("FINAL RESULT")
        print("=" * 60)
        
        if best[1] == 'power':
            print(f"\n*** Tau scales as: tau ~ (k - {k_c})^{gamma:.2f} ***")
            print(f"R² = {r2_pow:.3f}")
        elif best[1] == 'exp':
            print(f"\n*** Tau scales as: tau ~ exp({b_exp:.2f} * k) ***")
            print(f"R² = {r2_exp:.3f}")
        else:
            print(f"\n*** Tau scales linearly: tau ~ {z_lin[0]:.1f}*(k - {k_c}) ***")
            print(f"R² = {r2_lin:.3f}")
        
    else:
        print("Not enough data points")


if __name__ == '__main__':
    main()
