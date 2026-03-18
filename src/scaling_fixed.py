"""
修复版 Scaling Law 实验
======================

修复点：
1. 每个 k 用 20+ runs
2. 用概率 P(escape) 代替时间 τ
3. ΔV = log[P(L→H) / P(H→L)]
4. 精细扫描 k = 2.3, 2.35, 2.4, 2.45, 2.5, 2.55, 2.6
"""

import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json

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


def measure_with_probability(p_pair, k, n_runs=25, time_limit=150):
    """用概率方法测量 escape"""
    steps = 100  # 预处理步数
    
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
    
    # 计算 ΔV (用概率)
    if P_HL > 0 and P_LH > 0:
        dV = np.log(P_LH / P_HL)
    else:
        dV = 0
    
    return {
        'P_HL': P_HL,
        'P_LH': P_LH,
        'dV': dV,
        'has_bistability': P_LH > P_HL  # 如果能从 Low 逃到 High，说明有双稳态
    }


def main():
    print("=" * 60)
    print("修复版 Scaling Law 实验")
    print("=" * 60)
    
    # k = 2 + p_pair
    # 精细扫描 k_c 附近
    k_values = [2.3, 2.35, 2.4, 2.45, 2.5, 2.55, 2.6]
    p_values = [0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4]
    
    results = {}
    steps = 100  # 预处理步数
    
    print("\n[Measurement] Each k with 25 runs, probability-based ΔV")
    for p, k in zip(p_values, k_values):
        print(f"\nk={k}:", end=" ")
        
        # M* measurement
        Ms = []
        for i in range(15):
            h = HypergraphSimple(N=40, p_pair=p, seed=i*10+42)
            for _ in range(steps):
                h.apply_rules()
            Ms.append(h.get_M())
        
        M_mean = np.mean(Ms)
        
        # Probability measurement
        prob_results = measure_with_probability(p, k, n_runs=25, time_limit=150)
        
        print(f"M*={M_mean:.2f}, P_HL={prob_results['P_HL']:.2f}, P_LH={prob_results['P_LH']:.2f}, dV={prob_results['dV']:.2f}")
        
        results[k] = {
            'M': M_mean,
            'P_HL': prob_results['P_HL'],
            'P_LH': prob_results['P_LH'],
            'dV': prob_results['dV'],
            'has_bistability': prob_results['has_bistability']
        }
    
    # Plot
    print("\n[Plot] Three scaling tests")
    
    ks = sorted(results.keys())
    dVs = [results[k]['dV'] for k in ks]
    
    # 过滤掉无效点
    valid_data = [(k, dV) for k, dV in zip(ks, dVs) if dV > 0]
    if valid_data:
        valid_k = [x[0] for x in valid_data]
        valid_dV = [x[1] for x in valid_data]
    else:
        valid_k = []
        valid_dV = []
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. M* vs k
    ax1 = axes[0, 0]
    Ms = [results[k]['M'] for k in ks]
    ax1.plot(ks, Ms, 'bo-', markersize=10, linewidth=2)
    ax1.axhline(0.5, color='green', linestyle='--', alpha=0.5)
    ax1.set_xlabel('k')
    ax1.set_ylabel('M*')
    ax1.set_title('Order Parameter M*(k)')
    ax1.grid(True, alpha=0.3)
    
    # 2. ΔV vs k
    ax2 = axes[0, 1]
    colors = ['green' if dV > 0 else 'red' for dV in dVs]
    ax2.bar(ks, dVs, color=colors, alpha=0.7)
    ax2.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax2.set_xlabel('k')
    ax2.set_ylabel('ΔV')
    ax2.set_title('Barrier Height ΔV(k)')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Scaling 检验
    if len(valid_k) >= 3:
        k_c = 2.3  # 已知临界点
        dk = [k - k_c for k in valid_k]
        
        # 3a. 线性: ΔV vs (k - k_c)
        ax3 = axes[1, 0]
        ax3.scatter(dk, valid_dV, s=100, c='blue')
        # 线性拟合
        if len(dk) >= 2:
            z = np.polyfit(dk, valid_dV, 1)
            p = np.poly1d(z)
            x_fit = np.linspace(min(dk), max(dk), 50)
            ax3.plot(x_fit, p(x_fit), 'r--', linewidth=2, label=f'Linear: slope={z[0]:.2f}')
        ax3.set_xlabel('k - k_c')
        ax3.set_ylabel('ΔV')
        ax3.set_title('Test 1: Linear Scaling')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 3b. 幂律: log(ΔV) vs log(k - k_c)
        ax4 = axes[1, 1]
        log_dk = np.log(np.array(dk))
        log_dV = np.log(np.array(valid_dV))
        ax4.scatter(log_dk, log_dV, s=100, c='blue')
        if len(log_dk) >= 2:
            z = np.polyfit(log_dk, log_dV, 1)
            p = np.poly1d(z)
            x_fit = np.linspace(min(log_dk), max(log_dk), 50)
            ax4.plot(x_fit, p(x_fit), 'r--', linewidth=2, label=f'Power law: β={z[0]:.2f}')
        ax4.set_xlabel('log(k - k_c)')
        ax4.set_ylabel('log(ΔV)')
        ax4.set_title('Test 2: Power Law Scaling')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 判断 scaling 类型
        # 计算 R²
        if len(dk) >= 2:
            # 线性 R²
            p_lin = np.poly1d(np.polyfit(dk, valid_dV, 1))
            ss_res_lin = sum((valid_dV - p_lin(dk))**2)
            ss_tot = sum((valid_dV - np.mean(valid_dV))**2)
            r2_lin = 1 - ss_res_lin/ss_tot if ss_tot > 0 else 0
            
            # 幂律 R²
            p_pow = np.poly1d(np.polyfit(log_dk, log_dV, 1))
            ss_res_pow = sum((log_dV - p_pow(log_dk))**2)
            ss_tot_log = sum((log_dV - np.mean(log_dV))**2)
            r2_pow = 1 - ss_res_pow/ss_tot_log if ss_tot_log > 0 else 0
            
            print(f"\n[Scaling判定]")
            print(f"Linear R2 = {r2_lin:.3f}")
            print(f"Power law R2 = {r2_pow:.3f}")
            
            if r2_lin > r2_pow:
                print("Result: Linear scaling (dV ~ k - k_c)")
            else:
                print("Result: Power law scaling (dV ~ (k - k_c)^beta)")
    else:
        ax3 = axes[1, 0]
        ax3.text(0.5, 0.5, "数据点不足", ha='center', va='center', fontsize=14)
        ax4 = axes[1, 1]
        ax4.text(0.5, 0.5, "需要更多数据", ha='center', va='center', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('F:/hypergraph_bistability/figures/scaling_fixed.png', dpi=150)
    print("\n[OK] Saved: figures/scaling_fixed.png")
    
    # 保存
    with open('F:/hypergraph_bistability/results/scaling_fixed.json', 'w') as f:
        json.dump({
            'k_c': 2.3,
            'results': {str(k): v for k, v in results.items()}
        }, f, indent=2)
    print("[OK] Saved: results/scaling_fixed.json")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for k in ks:
        v = results[k]
        bist = "BISTABLE" if v['has_bistability'] else "NO BIST"
        print(f"k={k}: M*={v['M']:.2f}, dV={v['dV']:.2f}, {bist}")


if __name__ == '__main__':
    main()
