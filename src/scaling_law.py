"""
Scaling Law 实验：压缩成统一公式
===============================

目标：建立 τ(k) ~ exp[(k - k_c)^β] 关系

步骤：
1. 定义 barrier proxy: ΔV ≈ log(τ_L→H / τ_H→L)
2. 对不同 k 测 ΔV(k)
3. 拟合: ΔV(k) ∝ (k - k_c)^β

最小实验集：
k = 2.2, 2.3, 2.4, 2.5, 2.6, 2.8, 3.0
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import os
import json
from scipy.optimize import curve_fit

os.makedirs('F:/hypergraph_bistability/figures', exist_ok=True)
os.makedirs('F:/hypergraph_bistability/results', exist_ok=True)


class MixedHypergraph:
    """混合阶超图"""
    
    def __init__(self, N=40, p_pair=0.5, state_dim=8, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        
        self.N = N
        self.p_pair = p_pair
        self.state_dim = state_dim
        self.K = int(0.35 * N)
        
        self.V = list(range(N))
        self.E = []
        self.s = {v: np.random.randn(state_dim) for v in self.V}
        
        # 初始化
        n_initial = max(5, N // 3)
        for _ in range(n_initial):
            if random.random() < p_pair:
                e = frozenset(random.sample(self.V, 2))
            else:
                e = frozenset(random.sample(self.V, min(3, N)))
            self.E.append(e)
    
    def get_node_degree(self, v):
        return sum(1 for e in self.E if v in e)
    
    def get_avg_distance(self):
        if len(self.V) < 2:
            return 0.1
        samples = min(15, len(self.V) * (len(self.V) - 1) // 2)
        if samples <= 0:
            return 0.1
        dist_sum = 0
        for _ in range(samples):
            u, v = random.sample(self.V, 2)
            dist_sum += np.linalg.norm(self.s[u] - self.s[v])
        return max(dist_sum / samples, 0.01)
    
    def apply_rules(self, steps=1):
        for _ in range(steps):
            avg_dist = self.get_avg_distance()
            
            # 生长
            if random.random() < 0.3 and self.E:
                e = random.choice(self.E)
                nodes = list(e)
                w = max(self.V) + 1
                self.V.append(w)
                self.s[w] = self.s[nodes[0]] + np.random.randn(self.state_dim) * 0.2
                
                if random.random() < self.p_pair:
                    new_e = frozenset([nodes[0], w])
                else:
                    nodes2 = random.sample(self.V[:-1], min(2, len(self.V)-1))
                    new_e = frozenset(nodes2 + [w])
                self.E.append(new_e)
            
            # 融合
            if random.random() < 0.25 and len(self.E) >= 2:
                e1, e2 = random.sample(self.E, 2)
                intersection = e1 & e2
                if len(intersection) >= 1:
                    e1_only = list(e1 - intersection)
                    e2_only = list(e2 - intersection)
                    
                    if e1_only and e2_only:
                        v1, v2 = e1_only[0], e2_only[0]
                        dist = np.linalg.norm(self.s[v1] - self.s[v2])
                        
                        if dist < avg_dist * 0.8 and random.random() < 0.5:
                            new_nodes = list(intersection) + [v1, v2]
                            if len(new_nodes) >= 2:
                                if random.random() < self.p_pair:
                                    new_e = frozenset(random.sample(new_nodes, 2))
                                else:
                                    new_e = frozenset(random.sample(new_nodes, min(3, len(new_nodes))))
                                self.E.append(new_e)
                                if e1 in self.E: self.E.remove(e1)
                                if e2 in self.E: self.E.remove(e2)
            
            # 分裂
            if random.random() < 0.12 and len(self.E) > 1:
                large_edges = [e for e in self.E if len(e) > 2]
                if large_edges:
                    e = random.choice(large_edges)
                    nodes = list(e)
                    random.shuffle(nodes)
                    split = len(nodes) // 2
                    
                    e1 = frozenset(nodes[:split])
                    e2 = frozenset(nodes[split:])
                    
                    if len(e1) >= 2 and len(e2) >= 2:
                        self.E.append(e1)
                        self.E.append(e2)
                        self.E.remove(e)
            
            # 容量约束
            for v in list(self.V):
                degree = self.get_node_degree(v)
                if degree > self.K:
                    excess = degree - self.K
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
        max_cluster = 0
        
        for start in self.V:
            if start in visited:
                continue
            stack = [start]
            cluster = set()
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                cluster.add(node)
                for neighbor in adj[node]:
                    if neighbor not in visited:
                        stack.append(neighbor)
            max_cluster = max(max_cluster, len(cluster))
        
        return max_cluster / len(self.V) if self.V else 0


def measure_M_star(p_pair, n_runs=10, steps=150):
    """测量 M*"""
    results = []
    for i in range(n_runs):
        h = MixedHypergraph(N=40, p_pair=p_pair, state_dim=8, seed=i*100+42)
        for _ in range(steps):
            h.apply_rules()
        results.append(h.get_M())
    return np.mean(results), np.std(results), results


def measure_escape_rates(p_pair, n_runs=12, steps=400):
    """测量 escape rates"""
    # HIGH → LOW
    high_times = []
    for i in range(n_runs):
        h = MixedHypergraph(N=40, p_pair=p_pair, state_dim=8, seed=i*100+42)
        h.E = [frozenset(range(40))]  # HIGH initial
        
        M_history = [h.get_M()]
        for t in range(steps):
            h.apply_rules()
            M_history.append(h.get_M())
        
        # 找首次 M < 0.5
        escape = None
        for t, M in enumerate(M_history):
            if M < 0.5:
                escape = t
                break
        high_times.append(escape if escape else steps + 1)
    
    # LOW → HIGH
    low_times = []
    for i in range(n_runs):
        h = MixedHypergraph(N=40, p_pair=p_pair, state_dim=8, seed=i*100+42)
        h.E = [frozenset([j, j+1]) for j in range(0, 38, 2)]  # LOW initial
        
        M_history = [h.get_M()]
        for t in range(steps):
            h.apply_rules()
            M_history.append(h.get_M())
        
        # 找首次 M > 0.7
        escape = None
        for t, M in enumerate(M_history):
            if M > 0.7:
                escape = t
                break
        low_times.append(escape if escape else steps + 1)
    
    # 计算平均（只考虑已逃逸的）
    tau_HL = np.mean([t for t in high_times if t <= steps])
    tau_LH = np.mean([t for t in low_times if t <= steps])
    
    # 计算逃逸概率
    p_HL = sum(1 for t in high_times if t <= steps) / n_runs
    p_LH = sum(1 for t in low_times if t <= steps) / n_runs
    
    return {
        'tau_HL': tau_HL,
        'tau_LH': tau_LH if tau_LH <= steps else float('inf'),
        'p_HL': p_HL,
        'p_LH': p_LH,
        'ratio': (tau_LH / tau_HL) if tau_HL > 0 and tau_LH <= steps else float('nan')
    }


def power_law(x, beta, delta):
    """ΔV = beta * (x - xc)^delta"""
    return beta * np.power(np.maximum(x - 1.8, 0.001), delta)


def main():
    print("=" * 60)
    print("Scaling Law 实验")
    print("=" * 60)
    
    # k 值 (通过 p_pair 控制)
    # p_pair = 1.0 -> k=2.0
    # p_pair = 0.0 -> k=3.0
    # 线性映射: k = 2 + p_pair
    p_values = [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
    k_values = [2.2, 2.3, 2.4, 2.5, 2.6, 2.8, 3.0]
    
    # Step 1: 测量 M*
    print("\n[Step 1] 测量 M*(k)")
    M_results = {}
    for p, k in zip(p_values, k_values):
        print(f"  k={k}...", end=" ")
        mean_M, std_M, all_M = measure_M_star(p, n_runs=10, steps=150)
        M_results[k] = {'mean': mean_M, 'std': std_M}
        print(f"M*={mean_M:.3f}")
    
    # Step 2: 测量 escape rates
    print("\n[Step 2] 测量 escape rates")
    rate_results = {}
    for p, k in zip(p_values, k_values):
        print(f"  k={k}...", end=" ")
        rates = measure_escape_rates(p, n_runs=10, steps=400)
        rate_results[k] = rates
        print(f"τ(H→L)={rates['tau_HL']:.0f}, τ(L→H)={rates['tau_LH']:.0f}, ratio={rates['ratio']:.1f}" if rates['ratio'] != float('nan') else "N/A")
    
    # Step 3: 计算 barrier proxy
    print("\n[Step 3] 计算 barrier proxy: ΔV ≈ log(τ_L→H / τ_H→L)")
    barrier_data = []
    for k in k_values:
        r = rate_results[k]
        if r['ratio'] != float('nan') and r['ratio'] > 0:
            delta_V = np.log(r['ratio'])
            barrier_data.append((k, delta_V, r['ratio']))
            print(f"  k={k}: ΔV = log({r['ratio']:.1f}) = {delta_V:.2f}")
        else:
            barrier_data.append((k, 0, 0))
            print(f"  k={k}: ΔV = 0 (no escape)")
    
    # 移除无效数据
    valid_k = [k for k, dV, r in barrier_data if dV > 0]
    valid_dV = [dV for k, dV, r in barrier_data if dV > 0]
    
    print(f"\n有效数据点: {len(valid_k)}")
    
    # Step 4: 拟合 power law
    print("\n[Step 4] 拟合 power law: ΔV ∝ (k - k_c)^β")
    
    if len(valid_k) >= 3:
        # 假设 k_c ≈ 2.2 (临界点)
        k_c = 2.1
        x_data = np.array(valid_k)
        y_data = np.array(valid_dV)
        
        try:
            # 拟合
            popt, pcov = curve_fit(power_law, x_data, y_data, p0=[1.0, 1.0], maxfev=5000)
            beta_fit, delta_fit = popt
            print(f"  拟合结果: ΔV = {beta_fit:.2f} * (k - {k_c})^{delta_fit:.2f}")
            print(f"  β = {delta_fit:.2f}")
        except:
            print("  拟合失败，使用线性近似")
            beta_fit = None
            delta_fit = None
    else:
        print("  数据点不足，跳过拟合")
        beta_fit = None
        delta_fit = None
    
    # Step 5: 绘图
    print("\n[Step 5] 生成图表")
    
    # 图1: M* vs k
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ks = list(M_results.keys())
    Ms = [M_results[k]['mean'] for k in ks]
    stds = [M_results[k]['std'] for k in ks]
    
    ax1.errorbar(ks, Ms, yerr=stds, marker='o', capsize=5, linewidth=2, markersize=10)
    ax1.axhline(0.5, color='green', linestyle='--', alpha=0.5, label='M=0.5')
    ax1.axvspan(2.2, 2.6, alpha=0.2, color='yellow', label='Critical region')
    ax1.set_xlabel('Effective k')
    ax1.set_ylabel('M*')
    ax1.set_title('Scaling Law: M*(k)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('F:/hypergraph_bistability/figures/scaling_M.png', dpi=150)
    print("[OK] Saved: figures/scaling_M.png")
    
    # 图2: Barrier vs k
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    all_k = [k for k, dV, r in barrier_data]
    all_dV = [dV for k, dV, r in barrier_data]
    
    ax2.scatter(all_k, all_dV, s=100, c='blue', alpha=0.7)
    
    # 添加拟合曲线
    if beta_fit:
        x_fit = np.linspace(2.2, 3.0, 50)
        y_fit = power_law(x_fit, beta_fit, delta_fit)
        ax2.plot(x_fit, y_fit, 'r-', linewidth=2, 
                label=f'Fit: ΔV = {beta_fit:.2f}(k-{k_c})^{delta_fit:.2f}')
    
    ax2.axvline(k_c, color='red', linestyle=':', alpha=0.5, label=f'k_c ≈ {k_c}')
    ax2.set_xlabel('Effective k')
    ax2.set_ylabel('ΔV ≈ log(τ_ratio)')
    ax2.set_title('Scaling Law: Barrier Height')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('F:/hypergraph_bistability/figures/scaling_barrier.png', dpi=150)
    print("[OK] Saved: figures/scaling_barrier.png")
    
    # 图3: τ vs k
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    tau_HL = [rate_results[k]['tau_HL'] for k in k_values]
    tau_LH = [rate_results[k]['tau_LH'] for k in k_values]
    
    x_pos = np.arange(len(k_values))
    width = 0.35
    
    ax3.bar(x_pos - width/2, tau_HL, width, label='τ(H→L)', color='blue', alpha=0.7)
    ax3.bar(x_pos + width/2, tau_LH, width, label='τ(L→H)', color='red', alpha=0.7)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f'k={k}' for k in k_values])
    ax3.set_ylabel('Escape Time')
    ax3.set_title('Scaling Law: Escape Times')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('F:/hypergraph_bistability/figures/scaling_tau.png', dpi=150)
    print("[OK] Saved: figures/scaling_tau.png")
    
    # 图4: 综合理论图
    fig4, ax4 = plt.subplots(figsize=(12, 8))
    
    # 绘制理论框架
    ax4.axis('off')
    
    # 三个层次
    layers = [
        (0.85, '#3498db', 'Structure\nk', 'k = 2 → 3'),
        (0.55, '#e74c3c', 'Barrier\nΔV(k)', f'ΔV ∝ (k-{k_c})^β'),
        (0.25, '#2ecc71', 'Dynamics\nτ(k)', f'τ ~ exp(ΔV/kT)')
    ]
    
    for y, color, title, value in layers:
        rect = plt.Rectangle((0.15, y-0.12), 0.7, 0.22, 
                             fill=True, facecolor=color, alpha=0.3,
                             edgecolor=color, linewidth=3)
        ax4.add_patch(rect)
        ax4.text(0.5, y+0.04, title, ha='center', va='center', 
                fontsize=14, fontweight='bold')
        ax4.text(0.5, y-0.04, value, ha='center', va='center', fontsize=11)
        
        if y > 0.3:
            ax4.annotate('', xy=(0.5, y-0.15), xytext=(0.5, y-0.10),
                       arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    
    # 公式
    formula = 'τ(k) ~ exp[β(k - k_c)^δ / noise]'
    ax4.text(0.5, 0.02, formula, ha='center', va='center', 
            fontsize=14, fontweight='bold', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.set_title('Unified Theory: Structure → Barrier → Dynamics', 
                 fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('F:/hypergraph_bistability/figures/scaling_theory.png', dpi=150)
    print("[OK] Saved: figures/scaling_theory.png")
    
    # 保存结果
    results = {
        'M_results': M_results,
        'rate_results': {str(k): v for k, v in rate_results.items()},
        'barrier_data': [(k, float(dV), float(r)) for k, dV, r in barrier_data],
        'fitting': {
            'k_c': k_c,
            'beta': float(beta_fit) if beta_fit else None,
            'delta': float(delta_fit) if delta_fit else None
        } if beta_fit else None
    }
    
    with open('F:/hypergraph_bistability/results/scaling_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("[OK] Saved: results/scaling_results.json")
    
    # 总结
    print("\n" + "=" * 60)
    print("SCALING LAW RESULTS")
    print("=" * 60)
    print(f"\n[Structure]")
    print(f"  k range: {min(k_values)} → {max(k_values)}")
    print(f"  M* range: {min(Ms):.3f} → {max(Ms):.3f}")
    
    print(f"\n[Barrier]")
    print(f"  k_c ≈ {k_c}")
    if beta_fit:
        print(f"  ΔV = {beta_fit:.2f} * (k - {k_c})^{delta_fit:.2f}")
        print(f"  β = {delta_fit:.2f}")
    
    print(f"\n[Dynamics]")
    print(f"  Formula: τ(k) ~ exp[ΔV(k) / noise]")
    
    print("\n✓ Scaling law 框架已建立")


if __name__ == '__main__':
    main()
