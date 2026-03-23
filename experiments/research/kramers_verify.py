"""
Kramers 验证：log τ ∝ 1/noise
=============================

固定 k = 2.5，改变 noise，验证 τ ~ exp(ΔV/noise)
"""

import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

os.makedirs('F:/hypergraph_bistability/figures', exist_ok=True)


class HypergraphKramers:
    def __init__(self, N=40, p_pair=0.5, noise=1.0, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        
        self.N = N
        self.p_pair = p_pair
        self.noise = noise
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
            # 分裂受 noise 控制
            if random.random() < 0.12 * self.noise and self.E:
                large = [e for e in self.E if len(e) > 2]
                if large:
                    e = random.choice(large)
                    nodes = list(e)
                    if len(nodes) >= 4:
                        split = len(nodes) // 2
                        self.E.append(frozenset(nodes[:split]))
                        self.E.append(frozenset(nodes[split:]))
                        self.E.remove(e)
            
            # 其他规则
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
            
            # K 容量限制
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


def measure_escape_time(p_pair, noise, n_runs=25, max_time=200):
    """测量 escape 时间"""
    
    # HIGH -> LOW
    HL_times = []
    for i in range(n_runs):
        h = HypergraphKramers(N=40, p_pair=p_pair, noise=noise, seed=i*100+42)
        h.E = [frozenset(range(40))]
        
        escape_time = None
        for t in range(max_time):
            h.apply_rules()
            if h.get_M() < 0.5:
                escape_time = t + 1
                break
        
        if escape_time:
            HL_times.append(escape_time)
    
    # LOW -> HIGH
    LH_times = []
    for i in range(n_runs):
        h = HypergraphKramers(N=40, p_pair=p_pair, noise=noise, seed=i*100+42)
        h.E = [frozenset([j, j+1]) for j in range(0, 38, 2)]
        
        escape_time = None
        for t in range(max_time):
            h.apply_rules()
            if h.get_M() > 0.7:
                escape_time = t + 1
                break
        
        if escape_time:
            LH_times.append(escape_time)
    
    tau_HL = np.mean(HL_times) if HL_times else None
    tau_LH = np.mean(LH_times) if LH_times else None
    
    return tau_HL, tau_LH


def main():
    print("=" * 60)
    print("Kramers 验证: log τ ∝ 1/noise")
    print("=" * 60)
    
    # 固定 k = 2.5 (p_pair = 0.5)
    p_pair = 0.5
    k = 2.5
    
    noise_values = [0.5, 0.8, 1.0, 1.5, 2.0, 3.0]
    inv_noise = [1/n for n in noise_values]
    
    results = {}
    
    print(f"\n[测量] k={k}, 变化 noise")
    
    for noise in noise_values:
        print(f"noise={noise}:", end=" ")
        
        tau_HL, tau_LH = measure_escape_time(p_pair, noise, n_runs=25, max_time=200)
        
        print(f"τ_HL={tau_HL:.1f}, τ_LH={tau_LH:.1f}")
        
        results[noise] = {
            'tau_HL': tau_HL,
            'tau_LH': tau_LH,
            'inv_noise': 1/noise
        }
    
    # 提取数据
    valid_HL = [(1/n, t) for n, t in [(noise, results[noise]['tau_HL']) 
                                        for noise in noise_values] 
                if t is not None and t < 200]
    valid_LH = [(1/n, t) for n, t in [(noise, results[noise]['tau_LH']) 
                                        for noise in noise_values] 
                if t is not None and t < 200]
    
    # 线性拟合
    if len(valid_HL) >= 3:
        x_HL = np.array([v[0] for v in valid_HL])
        y_HL = np.array([v[1] for v in valid_HL])
        z_HL = np.polyfit(x_HL, np.log(y_HL), 1)
        
        print(f"\n[结果]")
        print(f"H→L: log τ = {z_HL[0]:.2f} * (1/noise) + {z_HL[1]:.2f}")
        
        y_pred_HL = np.exp(z_HL[0]*x_HL + z_HL[1])
        ss_res_HL = np.sum((y_HL - y_pred_HL)**2)
        ss_tot_HL = np.sum((y_HL - np.mean(y_HL))**2)
        r2_HL = 1 - ss_res_HL / ss_tot_HL
        print(f"     R2 = {r2_HL:.3f}")
    else:
        z_HL = None
        r2_HL = 0
    
    if len(valid_LH) >= 3:
        x_LH = np.array([v[0] for v in valid_LH])
        y_LH = np.array([v[1] for v in valid_LH])
        z_LH = np.polyfit(x_LH, np.log(y_LH), 1)
        
        print(f"L→H: log τ = {z_LH[0]:.2f} * (1/noise) + {z_LH[1]:.2f}")
        
        y_pred_LH = np.exp(z_LH[0]*x_LH + z_LH[1])
        ss_res_LH = np.sum((y_LH - y_pred_LH)**2)
        ss_tot_LH = np.sum((y_LH - np.mean(y_LH))**2)
        r2_LH = 1 - ss_res_LH / ss_tot_LH
        print(f"     R2 = {r2_LH:.3f}")
    else:
        z_LH = None
        r2_LH = 0
    
    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # H -> L
    ax1 = axes[0]
    if valid_HL:
        x = np.array([v[0] for v in valid_HL])
        y = np.array([v[1] for v in valid_HL])
        ax1.scatter(x, y, s=100, c='blue', label='Data')
        
        if z_HL is not None:
            x_fit = np.linspace(min(x), max(x), 50)
            y_fit = np.exp(z_HL[0]*x_fit + z_HL[1])
            ax1.plot(x_fit, y_fit, 'r-', linewidth=2,
                    label=f'Fit: log τ = {z_HL[0]:.2f}(1/noise) + {z_HL[1]:.2f}')
        
        ax1.set_xlabel('1/noise')
        ax1.set_ylabel('tau_HL')
        ax1.set_title(f'H -> L Escape Time (R²={r2_HL:.3f})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # L -> H
    ax2 = axes[1]
    if valid_LH:
        x = np.array([v[0] for v in valid_LH])
        y = np.array([v[1] for v in valid_LH])
        ax2.scatter(x, y, s=100, c='green', label='Data')
        
        if z_LH is not None:
            x_fit = np.linspace(min(x), max(x), 50)
            y_fit = np.exp(z_LH[0]*x_fit + z_LH[1])
            ax2.plot(x_fit, y_fit, 'r-', linewidth=2,
                    label=f'Fit: log τ = {z_LH[0]:.2f}(1/noise) + {z_LH[1]:.2f}')
        
        ax2.set_xlabel('1/noise')
        ax2.set_ylabel('tau_LH')
        ax2.set_title(f'L -> H Escape Time (R²={r2_LH:.3f})')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('F:/hypergraph_bistability/figures/kramers_verification.png', dpi=150)
    print("\n[OK] Saved: figures/kramers_verification.png")
    
    # 总结
    print("\n" + "=" * 60)
    print("KRAMERS 验证结果")
    print("=" * 60)
    
    if z_HL is not None and r2_HL > 0.7:
        print(f"\n✅ H→L: Kramers 关系成立!")
        print(f"   log τ_HL = {z_HL[0]:.2f}/noise + const")
        print(f"   R² = {r2_HL:.3f}")
    else:
        print(f"\n❌ H→L: Kramers 关系不成立")
        print(f"   R² = {r2_HL:.3f}")
    
    if z_LH is not None and r2_LH > 0.7:
        print(f"\n✅ L→H: Kramers 关系成立!")
        print(f"   log τ_LH = {z_LH[0]:.2f}/noise + const")
        print(f"   R² = {r2_LH:.3f}")
    else:
        print(f"\n❌ L→H: Kramers 关系不成立")
        print(f"   R² = {r2_LH:.3f}")
    
    # 判定
    if r2_HL > 0.7 or r2_LH > 0.7:
        print("\n🎉 至少一个方向满足 Kramers 关系!")
    else:
        print("\n⚠️ Kramers 关系需要更多数据验证")


if __name__ == '__main__':
    main()
