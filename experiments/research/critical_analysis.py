"""
临界区深度分析
==============

三件关键实验：
1. P(HIGH | n_init) 曲线
2. 收敛时间 vs n_init (critical slowing)
3. 结构 vs 数量的解耦
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import os

os.makedirs('F:/hypergraph_bistability/figures/critical', exist_ok=True)


class Hypergraph:
    def __init__(self, N=50, gamma=0.35, state_dim=16, seed=42, n_initial=4):
        random.seed(seed)
        np.random.seed(seed)
        
        self.N = N
        self.gamma = gamma
        self.state_dim = state_dim
        self.K = int(gamma * N)
        self.n_initial = n_initial
        
        self.V = list(range(N))
        self.E = []
        self.s = {v: np.random.randn(state_dim) for v in self.V}
        
        for _ in range(n_initial):
            size = random.randint(2, min(5, N // 3))
            e = frozenset(random.sample(self.V, size))
            self.E.append(e)
    
    def get_node_degree(self, v):
        return sum(1 for e in self.E if v in e)
    
    def get_avg_distance(self):
        if len(self.V) < 2:
            return 0.1
        samples = min(20, len(self.V) * (len(self.V) - 1) // 2)
        if samples <= 0:
            return 0.1
        dist_sum = 0
        for _ in range(samples):
            u, v = random.sample(self.V, 2)
            dist_sum += np.linalg.norm(self.s[u] - self.s[v])
        return max(dist_sum / samples, 0.01)
    
    def apply_rules(self):
        avg_dist = self.get_avg_distance()
        
        if random.random() < 0.35 and self.E:
            e = random.choice(self.E)
            nodes = list(e)
            w = len(self.V)
            self.V.append(w)
            self.s[w] = self.s[nodes[0]] + np.random.randn(self.state_dim) * 0.2
            new_e = frozenset([nodes[0], w])
            self.E.append(new_e)
        
        if random.random() < 0.3 and len(self.E) >= 2:
            e1, e2 = random.sample(self.E, 2)
            if len(e1 & e2) >= 1:
                weights1 = np.array([1.0 / (1 + self.get_node_degree(v)) for v in e1])
                weights1 = weights1 / weights1.sum()
                c1 = np.average([self.s[u] for u in e1], axis=0, weights=weights1)
                
                weights2 = np.array([1.0 / (1 + self.get_node_degree(v)) for v in e2])
                weights2 = weights2 / weights2.sum()
                c2 = np.average([self.s[u] for u in e2], axis=0, weights=weights2)
                
                dist = np.linalg.norm(c1 - c2)
                if dist < avg_dist * 0.8:
                    new_e = frozenset(e1 | e2)
                    if new_e not in self.E:
                        self.E.append(new_e)
                        self.E.remove(e1)
                        self.E.remove(e2)
        
        if random.random() < 0.15 and len(self.E) > 1:
            large_edges = [e for e in self.E if len(e) > 2]
            if large_edges:
                e = random.choice(large_edges)
                nodes = list(e)
                random.shuffle(nodes)
                split = len(nodes) // 2
                self.E.append(frozenset(nodes[:split]))
                self.E.append(frozenset(nodes[split:]))
                self.E.remove(e)
        
        for v in self.V:
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
                for e in self.E:
                    if node in e:
                        for neighbor in e:
                            if neighbor not in visited:
                                stack.append(neighbor)
            max_cluster = max(max_cluster, len(cluster))
        
        return max_cluster / len(self.V) if self.V else 0
    
    def run_with_convergence(self, steps=80):
        """运行并记录收敛时间"""
        trajectory = [self.get_M()]
        
        for t in range(steps):
            self.apply_rules()
            trajectory.append(self.get_M())
        
        final_M = trajectory[-1]
        
        # 计算收敛时间：首次进入最终态10%范围内
        conv_time = steps
        for t in range(len(trajectory)):
            if abs(trajectory[t] - final_M) < max(final_M * 0.1, 0.05):
                conv_time = t
                break
        
        return final_M, conv_time, trajectory


def experiment1_P_high_curve():
    """实验1: P(HIGH | n_init) 曲线"""
    print("=" * 60)
    print("实验1: P(HIGH | n_init) 曲线")
    print("=" * 60)
    
    results = []
    
    for n_init in range(5, 31):
        high_count = 0
        total = 100
        
        for seed in range(total):
            sys = Hypergraph(N=50, gamma=0.35, seed=seed, n_initial=n_init)
            M, _, _ = sys.run_with_convergence(steps=60)
            
            if M > 0.4:  # HIGH 定义
                high_count += 1
        
        P_high = high_count / total
        results.append({
            'n_init': n_init,
            'P_high': P_high,
            'high_count': high_count,
            'low_count': total - high_count
        })
        
        print(f"n_init = {n_init}: P(HIGH) = {P_high:.2f} ({high_count}/{total})")
    
    # 绘图
    n_inits = [r['n_init'] for r in results]
    P_highs = [r['P_high'] for r in results]
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(n_inits, P_highs, marker='o', linewidth=2, markersize=6)
    plt.xlabel('Initial Hyperedge Count', fontsize=12)
    plt.ylabel('P(HIGH)', fontsize=12)
    plt.title('Transition Curve: P(HIGH | n_init)', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # 标记临界区
    plt.axvline(x=15, color='r', linestyle='--', alpha=0.5, label='Critical Zone')
    plt.axvline(x=16, color='r', linestyle='--', alpha=0.5)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    # 堆叠柱状图
    lows = [r['low_count'] for r in results]
    highs = [r['high_count'] for r in results]
    plt.bar(n_inits, lows, label='LOW', alpha=0.7)
    plt.bar(n_inits, highs, bottom=lows, label='HIGH', alpha=0.7)
    plt.xlabel('Initial Hyperedge Count')
    plt.ylabel('Count')
    plt.title('LOW vs HIGH Distribution')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('F:/hypergraph_bistability/figures/critical/P_high_curve.png', dpi=150)
    plt.close()
    
    # 分析曲线特性
    # 找50%点
    for r in results:
        if r['P_high'] >= 0.5:
            print(f"\n临界点 (P=50%): n_init ≈ {r['n_init']}")
            break
    
    # 计算斜率
    for i in range(1, len(results)):
        dP = results[i]['P_high'] - results[i-1]['P_high']
        dn = results[i]['n_init'] - results[i-1]['n_init']
        if dn > 0:
            slope = dP / dn
            if abs(slope) > 0.1:
                print(f"最大斜率: n_init={results[i]['n_init']}, slope={slope:.2f}")
    
    return results


def experiment2_convergence_time():
    """实验2: 收敛时间 vs n_init (critical slowing)"""
    print("\n" + "=" * 60)
    print("实验2: 收敛时间 vs n_init (Critical Slowing)")
    print("=" * 60)
    
    results = []
    
    for n_init in range(8, 25):
        conv_times = []
        
        for seed in range(100):
            sys = Hypergraph(N=50, gamma=0.35, seed=seed, n_initial=n_init)
            _, conv_time, _ = sys.run_with_convergence(steps=80)
            conv_times.append(conv_time)
        
        mean_conv = np.mean(conv_times)
        std_conv = np.std(conv_times)
        
        results.append({
            'n_init': n_init,
            'mean_conv': mean_conv,
            'std_conv': std_conv
        })
        
        print(f"n_init = {n_init}: 收敛时间 = {mean_conv:.1f} ± {std_conv:.1f}")
    
    # 绘图
    n_inits = [r['n_init'] for r in results]
    mean_convs = [r['mean_conv'] for r in results]
    std_convs = [r['std_conv'] for r in results]
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(n_inits, mean_convs, yerr=std_convs, marker='o', capsize=5)
    plt.xlabel('Initial Hyperedge Count', fontsize=12)
    plt.ylabel('Convergence Time', fontsize=12)
    plt.title('Critical Slowing Down: Convergence Time vs n_init', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # 标记临界区
    plt.axvline(x=15, color='r', linestyle='--', alpha=0.5, label='Critical Zone')
    plt.axvline(x=16, color='r', linestyle='--', alpha=0.5)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('F:/hypergraph_bistability/figures/critical/convergence_time.png', dpi=150)
    plt.close()
    
    # 找收敛最慢的点
    max_conv_idx = np.argmax(mean_convs)
    print(f"\n收敛最慢: n_init = {n_inits[max_conv_idx]}, 时间 = {mean_convs[max_conv_idx]:.1f}")
    
    return results


def experiment3_structure_vs_quantity():
    """实验3: 结构 vs 数量的解耦"""
    print("\n" + "=" * 60)
    print("实验3: 结构 vs 数量 解耦实验")
    print("=" * 60)
    
    # 固定 n_init = 16 (临界区)，但改变初始结构
    
    # 方案1: 大超边为主
    print("\n方案1: 大超边为主")
    results1 = []
    for seed in range(50):
        random.seed(seed)
        np.random.seed(seed)
        
        sys = Hypergraph(N=50, gamma=0.35, seed=seed, n_initial=16)
        # 修改初始超边：都变成大超边
        sys.E = []
        for _ in range(8):  # 8个大超边
            size = random.randint(4, 6)  # 大
            e = frozenset(random.sample(sys.V, size))
            sys.E.append(e)
        for _ in range(8):  # 8个小超边
            size = random.randint(2, 3)  # 小
            e = frozenset(random.sample(sys.V, size))
            sys.E.append(e)
        
        M, _, _ = sys.run_with_convergence(steps=60)
        results1.append(M)
    
    print(f"  大超边为主: M = {np.mean(results1):.3f} ± {np.std(results1):.3f}")
    
    # 方案2: 小超边为主
    print("\n方案2: 小超边为主")
    results2 = []
    for seed in range(50):
        random.seed(seed)
        np.random.seed(seed)
        
        sys = Hypergraph(N=50, gamma=0.35, seed=seed, n_initial=16)
        sys.E = []
        for _ in range(16):  # 16个小超边
            size = random.randint(2, 3)
            e = frozenset(random.sample(sys.V, size))
            sys.E.append(e)
        
        M, _, _ = sys.run_with_convergence(steps=60)
        results2.append(M)
    
    print(f"  小超边为主: M = {np.mean(results2):.3f} ± {np.std(results2):.3f}")
    
    # 方案3: 随机（对照）
    print("\n方案3: 随机（对照）")
    results3 = []
    for seed in range(50):
        sys = Hypergraph(N=50, gamma=0.35, seed=seed, n_initial=16)
        M, _, _ = sys.run_with_convergence(steps=60)
        results3.append(M)
    
    print(f"  随机: M = {np.mean(results3):.3f} ± {np.std(results3):.3f}")
    
    # 方案4: 单一超大超边
    print("\n方案4: 单一超大超边")
    results4 = []
    for seed in range(50):
        random.seed(seed)
        np.random.seed(seed)
        
        sys = Hypergraph(N=50, gamma=0.35, seed=seed, n_initial=16)
        sys.E = []
        # 1个超大 + 15个小
        e = frozenset(random.sample(sys.V, 15))
        sys.E.append(e)
        for _ in range(15):
            size = random.randint(2, 3)
            e = frozenset(random.sample(sys.V, size))
            sys.E.append(e)
        
        M, _, _ = sys.run_with_convergence(steps=60)
        results4.append(M)
    
    print(f"  单一超大: M = {np.mean(results4):.3f} ± {np.std(results4):.3f}")
    
    # 绘图
    plt.figure(figsize=(10, 6))
    
    data = [results1, results2, results3, results4]
    labels = ['Big Edges', 'Small Edges', 'Random', 'One Big + Small']
    
    plt.boxplot(data, labels=labels)
    plt.ylabel('Final M', fontsize=12)
    plt.title('Structure vs Quantity: n_init=16 (Critical Zone)', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('F:/hypergraph_bistability/figures/critical/structure_quantity.png', dpi=150)
    plt.close()
    
    # 统计分析
    from scipy import stats
    
    # t检验
    t1, p1 = stats.ttest_ind(results1, results2)
    print(f"\n统计检验 (大 vs 小超边): t = {t1:.2f}, p = {p1:.4f}")
    
    if p1 < 0.05:
        print("→ 结构差异显著！")
    else:
        print("→ 结构差异不显著")
    
    return {
        'big': results1,
        'small': results2,
        'random': results3,
        'one_big': results4
    }


def main():
    print("=" * 60)
    print("临界区深度分析")
    print("=" * 60)
    
    # 实验1: P(HIGH) 曲线
    r1 = experiment1_P_high_curve()
    
    # 实验2: 收敛时间
    r2 = experiment2_convergence_time()
    
    # 实验3: 结构vs数量
    r3 = experiment3_structure_vs_quantity()
    
    print("\n" + "=" * 60)
    print("所有实验完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
