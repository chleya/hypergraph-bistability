"""
Constraint Complexity Test - 约束复杂度测试
=====================================

目标：验证"约束复杂度 → 稳态数量选择"

核心问题：
- 约束复杂度 C 增加时，稳态数量如何变化？
- 是否存在 forbidden regions（不可能的稳态数）？
- 是否存在 plateau（平台区）？

操作定义：
C = 竞争的约束数量
   = 群体冲突强度 + 规则多样性 + 容量约束变化
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import os
from collections import defaultdict

os.makedirs('F:/hypergraph_bistability/figures/constraint_complexity', exist_ok=True)


class Hypergraph:
    """原项目核心超图系统"""
    
    def __init__(self, N=50, gamma=0.35, state_dim=16, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        
        self.N = N
        self.gamma = gamma
        self.state_dim = state_dim
        self.K = int(gamma * N)
        
        self.V = list(range(N))
        self.E = []
        self.s = {v: np.random.randn(state_dim) for v in self.V}
        
        n_initial = max(4, N // 4)
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
    
    def run(self, steps=80):
        for _ in range(steps):
            self.apply_rules()
        return self.get_M()


class VariableConstraintHypergraph:
    """可变约束超图：可以调节约束复杂度"""
    
    def __init__(self, N=50, gamma=0.35, constraint_conflict=0.0, 
                 rule_diversity=1.0, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        
        self.N = N
        self.gamma = gamma
        self.constraint_conflict = constraint_conflict  # 约束冲突强度 (0-1)
        self.rule_diversity = rule_diversity  # 规则多样性 (0-1)
        self.K = int(gamma * N)
        
        self.V = list(range(N))
        self.E = []
        self.s = {v: np.random.randn(16) for v in self.V}
        
        n_initial = max(4, N // 4)
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
        
        # 规则1: 生长 (受 rule_diversity 影响)
        growth_prob = 0.35 * self.rule_diversity
        if random.random() < growth_prob and self.E:
            e = random.choice(self.E)
            nodes = list(e)
            w = len(self.V)
            self.V.append(w)
            # 状态受 constraint_conflict 影响
            noise = np.random.randn(16) * (1 + self.constraint_conflict)
            self.s[w] = self.s[nodes[0]] + noise * 0.2
            new_e = frozenset([nodes[0], w])
            self.E.append(new_e)
        
        # 规则2: 融合 (受 constraint_conflict 影响)
        # 高冲突 = 低融合率
        fusion_prob = 0.3 * (1 - self.constraint_conflict * 0.5)
        if random.random() < fusion_prob and len(self.E) >= 2:
            e1, e2 = random.sample(self.E, 2)
            if len(e1 & e2) >= 1:
                weights1 = np.array([1.0 / (1 + self.get_node_degree(v)) for v in e1])
                weights1 = weights1 / weights1.sum()
                c1 = np.average([self.s[u] for u in e1], axis=0, weights=weights1)
                
                weights2 = np.array([1.0 / (1 + self.get_node_degree(v)) for v in e2])
                weights2 = weights2 / weights2.sum()
                c2 = np.average([self.s[u] for u in e2], axis=0, weights=weights2)
                
                dist = np.linalg.norm(c1 - c2)
                # 高冲突 = 更高的融合阈值
                threshold_mult = 1 + self.constraint_conflict
                if dist < avg_dist * 0.8 * threshold_mult:
                    new_e = frozenset(e1 | e2)
                    if new_e not in self.E:
                        self.E.append(new_e)
                        self.E.remove(e1)
                        self.E.remove(e2)
        
        # 规则3: 分裂 (受 rule_diversity 影响)
        split_prob = 0.15 * self.rule_diversity
        if random.random() < split_prob and len(self.E) > 1:
            large_edges = [e for e in self.E if len(e) > 2]
            if large_edges:
                e = random.choice(large_edges)
                nodes = list(e)
                random.shuffle(nodes)
                split = len(nodes) // 2
                self.E.append(frozenset(nodes[:split]))
                self.E.append(frozenset(nodes[split:]))
                self.E.remove(e)
        
        # 规则4: 容量约束
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
    
    def run(self, steps=80):
        for _ in range(steps):
            self.apply_rules()
        return self.get_M()


def test_constraint_conflict():
    """测试约束冲突强度对稳态的影响"""
    print("=" * 60)
    print("Test 1: Constraint Conflict vs Stable States")
    print("=" * 60)
    
    results = []
    
    for conflict in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        print(f"\nconstraint_conflict = {conflict}")
        
        M_values = []
        for seed in range(100):
            sys = VariableConstraintHypergraph(
                N=50, gamma=0.35, 
                constraint_conflict=conflict,
                seed=seed
            )
            M = sys.run(steps=60)
            M_values.append(M)
        
        # 统计
        mean_M = np.mean(M_values)
        std_M = np.std(M_values)
        
        # 找双峰
        hist, bins = np.histogram(M_values, bins=20)
        peaks = []
        for i in range(1, len(hist) - 1):
            if hist[i] > 5 and hist[i] > hist[i-1] and hist[i] > hist[i+1]:
                peaks.append((bins[i] + bins[i+1]) / 2)
        
        results.append({
            'conflict': conflict,
            'mean_M': mean_M,
            'std_M': std_M,
            'n_peaks': len(peaks),
            'M_values': M_values
        })
        
        print(f"  M = {mean_M:.3f} ± {std_M:.3f}, peaks = {len(peaks)}")
    
    # 绘图
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    conflicts = [r['conflict'] for r in results]
    means = [r['mean_M'] for r in results]
    stds = [r['std_M'] for r in results]
    plt.errorbar(conflicts, means, yerr=stds, marker='o', capsize=5)
    plt.xlabel('Constraint Conflict')
    plt.ylabel('Final M')
    plt.title('Mean M vs Constraint Conflict')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    n_peaks = [r['n_peaks'] for r in results]
    plt.bar(conflicts, n_peaks)
    plt.xlabel('Constraint Conflict')
    plt.ylabel('Number of Peaks')
    plt.title('Peak Count vs Constraint Conflict')
    
    plt.subplot(2, 2, 3)
    for i, r in enumerate(results):
        plt.hist(r['M_values'], bins=15, alpha=0.5, label=f'c={r["conflict"]}')
    plt.xlabel('Final M')
    plt.ylabel('Count')
    plt.title('M Distribution by Constraint Conflict')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('F:/hypergraph_bistability/figures/constraint_complexity/conflict_test.png', dpi=150)
    plt.close()
    
    return results


def test_rule_diversity():
    """测试规则多样性对稳态的影响"""
    print("\n" + "=" * 60)
    print("Test 2: Rule Diversity vs Stable States")
    print("=" * 60)
    
    results = []
    
    for diversity in [0.2, 0.4, 0.6, 0.8, 1.0, 1.5]:
        print(f"\nrule_diversity = {diversity}")
        
        M_values = []
        for seed in range(100):
            sys = VariableConstraintHypergraph(
                N=50, gamma=0.35, 
                rule_diversity=diversity,
                seed=seed
            )
            M = sys.run(steps=60)
            M_values.append(M)
        
        mean_M = np.mean(M_values)
        std_M = np.std(M_values)
        
        # 找双峰
        hist, bins = np.histogram(M_values, bins=20)
        peaks = []
        for i in range(1, len(hist) - 1):
            if hist[i] > 5 and hist[i] > hist[i-1] and hist[i] > hist[i+1]:
                peaks.append((bins[i] + bins[i+1]) / 2)
        
        results.append({
            'diversity': diversity,
            'mean_M': mean_M,
            'std_M': std_M,
            'n_peaks': len(peaks)
        })
        
        print(f"  M = {mean_M:.3f} ± {std_M:.3f}, peaks = {len(peaks)}")
    
    # 绘图
    plt.figure(figsize=(10, 6))
    diversities = [r['diversity'] for r in results]
    means = [r['mean_M'] for r in results]
    stds = [r['std_M'] for r in results]
    plt.errorbar(diversities, means, yerr=stds, marker='s', capsize=5)
    plt.xlabel('Rule Diversity')
    plt.ylabel('Final M')
    plt.title('Mean M vs Rule Diversity')
    plt.grid(True, alpha=0.3)
    plt.savefig('F:/hypergraph_bistability/figures/constraint_complexity/diversity_test.png', dpi=150)
    plt.close()
    
    return results


def test_combined_constraint():
    """测试组合约束复杂度"""
    print("\n" + "=" * 60)
    print("Test 3: Combined Constraint Complexity")
    print("=" * 60)
    
    # 定义约束复杂度 C = conflict + diversity
    results = []
    
    param_combinations = [
        (0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0),
        (0.2, 0.2), (0.4, 0.4), (0.6, 0.6), (0.8, 0.8), (1.0, 1.0),
    ]
    
    for conflict, diversity in param_combinations:
        C = conflict + diversity  # 约束复杂度
        
        M_values = []
        for seed in range(50):
            sys = VariableConstraintHypergraph(
                N=50, gamma=0.35,
                constraint_conflict=conflict,
                rule_diversity=diversity,
                seed=seed
            )
            M = sys.run(steps=60)
            M_values.append(M)
        
        mean_M = np.mean(M_values)
        std_M = np.std(M_values)
        
        results.append({
            'conflict': conflict,
            'diversity': diversity,
            'C': C,
            'mean_M': mean_M,
            'std_M': std_M
        })
        
        print(f"c={conflict}, d={diversity} → C={C:.1f}: M = {mean_M:.3f} ± {std_M:.3f}")
    
    # 绘图 3D
    fig = plt.figure(figsize=(12, 8))
    
    ax = fig.add_subplot(111, projection='3d')
    
    xs = [r['conflict'] for r in results]
    ys = [r['diversity'] for r in results]
    zs = [r['mean_M'] for r in results]
    
    ax.scatter(xs, ys, zs, c=zs, cmap='viridis', s=100)
    
    ax.set_xlabel('Constraint Conflict')
    ax.set_ylabel('Rule Diversity')
    ax.set_zlabel('Final M')
    ax.set_title('Constraint Complexity Surface')
    
    plt.tight_layout()
    plt.savefig('F:/hypergraph_bistability/figures/constraint_complexity/combined_test.png', dpi=150)
    plt.close()
    
    return results


def find_stable_state_jumps():
    """寻找稳态数量的跃迁"""
    print("\n" + "=" * 60)
    print("Test 4: Find Stable State Jumps")
    print("=" * 60)
    
    # 细粒度扫描
    results = []
    
    for gamma in np.arange(0.20, 0.60, 0.02):
        M_values = []
        for seed in range(50):
            sys = Hypergraph(N=50, gamma=gamma, seed=seed)
            M = sys.run(steps=60)
            M_values.append(M)
        
        mean_M = np.mean(M_values)
        std_M = np.std(M_values)
        
        results.append({
            'gamma': gamma,
            'mean_M': mean_M,
            'std_M': std_M
        })
    
    # 找跃迁点
    means = [r['mean_M'] for r in results]
    gammas = [r['gamma'] for r in results]
    
    # 计算导数
    jumps = []
    for i in range(1, len(means)):
        dM = means[i] - means[i-1]
        dg = gammas[i] - gammas[i-1]
        if abs(dM / dg) > 2:  # 显著跃迁
            jumps.append((gammas[i], means[i], dM/dg))
    
    print("\n发现跃迁点:")
    for g, m, d in jumps:
        print(f"  gamma ≈ {g:.2f}: M jump = {d:.2f}")
    
    # 绘图
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(gammas, means, marker='o')
    plt.xlabel('Gamma (Capacity Constraint)')
    plt.ylabel('Mean Final M')
    plt.title('M vs Gamma')
    plt.grid(True, alpha=0.3)
    
    # 标记跃迁点
    for g, m, d in jumps:
        plt.axvline(x=g, color='r', linestyle='--', alpha=0.5)
    
    plt.subplot(1, 2, 2)
    plt.plot(gammas, means, marker='o')
    plt.xlabel('Gamma')
    plt.ylabel('M')
    plt.title('M vs Gamma (Log Scale)')
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('F:/hypergraph_bistability/figures/constraint_complexity/jumps.png', dpi=150)
    plt.close()
    
    return results, jumps


def main():
    print("=" * 60)
    print("Constraint Complexity Test - Start")
    print("=" * 60)
    
    # 实验1: 约束冲突
    r1 = test_constraint_conflict()
    
    # 实验2: 规则多样性
    r2 = test_rule_diversity()
    
    # 实验3: 组合约束
    r3 = test_combined_constraint()
    
    # 实验4: 稳态跃迁
    r4, jumps = find_stable_state_jumps()
    
    print("\n" + "=" * 60)
    print("所有实验完成!")
    print("=" * 60)
    
    print("\n=== 结论 ===")
    print(f"1. 约束冲突影响: 改变M的分布")
    print(f"2. 规则多样性影响: 改变系统动力学")
    print(f"3. 发现跃迁点: {len(jumps)} 个")


if __name__ == "__main__":
    main()
