"""
Rule Perturbation Experiment
===========================
在 critical_dimension_analysis.py 基础上添加扰动参数
测试核心不变量是否"结构定律"
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import os
import json

os.makedirs('F:/hypergraph_bistability/figures', exist_ok=True)
os.makedirs('F:/hypergraph_bistability/results', exist_ok=True)


def run_experiment(N=50, gamma=0.35, state_dim=16, steps=300, seed=42, 
                  fusion_bias='linear', split_rule='size', noise=0.0, verbose=False):
    """运行实验，返回稳态 M"""
    
    random.seed(seed)
    np.random.seed(seed)
    
    K = int(gamma * N)
    
    class Hypergraph:
        def __init__(self, n_vertices):
            self.V = list(range(n_vertices))
            self.E = []
            self.s = {v: np.random.randn(state_dim) for v in self.V}
            self.conflicts = set()
            self.faction = {}
            self.K = K
            
            n_initial = max(4, n_vertices // 4)
            for _ in range(n_initial):
                size = random.randint(2, min(5, n_vertices // 3))
                e = frozenset(random.sample(self.V, size))
                self.E.append(e)
                self.faction[e] = random.randint(0, 2)
        
        def get_avg_distance(self):
            """优化版本：不每次都采样"""
            if not hasattr(self, '_cached_dist'):
                if len(self.V) < 2: 
                    self._cached_dist = 0.1
                    return 0.1
                samples = min(20, len(self.V) * (len(self.V) - 1) // 2)
                if samples <= 0: 
                    self._cached_dist = 0.1
                    return 0.1
                dist_sum = 0
                for _ in range(samples):
                    u, v = random.sample(self.V, 2)
                    dist_sum += np.linalg.norm(self.s[u] - self.s[v])
                self._cached_dist = max(dist_sum / samples, 0.01)
            return self._cached_dist
        
        def get_node_degree(self, v):
            return sum(1 for e in self.E if v in self.E and v in e)
        
        def get_influence(self, v):
            degree = self.get_node_degree(v)
            return 1.0 / (1.0 + 0.0 * degree)
        
        def enforce_resource_constraint(self):
            for v in self.V:
                degree = self.get_node_degree(v)
                if degree > self.K:
                    excess = degree - self.K
                    edges_with_v = [e for e in self.E if v in e]
                    edges_with_v.sort(key=lambda e: len(e), reverse=True)
                    for _ in range(min(excess, len(edges_with_v))):
                        if edges_with_v:
                            smallest = edges_with_v.pop()
                            if smallest in self.E:
                                self.E.remove(smallest)
                                if smallest in self.faction:
                                    del self.faction[smallest]
        
        def apply_rules(self):
            avg_dist = self.get_avg_distance()
            
            # ===== 规则1: 支化 (Growth) =====
            if random.random() < 0.35 and self.E:
                e = random.choice(self.E)
                nodes = list(e)
                weights = np.array([self.get_influence(v) for v in nodes])
                weights = weights / weights.sum()
                x = np.random.choice(nodes, p=weights)
                
                w = len(self.V)
                self.V.append(w)
                self.s[w] = self.s[x] + np.random.randn(state_dim) * 0.2
                new_e = frozenset([x, w])
                self.E.append(new_e)
                self.faction[new_e] = self.faction.get(e, random.randint(0, 2))
            
            # ===== 规则2: 融合 (Fusion) - 扰动版本 =====
            if random.random() < 0.3 and len(self.E) >= 2:
                e1 = random.choice(self.E)
                e2 = random.choice(self.E)
                
                if e1 != e2 and len(e1 & e2) >= 1:
                    faction1 = self.faction.get(e1, 0)
                    faction2 = self.faction.get(e2, 0)
                    faction_penalty = 0.5 if faction1 != faction2 else 0.0
                    
                    # 计算加权状态中心
                    weights1 = np.array([self.get_influence(v) for v in e1])
                    weights1 = weights1 / weights1.sum()
                    c1 = np.average([self.s[u] for u in e1], axis=0, weights=weights1)
                    
                    weights2 = np.array([self.get_influence(v) for v in e2])
                    weights2 = weights2 / weights2.sum()
                    c2 = np.average([self.s[u] for u in e2], axis=0, weights=weights2)
                    
                    distance = np.linalg.norm(c1 - c2)
                    
                    effective_threshold = 0.8 * avg_dist - faction_penalty
                    
                    # ===== 扰动：修改融合概率 =====
                    if fusion_bias == 'linear':
                        fusion_prob = distance / (avg_dist * 2)
                    elif fusion_bias == 'square':
                        # 强化强者：距离权重平方
                        fusion_prob = (distance / (avg_dist * 2)) ** 2
                    elif fusion_bias == 'sqrt':
                        # 弱化强者
                        fusion_prob = np.sqrt(distance / (avg_dist * 2))
                    else:
                        fusion_prob = distance / (avg_dist * 2)
                    
                    # 加噪声
                    if noise > 0 and random.random() < noise:
                        fusion_prob = 1.0
                    
                    if distance < effective_threshold or random.random() < fusion_prob:
                        new_e = frozenset(e1 | e2)
                        if new_e not in self.E:
                            self.E.append(new_e)
                            self.faction[new_e] = faction1
                            self.E.remove(e1)
                            self.E.remove(e2)
                            if e1 in self.faction:
                                del self.faction[e1]
                            if e2 in self.faction:
                                del self.faction[e2]
            
            # ===== 规则3: 分裂 (Split) - 扰动版本 =====
            if random.random() < 0.15 and len(self.E) > 1:
                # ===== 扰动：修改分裂源选择 =====
                if split_rule == 'size':
                    # 原版：按大小加权
                    sizes = [len(e) for e in self.E if len(e) > 1]
                    if sizes:
                        weights = np.array(sizes, dtype=float)
                        weights = weights / weights.sum()
                        idx = np.random.choice(len([e for e in self.E if len(e) > 1]), p=weights)
                        e = [e for e in self.E if len(e) > 1][idx]
                    else:
                        e = None
                elif split_rule == 'random':
                    # 随机选择
                    large_edges = [e for e in self.E if len(e) > 1]
                    e = random.choice(large_edges) if large_edges else None
                
                if e is not None and len(e) > 2:
                    nodes = list(e)
                    random.shuffle(nodes)
                    split_point = len(nodes) // 2
                    
                    # 加噪声
                    if noise > 0 and random.random() < noise:
                        split_point = 1  # 强制最小分裂
                    
                    new_e1 = frozenset(nodes[:split_point])
                    new_e2 = frozenset(nodes[split_point:])
                    
                    faction = self.faction.get(e, random.randint(0, 2))
                    self.faction[new_e1] = faction
                    self.faction[new_e2] = faction
                    
                    self.E.remove(e)
                    self.E.extend([new_e1, new_e2])
                    if e in self.faction:
                        del self.faction[e]
            
            # ===== 规则4: 修剪 (Prune) =====
            if random.random() < 0.05:
                isolated = [v for v in self.V if not any(v in e for e in self.E)]
                for v in isolated[:1]:
                    if v in self.s:
                        del self.s[v]
                    if v in self.V:
                        self.V.remove(v)
            
            self.enforce_resource_constraint()
            
            # 清除缓存
            if hasattr(self, '_cached_dist'):
                del self._cached_dist
        
        def get_M(self):
            if not self.E:
                return 0
            # 构建连通分量
            if not self.V:
                return 0
            visited = set()
            components = []
            for v in self.V:
                if v not in visited:
                    stack = [v]
                    comp = set()
                    while stack:
                        n = stack.pop()
                        if n in visited:
                            continue
                        visited.add(n)
                        comp.add(n)
                        for e in self.E:
                            if n in e:
                                for u in e:
                                    if u not in visited:
                                        stack.append(u)
                    components.append(comp)
            
            if not components:
                return 0
            return max(len(c) for c in components) / len(self.V)
    
    # 运行
    g = Hypergraph(N)
    for _ in range(steps):
        g.apply_rules()
    
    return g.get_M()


def run_batch(N=50, gamma=0.35, state_dim=16, steps=300, n_runs=20,
              fusion_bias='linear', split_rule='size', noise=0.0):
    """批量运行"""
    results = []
    for i in range(n_runs):
        M = run_experiment(N=N, gamma=gamma, state_dim=state_dim, 
                          steps=steps, seed=i*100+42,
                          fusion_bias=fusion_bias, split_rule=split_rule, 
                          noise=noise)
        results.append(M)
    return np.mean(results), np.std(results), results


def main():
    print("=" * 60)
    print("Rule Perturbation Experiment")
    print("=" * 60)
    
    N = 40  # 减小 N
    gamma = 0.35
    state_dim = 16
    steps = 150
    n_runs = 8  # 减少运行次数
    
    # 实验配置 - 扩展版
    experiments = [
        # 基准组
        {'name': 'baseline', 'fusion_bias': 'linear', 'split_rule': 'size', 'noise': 0.0},
        
        # Fusion bias 扰动
        {'name': 'fusion_square', 'fusion_bias': 'square', 'split_rule': 'size', 'noise': 0.0},
        {'name': 'fusion_sqrt', 'fusion_bias': 'sqrt', 'split_rule': 'size', 'noise': 0.0},
        
        # Split 机制扰动
        {'name': 'split_random', 'fusion_bias': 'linear', 'split_rule': 'random', 'noise': 0.0},
        
        # 噪声扰动
        {'name': 'noise_low', 'fusion_bias': 'linear', 'split_rule': 'size', 'noise': 0.05},
        {'name': 'noise_high', 'fusion_bias': 'linear', 'split_rule': 'size', 'noise': 0.1},
    ]
    
    results = {}
    
    for exp in experiments:
        print(f"\nRunning: {exp['name']}...")
        mean_M, std_M, all_M = run_batch(
            N=N, gamma=gamma, state_dim=state_dim, steps=steps, n_runs=n_runs,
            fusion_bias=exp['fusion_bias'],
            split_rule=exp['split_rule'],
            noise=exp['noise']
        )
        results[exp['name']] = {
            'mean': mean_M,
            'std': std_M,
            'all': all_M
        }
        print(f"  M* = {mean_M:.3f} +/- {std_M:.3f}")
    
    # 打印对比表
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"{'Experiment':<20} | {'M*':>8} | {'Std':>6} | {'Delta':>10}")
    print("-" * 60)
    
    baseline = results['baseline']['mean']
    for name, data in results.items():
        delta = data['mean'] - baseline
        print(f"{name:<20} | {data['mean']:>8.3f} | {data['std']:>6.3f} | {delta:>+10.3f}")
    
    # 分析
    print("\n" + "=" * 60)
    print("Analysis")
    print("=" * 60)
    
    # 检查 M* 是否接近 0.45
    print("\n[M* Check - close to 0.45?]:")
    for name, data in results.items():
        M = data['mean']
        if 0.35 <= M <= 0.55:
            status = "[OK]"
        else:
            status = "[FAIL]"
        print(f"  {name}: M* = {M:.3f} {status}")
    
    # 绘制对比图
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    
    for idx, (name, data) in enumerate(results.items()):
        ax = axes[idx]
        ax.hist(data['all'], bins=15, alpha=0.7, edgecolor='black')
        ax.axvline(data['mean'], color='red', linestyle='--', label=f'Mean={data["mean"]:.3f}')
        ax.axvline(0.45, color='green', linestyle=':', label='Target=0.45')
        ax.set_xlabel('M')
        ax.set_ylabel('Count')
        ax.set_title(name)
        ax.legend()
        ax.set_xlim(0, 1)
    
    plt.tight_layout()
    plt.savefig('F:/hypergraph_bistability/figures/rule_perturbation.png', dpi=150)
    print(f"\n[OK] Figure saved: figures/rule_perturbation.png")
    
    # 保存结果
    with open('F:/hypergraph_bistability/results/rule_perturbation.json', 'w') as f:
        json.dump({k: {'mean': v['mean'], 'std': v['std']} for k, v in results.items()}, f, indent=2)
    print(f"[OK] Results saved: results/rule_perturbation.json")


if __name__ == '__main__':
    main()
