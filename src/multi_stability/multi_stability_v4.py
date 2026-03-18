"""
多稳态超图 v4 - 关键突破
======================
核心洞察：原项目双稳态来自 k vs k_c 的临界
多群体多稳态：让不同群体有不同的 gamma 值
- group1: gamma < k_c → LOW态
- group2: gamma > k_c → HIGH态
- group3: gamma ≈ k_c → 临界态
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import os

os.makedirs('F:/hypergraph_bistability/figures/multi_stability', exist_ok=True)


class MultiGammaHypergraph:
    """每个群体有不同gamma值的超图"""
    
    def __init__(self, N=50, gammas=None, seed=42):
        """
        gammas: list of gamma values for each group
        """
        random.seed(seed)
        np.random.seed(seed)
        
        if gammas is None:
            gammas = [0.3, 0.4, 0.35]  # 默认3个群体
        
        self.N = N
        self.n_groups = len(gammas)
        self.gammas = gammas  # 每个群体独立的gamma
        self.Ks = [int(g * N) for g in gammas]
        
        # 节点分组
        self.group_assignments = {}
        nodes_per_group = N // self.n_groups
        for i in range(self.n_groups):
            start = i * nodes_per_group
            end = start + nodes_per_group if i < self.n_groups - 1 else N
            for v in range(start, end):
                self.group_assignments[v] = i
        
        # 初始化
        self.V = list(range(N))
        self.E = []
        
        # 每群体独立初始化
        for g in range(self.n_groups):
            group_nodes = [v for v in self.V if self.group_assignments[v] == g]
            # 初始少量超边
            for _ in range(2):
                if len(group_nodes) >= 2:
                    e = frozenset(random.sample(group_nodes, 2))
                    self.E.append(e)
    
    def get_node_degree(self, v):
        return sum(1 for e in self.E if v in e)
    
    def get_group_M(self, g):
        """群体g的序参量"""
        group_nodes = [v for v in self.V if self.group_assignments[v] == g]
        if not group_nodes:
            return 0.0
        degrees = [self.get_node_degree(v) for v in group_nodes]
        # 归一化到群体的K值
        return np.mean(degrees) / max(self.Ks[g], 1)
    
    def apply_rules(self):
        """每群体按自己的gamma运行"""
        
        for g in range(self.n_groups):
            group_nodes = [v for v in self.V if self.group_assignments[v] == g]
            K_g = self.Ks[g]
            
            # 生长 (概率随K增大)
            if random.random() < 0.3 * K_g / 10:
                if group_nodes:
                    v = random.choice(group_nodes)
                    w = len(self.V)
                    self.V.append(w)
                    self.group_assignments[w] = g
                    self.E.append(frozenset([v, w]))
            
            # 融合 (同群体优先)
            if random.random() < 0.25:
                if len(self.E) >= 2:
                    e1, e2 = random.sample(self.E, 2)
                    if len(e1 & e2) >= 1:
                        g1 = self.group_assignments[list(e1)[0]]
                        g2 = self.group_assignments[list(e2)[0]]
                        
                        if g1 == g2 == g:
                            new_e = frozenset(e1 | e2)
                            if new_e not in self.E and len(new_e) <= K_g:
                                self.E.extend([new_e])
                                self.E.remove(e1)
                                self.E.remove(e2)
            
            # 分裂
            if random.random() < 0.15:
                group_edges = [e for e in self.E if self.group_assignments[list(e)[0]] == g]
                large = [e for e in group_edges if len(e) > 2]
                if large:
                    e = random.choice(large)
                    nodes = list(e)
                    if len(nodes) >= 2:
                        split = len(nodes) // 2
                        self.E.append(frozenset(nodes[:split]))
                        self.E.append(frozenset(nodes[split:]))
                        self.E.remove(e)
            
            # 删除
            if random.random() < 0.2:
                group_edges = [e for e in self.E if self.group_assignments[list(e)[0]] == g]
                if len(group_edges) > 1:
                    self.E.remove(random.choice(group_edges))
    
    def run(self, steps=80):
        """运行"""
        history = []
        for _ in range(steps):
            self.apply_rules()
            M = [self.get_group_M(g) for g in range(self.n_groups)]
            history.append(M)
        return np.array(history)


def test_key_insight():
    """测试关键洞察：不同gamma → 不同稳态"""
    
    print("=" * 60)
    print("测试：不同gamma → 不同稳态")
    print("=" * 60)
    
    # 关键测试：固定两个群体的gamma在临界点两侧
    test_configs = [
        # (gamma列表, 预期)
        ([0.25, 0.45], "LOW, HIGH"),      # 两侧
        ([0.30, 0.40], "LOW, HIGH"),      # 临界附近
        ([0.20, 0.35, 0.50], "LOW, MID, HIGH"),  # 三档
        ([0.25, 0.35, 0.45], "LOW, MID, HIGH"),
        ([0.30, 0.30, 0.30], "ALL SAME"),
    ]
    
    results = {}
    
    for gammas, expected in test_configs:
        print(f"\ngammas: {gammas} (预期: {expected})")
        
        all_M = []
        for seed in range(20):
            sys = MultiGammaHypergraph(N=40, gammas=gammas, seed=seed)
            history = sys.run(steps=60)
            final_M = history[-1]
            all_M.append(final_M)
        
        all_M = np.array(all_M)
        
        # 统计
        for i, g in enumerate(gammas):
            vals = all_M[:, i]
            print(f"  群体{i} (γ={g}): mean_M = {np.mean(vals):.2f}, std = {np.std(vals):.2f}")
        
        # 计算差异
        means = [np.mean(all_M[:, i]) for i in range(len(gammas))]
        spread = max(means) - min(means)
        print(f"  稳态差异: {spread:.2f}")
        
        results[str(gammas)] = {
            'means': means,
            'spread': spread,
            'expected': expected
        }
    
    # 绘图
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(test_configs))
    width = 0.25
    
    for group_idx in range(3):
        means = [results[str(g)]['means'][group_idx] if group_idx < len(results[str(g)]['means']) else 0 
                 for g in test_configs]
        ax.bar(x + group_idx * width, means, width, label=f'Group {group_idx}')
    
    ax.set_xlabel('Gamma Configuration')
    ax.set_ylabel('Final Order Parameter M')
    ax.set_title('Multi-group with Different Gammas')
    ax.set_xticks(x + width)
    ax.set_xticklabels([str(g[0]) for g in test_configs], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('F:/hypergraph_bistability/figures/multi_stability/v4_key_insight.png', dpi=150)
    plt.close()
    
    return results


def detailed_scan():
    """更细致的扫描"""
    
    print("\n" + "=" * 60)
    print("细致扫描：gamma vs M")
    print("=" * 60)
    
    gamma_range = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    
    results = []
    for gamma in gamma_range:
        M_values = []
        for seed in range(20):
            sys = MultiGammaHypergraph(N=40, gammas=[gamma], seed=seed)
            history = sys.run(steps=60)
            final_M = history[-1][0]  # 只有一个群体
            M_values.append(final_M)
        
        mean_M = np.mean(M_values)
        std_M = np.std(M_values)
        results.append((gamma, mean_M, std_M))
        print(f"γ={gamma:.2f}: M = {mean_M:.2f} ± {std_M:.2f}")
    
    # 绘图
    gammas = [r[0] for r in results]
    means = [r[1] for r in results]
    stds = [r[2] for r in results]
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(gammas, means, yerr=stds, marker='o', capsize=5)
    plt.axvline(x=0.35, color='r', linestyle='--', label='k_c ≈ 0.35')
    plt.xlabel('Gamma (Capacity Constraint)')
    plt.ylabel('Order Parameter M')
    plt.title('Single Group: Gamma vs M')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('F:/hypergraph_bistability/figures/multi_stability/v4_gamma_scan.png', dpi=150)
    plt.close()


if __name__ == "__main__":
    test_key_insight()
    detailed_scan()
    print("\n完成!")
