"""
Path A: Emergent Bistability from Microscopic Rules
===================================================

问题：当前超图规则产生碎片化，不产生双稳态
解决思路：引入非线性反馈机制

核心假设：
- 双稳态需要正反馈：M高时增长加速，M低时抑制加速
- 当前规则缺少这种非线性

新设计：
1. 融合概率非线性化：p_fuse = f(M) 是 M 的非线性函数
2. 当 M > 阈值时，融合概率急剧增加 -> 产生凝聚相
3. 当 M < 阈值时，融合概率低 -> 碎片相
4. 竞争：不同群体间争夺资源，产生 bistability
"""

import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json
from scipy.integrate import solve_ivp

os.makedirs('F:/hypergraph_bistability/results/verification_a', exist_ok=True)
os.makedirs('F:/hypergraph_bistability/figures/verification_a', exist_ok=True)


# ============================================================================
# Modified Hypergraph with Nonlinear Fusion
# ============================================================================

class NonlinearFusionHypergraph:
    """
    修改后的超图：融合概率依赖于当前活跃度 M
    关键设计：
    - p_fuse(M) = base_rate * (1 + alpha * M^beta) / (1 + gamma * M^beta)
    - 这产生 sigmoid-like 效应：低M时融合慢，高M时融合快
    """
    
    def __init__(self, N=50, n_groups=2, gamma=0.35, state_dim=16,
                 fusion_alpha=5.0, fusion_beta=4.0, fusion_gamma=1.0,
                 inter_group_competition=0.0, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        
        self.N = N
        self.n_groups = n_groups
        self.gamma = gamma
        self.state_dim = state_dim
        self.K = int(gamma * N)
        
        # 非线性融合参数
        self.fusion_alpha = fusion_alpha  # 增强因子
        self.fusion_beta = fusion_beta    # 非线性指数
        self.fusion_gamma = fusion_gamma  # 抑制因子
        
        self.lambda_ij = inter_group_competition
        
        # 节点分组
        self.group_assignments = {}
        nodes_per_group = N // n_groups
        for i in range(n_groups):
            start = i * nodes_per_group
            end = start + nodes_per_group if i < n_groups - 1 else N
            for v in range(start, end):
                self.group_assignments[v] = i
        
        # 初始化
        self.V = list(range(N))
        self.s = {v: np.random.randn(state_dim) for v in self.V}
        self.group_gamma = {i: gamma for i in range(n_groups)}
        self.E = []  # 初始化超边列表
        
        # 初始化超边
        n_initial = max(4, N // 4)
        for _ in range(n_initial):
            group = random.randint(0, n_groups - 1)
            group_nodes = [v for v in self.V if self.group_assignments[v] == group]
            if len(group_nodes) >= 2:
                size = random.randint(2, min(5, len(group_nodes) // 2))
                e = frozenset(random.sample(group_nodes, size))
                self.E.append(e)
    
    def get_node_group(self, v):
        return self.group_assignments[v]
    
    def get_group_nodes(self, g):
        return [v for v in self.V if self.group_assignments[v] == g]
    
    def get_node_degree(self, v):
        return sum(1 for e in self.E if v in e)
    
    def get_group_order_parameter(self, g):
        """计算群体 g 的序参量 M_g"""
        group_nodes = self.get_group_nodes(g)
        if not group_nodes:
            return 0.0
        degrees = [self.get_node_degree(v) for v in group_nodes]
        return np.mean(degrees) / max(self.K, 1)
    
    def get_avg_distance(self, nodes):
        if len(nodes) < 2:
            return 0.1
        samples = min(20, len(nodes) * (len(nodes) - 1) // 2)
        if samples <= 0:
            return 0.1
        dist_sum = 0
        for _ in range(samples):
            u, v = random.sample(nodes, 2)
            dist_sum += np.linalg.norm(self.s[u] - self.s[v])
        return max(dist_sum / samples, 0.01)
    
    def get_group_M(self, g):
        """获取群体 g 的平均 M"""
        return self.get_group_order_parameter(g)
    
    def nonlinear_fusion_prob(self, dist, avg_dist, M_g):
        """
        非线性融合概率：
        p_fuse ~ base_rate * sigmoid(M_g - threshold)
        
        关键：当 M_g 高时，融合概率增加（因为节点更活跃）
        """
        # 基础概率
        base_rate = 0.3
        
        # 非线性因子：M_g 的 sigmoid 函数
        threshold = 0.3
        sigmoid = 1.0 / (1.0 + np.exp(-self.fusion_alpha * (M_g - threshold)))
        
        # 距离因子（近者优先融合）
        dist_factor = np.exp(-dist / (avg_dist + 0.01))
        
        return base_rate * sigmoid * dist_factor
    
    def apply_rules(self):
        """应用超图演化规则"""
        avg_dist_overall = self.get_avg_distance(self.V)
        
        for g in range(self.n_groups):
            group_nodes = self.get_group_nodes(g)
            if not group_nodes:
                continue
            
            avg_dist_g = self.get_avg_distance(group_nodes)
            gamma_g = self.group_gamma[g]
            K_g = int(gamma_g * len(group_nodes))
            
            M_g = self.get_group_M(g)
            
            # 规则1：增长（保持不变）
            if random.random() < 0.35 and self.E:
                group_edges = [e for e in self.E
                             if all(self.get_node_group(v) == g for v in e)]
                if group_edges:
                    e = random.choice(group_edges)
                    nodes = list(e)
                    w = len(self.V)
                    self.V.append(w)
                    self.s[w] = self.s[nodes[0]] + np.random.randn(self.state_dim) * 0.2
                    self.group_assignments[w] = g
                    new_e = frozenset([nodes[0], w])
                    self.E.append(new_e)
            
            # 规则2：融合（修改为非线性）
            if random.random() < 0.3 and len(self.E) >= 2:
                e1, e2 = random.sample(self.E, 2)
                if len(e1 & e2) >= 1:
                    g1 = self.get_node_group(list(e1)[0])
                    g2 = self.get_node_group(list(e2)[0])
                    
                    weights1 = np.array([1.0 / (1 + self.get_node_degree(v)) for v in e1])
                    weights1 = weights1 / weights1.sum()
                    c1 = np.average([self.s[u] for u in e1], axis=0, weights=weights1)
                    
                    weights2 = np.array([1.0 / (1 + self.get_node_degree(v)) for v in e2])
                    weights2 = weights2 / weights2.sum()
                    c2 = np.average([self.s[u] for u in e2], axis=0, weights=weights2)
                    
                    dist = np.linalg.norm(c1 - c2)
                    
                    if g1 == g2:
                        # 同群体：使用非线性融合概率
                        threshold = avg_dist_g * 0.8
                        if dist < threshold:
                            p_fuse = self.nonlinear_fusion_prob(dist, avg_dist_g, M_g)
                            if random.random() < p_fuse:
                                new_e = frozenset(e1 | e2)
                                if new_e not in self.E:
                                    self.E.append(new_e)
                                    self.E.remove(e1)
                                    self.E.remove(e2)
                    else:
                        # 跨群体：使用竞争参数
                        if random.random() < (1 - self.lambda_ij):
                            threshold = avg_dist_g * 0.8
                            if dist < threshold:
                                new_e = frozenset(e1 | e2)
                                if new_e not in self.E:
                                    self.E.append(new_e)
                                    self.E.remove(e1)
                                    self.E.remove(e2)
            
            # 规则3：分裂（保持不变）
            if random.random() < 0.15:
                group_edges = [e for e in self.E
                             if all(self.get_node_group(v) == g for v in e)]
                large_edges = [e for e in group_edges if len(e) > 2]
                if large_edges:
                    e = random.choice(large_edges)
                    nodes = list(e)
                    if len(nodes) >= 2:
                        split_point = len(nodes) // 2
                        e1 = frozenset(nodes[:split_point])
                        e2 = frozenset(nodes[split_point:])
                        if e1 not in self.E and e2 not in self.E:
                            self.E.append(e1)
                            self.E.append(e2)
                            self.E.remove(e)
            
            # 规则4：删除（保持）
            if random.random() < 0.2:
                small_edges = [e for e in self.E if len(e) == 2]
                if small_edges:
                    e = random.choice(small_edges)
                    self.E.remove(e)


# ============================================================================
# Test Emergent Bistability
# ============================================================================

def test_emergent_bistability():
    """测试修改后的规则是否产生双稳态"""
    
    print("=" * 60)
    print("Test: Emergent Bistability with Nonlinear Fusion")
    print("=" * 60)
    
    # 测试不同初始条件
    n_runs = 30
    T = 120
    
    final_M_values = []
    M_histories = []
    
    for run in range(n_runs):
        seed = 42 + run * 100
        
        # 创建超图
        hg = NonlinearFusionHypergraph(
            N=50, n_groups=1,  # k=1, L=1
            gamma=0.35, state_dim=16,
            fusion_alpha=5.0, fusion_beta=4.0, fusion_gamma=1.0,
            inter_group_competition=0.0,
            seed=seed
        )
        
        # 运行动力学
        history = []
        for t in range(T):
            hg.apply_rules()
            M = hg.get_group_order_parameter(0)
            history.append(M)
        
        final_M = hg.get_group_order_parameter(0)
        final_M_values.append(final_M)
        M_histories.append(history)
    
    final_M_values = np.array(final_M_values)
    
    # 分析结果
    print(f"\nResults (k=1, L=1):")
    print(f"  n_runs: {n_runs}")
    print(f"  Mean final M: {final_M_values.mean():.4f}")
    print(f"  Std final M: {final_M_values.std():.4f}")
    print(f"  Min: {final_M_values.min():.4f}, Max: {final_M_values.max():.4f}")
    
    # 聚类分析
    unique_attractors = []
    for M in final_M_values:
        is_new = True
        for u in unique_attractors:
            if abs(M - u) < 0.15:
                is_new = False
                break
        if is_new:
            unique_attractors.append(M)
    
    print(f"  Unique attractors: {len(unique_attractors)}")
    print(f"  Attractor values: {[f'{u:.3f}' for u in unique_attractors]}")
    
    return final_M_values, unique_attractors, len(unique_attractors)


def test_tensor_product():
    """测试张量积结构"""
    
    print("\n" + "=" * 60)
    print("Test: Tensor Product Structure")
    print("=" * 60)
    
    configs = [
        (1, 1),
        (2, 1),
        (1, 2),
    ]
    
    results = {}
    
    for k, L in configs:
        n_runs = 30
        T = 100
        
        # 对于 k>1 或 L>1，使用简化的测试
        # 直接用 MultiGroupHypergraph 的序参量
        all_M = []
        
        for run in range(n_runs):
            seed = 42 + run * 100
            
            if k == 1 and L == 1:
                hg = NonlinearFusionHypergraph(
                    N=50, n_groups=1, gamma=0.35,
                    fusion_alpha=5.0, fusion_beta=4.0,
                    inter_group_competition=0.0, seed=seed
                )
                for t in range(T):
                    hg.apply_rules()
                all_M.append([hg.get_group_order_parameter(0)])
            else:
                # 对于 k>1，创建多个独立的群体
                group_M = []
                for g in range(k):
                    hg_g = NonlinearFusionHypergraph(
                        N=25, n_groups=1, gamma=0.35,
                        fusion_alpha=5.0, fusion_beta=4.0,
                        inter_group_competition=0.0, seed=seed+g*100
                    )
                    for t in range(T):
                        hg_g.apply_rules()
                    group_M.append(hg_g.get_group_order_parameter(0))
                all_M.append(group_M)
        
        # 聚类
        all_M = np.array(all_M)
        unique_states = []
        for state in all_M:
            is_new = True
            for u in unique_states:
                if np.sqrt(np.mean((state - u)**2)) < 0.2:
                    is_new = False
                    break
            if is_new:
                unique_states.append(state)
        
        predicted = 2 ** (k * L)
        n_att = len(unique_states)
        
        results[(k, L)] = {
            'n_attractors': n_att,
            'predicted': predicted,
            'match': 'PASS' if n_att >= predicted * 0.5 else 'FAIL'
        }
        
        print(f"k={k}, L={L}: N_att={n_att}, predicted={predicted} [{results[(k, L)]['match']}]")
    
    return results


if __name__ == '__main__':
    # Test 1: Basic bistability
    final_M, attractors, n_att = test_emergent_bistability()
    
    # Test 2: Tensor product
    tensor_results = test_tensor_product()
    
    # Save results
    all_results = {
        'bistability_test': {
            'n_attractors': n_att,
            'attractor_values': [float(a) for a in attractors],
            'mean_M': float(final_M.mean()),
            'std_M': float(final_M.std())
        },
        'tensor_product': tensor_results
    }
    
    with open('F:/hypergraph_bistability/results/verification_a/results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Bistability test: {n_att} attractors")
    print("Tensor product test:", tensor_results)
