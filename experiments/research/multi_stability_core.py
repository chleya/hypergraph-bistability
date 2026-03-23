"""
多稳态超图动力学系统
==================
从双稳态扩展到多稳态：
- Step 1: 多群体竞争
- Step 2: 多层超图
- Step 3: 多稳相图

核心思想：
- 原项目：单群体竞争 → 双稳态
- 扩展后：多群体竞争 + 多层超图 → 多稳态
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import os
import json
from collections import defaultdict

os.makedirs('F:/hypergraph_bistability/figures/multi_stability', exist_ok=True)
os.makedirs('F:/hypergraph_bistability/src/multi_stability', exist_ok=True)

# ============================================================================
# 第一步：多群体竞争超图
# ============================================================================

class MultiGroupHypergraph:
    """多群体竞争超图动力学系统"""
    
    def __init__(self, N=50, n_groups=3, gamma=0.35, state_dim=16, 
                 inter_group_competition=0.5, seed=42):
        """
        参数:
        - N: 总节点数
        - n_groups: 群体数量 (k≥3 对应三稳态+)
        - gamma: 容量约束 (原项目参数)
        - state_dim: 状态维度
        - inter_group_competition: 跨群体竞争强度 λ (0~1)
        """
        random.seed(seed)
        np.random.seed(seed)
        
        self.N = N
        self.n_groups = n_groups
        self.gamma = gamma
        self.state_dim = state_dim
        self.K = int(gamma * N)
        self.lambda_ij = inter_group_competition  # 跨群体竞争强度
        
        # 节点分组
        self.group_assignments = {}
        nodes_per_group = N // n_groups
        for i in range(n_groups):
            start = i * nodes_per_group
            end = start + nodes_per_group if i < n_groups - 1 else N
            for v in range(start, end):
                self.group_assignments[v] = i
        
        # 初始化节点状态
        self.V = list(range(N))
        self.E = []
        self.s = {v: np.random.randn(state_dim) for v in self.V}
        
        # 每群体独立的容量约束
        self.group_gamma = {i: gamma for i in range(n_groups)}
        
        # 初始化超边 (每群体独立初始化)
        n_initial = max(4, N // 4)
        for _ in range(n_initial):
            # 按群体比例生成超边
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
        """计算群体g的序参量 M_g (该群体的平均交互强度)"""
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
    
    def apply_rules(self):
        """应用四条规则 (扩展为多群体版本)"""
        avg_dist_overall = self.get_avg_distance(self.V)
        
        for g in range(self.n_groups):
            group_nodes = self.get_group_nodes(g)
            if not group_nodes:
                continue
                
            avg_dist_g = self.get_avg_distance(group_nodes)
            gamma_g = self.group_gamma[g]
            K_g = int(gamma_g * len(group_nodes))
            
            # 规则1: 生长 (群体内)
            if random.random() < 0.35 and self.E:
                # 选择同群体内的超边
                group_edges = [e for e in self.E if all(self.get_node_group(v) == g for v in e)]
                if group_edges:
                    e = random.choice(group_edges)
                    nodes = list(e)
                    w = len(self.V)
                    self.V.append(w)
                    self.s[w] = self.s[nodes[0]] + np.random.randn(self.state_dim) * 0.2
                    self.group_assignments[w] = g
                    new_e = frozenset([nodes[0], w])
                    self.E.append(new_e)
            
            # 规则2: 融合 (群体内 + 跨群体)
            if random.random() < 0.3 and len(self.E) >= 2:
                e1, e2 = random.sample(self.E, 2)
                if len(e1 & e2) >= 1:
                    g1 = self.get_node_group(list(e1)[0])
                    g2 = self.get_node_group(list(e2)[0])
                    
                    # 计算融合中心
                    weights1 = np.array([1.0 / (1 + self.get_node_degree(v)) for v in e1])
                    weights1 = weights1 / weights1.sum()
                    c1 = np.average([self.s[u] for u in e1], axis=0, weights=weights1)
                    
                    weights2 = np.array([1.0 / (1 + self.get_node_degree(v)) for v in e2])
                    weights2 = weights2 / weights2.sum()
                    c2 = np.average([self.s[u] for u in e2], axis=0, weights=weights2)
                    
                    dist = np.linalg.norm(c1 - c2)
                    
                    # 群体内融合 vs 跨群体融合
                    if g1 == g2:
                        # 群体内融合 (更容易)
                        threshold = avg_dist_g * 0.8
                        if dist < threshold:
                            new_e = frozenset(e1 | e2)
                            if new_e not in self.E:
                                self.E.append(new_e)
                                self.E.remove(e1)
                                self.E.remove(e2)
                    else:
                        # 跨群体融合 (受 lambda 控制)
                        if random.random() < (1 - self.lambda_ij):
                            threshold = avg_dist_g * 0.8
                            if dist < threshold:
                                new_e = frozenset(e1 | e2)
                                if new_e not in self.E:
                                    self.E.append(new_e)
                                    self.E.remove(e1)
                                    self.E.remove(e2)
            
            # 规则3: 分裂 (群体内)
            if random.random() < 0.15:
                group_edges = [e for e in self.E if all(self.get_node_group(v) == g for v in e)]
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
            
            # 规则4: 删除 (保持容量约束)
            if random.random() < 0.2:
                small_edges = [e for e in self.E if len(e) == 2]
                if small_edges:
                    e = random.choice(small_edges)
                    self.E.remove(e)
    
    def run_dynamics(self, steps=80):
        """运行动力学演化"""
        M_history = {g: [] for g in range(self.n_groups)}
        
        for t in range(steps):
            self.apply_rules()
            for g in range(self.n_groups):
                M_history[g].append(self.get_group_order_parameter(g))
        
        return M_history
    
    def get_stable_states(self):
        """获取所有群体的稳定态序参量"""
        final_M = {}
        for g in range(self.n_groups):
            M_g = self.get_group_order_parameter(g)
            final_M[g] = M_g
        return final_M


# ============================================================================
# 第二步：多层超图 (扩展多群体到多层)
# ============================================================================

class MultiLayerMultiGroupHypergraph:
    """多层超图 + 多群体竞争"""
    
    def __init__(self, N=50, n_groups=3, n_layers=2, gamma=0.35, 
                 state_dim=16, inter_group_competition=0.5, 
                 inter_layer_coupling=0.2, seed=42):
        """
        参数:
        - n_layers: 超图层数 L≥2
        - inter_layer_coupling: 跨层交互权重 μ (-0.5~0.5)
        """
        random.seed(seed)
        np.random.seed(seed)
        
        self.N = N
        self.n_groups = n_groups
        self.n_layers = n_layers
        self.gamma = gamma
        self.state_dim = state_dim
        self.lambda_ij = inter_group_competition
        self.mu = inter_layer_coupling  # 跨层耦合
        
        # 每层是一个独立的超图，但共享节点
        self.layers = []
        for l in range(n_layers):
            layer = MultiGroupHypergraph(
                N=N, n_groups=n_groups, gamma=gamma, 
                state_dim=state_dim, 
                inter_group_competition=inter_group_competition,
                seed=seed + l * 100
            )
            self.layers.append(layer)
    
    def apply_cross_layer_rules(self):
        """跨层交互规则"""
        if self.mu == 0:
            return
        
        for l1 in range(self.n_layers):
            for l2 in range(self.n_layers):
                if l1 >= l2:
                    continue
                
                # 跨层信息传递
                for v in self.layers[l1].V:
                    if v in self.layers[l2].V:
                        # 状态混合
                        s1 = self.layers[l1].s[v]
                        s2 = self.layers[l2].s[v]
                        
                        if self.mu > 0:
                            # 促进: 趋向平均
                            self.layers[l1].s[v] = (1 - self.mu) * s1 + self.mu * s2
                            self.layers[l2].s[v] = (1 - self.mu) * s2 + self.mu * s1
                        else:
                            # 抑制: 趋向正交
                            self.layers[l1].s[v] = s1 - self.mu * s2
                            self.layers[l2].s[v] = s2 - self.mu * s1
    
    def run_dynamics(self, steps=80):
        """运行动力学演化"""
        M_history = {l: {g: [] for g in range(self.n_groups)} for l in range(self.n_layers)}
        
        for t in range(steps):
            # 每层独立演化
            for layer in self.layers:
                layer.apply_rules()
            
            # 跨层交互
            self.apply_cross_layer_rules()
            
            # 记录
            for l in range(self.n_layers):
                for g in range(self.n_groups):
                    M_history[l][g].append(self.layers[l].get_group_order_parameter(g))
        
        return M_history
    
    def get_all_stable_states(self):
        """获取所有层-群体的稳定态"""
        final_M = {}
        for l in range(self.n_layers):
            for g in range(self.n_groups):
                final_M[(l, g)] = self.layers[l].get_group_order_parameter(g)
        return final_M


# ============================================================================
# 第三步：多稳相图绘制
# ============================================================================

def count_stable_states(M_dict, tolerance=0.1):
    """
    统计稳定态数量
    简化版：统计序参量的唯一值数量
    """
    M_values = list(M_dict.values())
    unique_states = []
    
    for M in M_values:
        is_new = True
        for uM in unique_states:
            if abs(M - uM) < tolerance:
                is_new = False
                break
        if is_new:
            unique_states.append(M)
    
    return len(unique_states), unique_states


def draw_2d_phase_diagram(param1_range, param2_range, fixed_param, 
                          param_name1, param_name2, fixed_name,
                          save_path):
    """绘制二维多稳相图"""
    
    results = np.zeros((len(param1_range), len(param2_range)))
    
    for i, p1 in enumerate(param1_range):
        for j, p2 in enumerate(param2_range):
            # 设置参数
            if fixed_name == 'lambda':
                lambda_val = fixed_param
                gamma_val = p1
                mu_val = p2
            elif fixed_name == 'gamma':
                lambda_val = p1
                gamma_val = fixed_param
                mu_val = p2
            else:
                lambda_val = p1
                gamma_val = p2
                mu_val = fixed_param
            
            # 运行多次取平均
            stable_counts = []
            for seed in range(5):
                system = MultiLayerMultiGroupHypergraph(
                    N=30, n_groups=3, n_layers=2,
                    gamma=gamma_val,
                    inter_group_competition=lambda_val,
                    inter_layer_coupling=mu_val,
                    seed=seed
                )
                system.run_dynamics(steps=50)
                states = system.get_all_stable_states()
                count, _ = count_stable_states(states)
                stable_counts.append(count)
            
            results[i, j] = np.mean(stable_counts)
    
    # 绘制
    plt.figure(figsize=(10, 8))
    plt.imshow(results, extent=[param2_range[0], param2_range[-1], 
                                param1_range[0], param1_range[-1]],
               aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label='Number of Stable States')
    plt.xlabel(param_name2)
    plt.ylabel(param_name1)
    plt.title(f'Multi-stability Phase Diagram ({fixed_name}={fixed_param})')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return results


# ============================================================================
# 实验1: 多群体竞争对多稳态的影响
# ============================================================================

def experiment1_multi_group_effect():
    """验证多群体竞争对稳态度数的影响"""
    
    print("=" * 60)
    print("实验1: 多群体竞争对多稳态的影响")
    print("=" * 60)
    
    results = {}
    
    for n_groups in [3, 4, 5]:
        print(f"\n群体数量: {n_groups}")
        lambda_results = []
        
        for lambda_val in np.arange(0, 1.1, 0.1):
            counts = []
            for seed in range(5):
                system = MultiGroupHypergraph(
                    N=30, n_groups=n_groups, gamma=0.4,
                    inter_group_competition=lambda_val,
                    seed=seed
                )
                system.run_dynamics(steps=50)
                states = system.get_stable_states()
                count, _ = count_stable_states(states)
                counts.append(count)
            
            avg_count = np.mean(counts)
            lambda_results.append(avg_count)
            print(f"  λ={lambda_val:.1f}: 稳态度数 = {avg_count:.1f}")
        
        results[n_groups] = lambda_results
    
    # 绘图
    plt.figure(figsize=(10, 6))
    for n_groups, lambda_results in results.items():
        plt.plot(np.arange(0, 1.1, 0.1), lambda_results, 
                 marker='o', label=f'{n_groups} groups')
    
    plt.xlabel('Inter-group Competition (λ)')
    plt.ylabel('Number of Stable States')
    plt.title('Effect of Multi-Group Competition on Stability')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('F:/hypergraph_bistability/figures/multi_stability/exp1_multi_group.png', 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    return results


# ============================================================================
# 实验2: 多层超图对多稳态的影响
# ============================================================================

def experiment2_multi_layer_effect():
    """验证多层超图对稳态度数的影响"""
    
    print("=" * 60)
    print("实验2: 多层超图对多稳态的影响")
    print("=" * 60)
    
    results = {}
    
    for n_layers in [2, 3, 4]:
        print(f"\n层数: {n_layers}")
        mu_results = []
        
        for mu in np.arange(-0.5, 0.6, 0.1):
            counts = []
            for seed in range(5):
                system = MultiLayerMultiGroupHypergraph(
                    N=30, n_groups=3, n_layers=n_layers,
                    gamma=0.4, inter_group_competition=0.5,
                    inter_layer_coupling=mu,
                    seed=seed
                )
                system.run_dynamics(steps=50)
                states = system.get_all_stable_states()
                count, _ = count_stable_states(states)
                counts.append(count)
            
            avg_count = np.mean(counts)
            mu_results.append(avg_count)
            print(f"  μ={mu:.1f}: 稳态度数 = {avg_count:.1f}")
        
        results[n_layers] = mu_results
    
    # 绘图
    plt.figure(figsize=(10, 6))
    for n_layers, mu_results in results.items():
        plt.plot(np.arange(-0.5, 0.6, 0.1), mu_results, 
                 marker='o', label=f'{n_layers} layers')
    
    plt.xlabel('Inter-layer Coupling (μ)')
    plt.ylabel('Number of Stable States')
    plt.title('Effect of Multi-Layer Hypergraph on Stability')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('F:/hypergraph_bistability/figures/multi_stability/exp2_multi_layer.png', 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    return results


# ============================================================================
# 实验3: 多稳相图绘制
# ============================================================================

def experiment3_phase_diagram():
    """绘制多稳相图"""
    
    print("=" * 60)
    print("实验3: 多稳相图绘制")
    print("=" * 60)
    
    # 固定一个参数，画二维相图
    gamma_range = np.arange(0.1, 0.8, 0.1)
    lambda_range = np.arange(0, 1.1, 0.1)
    
    print("\n绘制 γ-λ 相图 (μ=0.2)...")
    results = draw_2d_phase_diagram(
        gamma_range, lambda_range, 0.2,
        'gamma', 'lambda', 'mu',
        'F:/hypergraph_bistability/figures/multi_stability/phase_gamma_lambda.png'
    )
    
    print("相图绘制完成!")
    return results


# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("多稳态超图动力学 - 实验框架")
    print("=" * 60)
    
    # 实验1: 多群体竞争
    exp1_results = experiment1_multi_group_effect()
    
    # 实验2: 多层超图
    exp2_results = experiment2_multi_layer_effect()
    
    # 实验3: 相图
    exp3_results = experiment3_phase_diagram()
    
    print("\n" + "=" * 60)
    print("所有实验完成!")
    print("=" * 60)
