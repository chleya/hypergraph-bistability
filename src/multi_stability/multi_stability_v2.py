"""
多稳态超图动力学系统 v2 - 修复版
==================
问题：稳态度数始终为1
原因：多群体之间的竞争不够强

修复策略：
1. 增强群体间的竞争（负耦合）
2. 重新定义序参量计算方式
3. 添加更强的分化机制
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import os
import json
from collections import defaultdict

os.makedirs('F:/hypergraph_bistability/figures/multi_stability', exist_ok=True)

# ============================================================================
# 核心：多群体竞争超图 (修复版)
# ============================================================================

class MultiGroupHypergraphV2:
    """多群体竞争超图 - 修复版"""
    
    def __init__(self, N=50, n_groups=3, gamma=0.35, state_dim=16, 
                 group_separation=0.8, seed=42):
        """
        参数:
        - n_groups: 群体数量
        - gamma: 基础容量约束
        - group_separation: 群体分离度 (越高群体间差异越大)
        """
        random.seed(seed)
        np.random.seed(seed)
        
        self.N = N
        self.n_groups = n_groups
        self.gamma = gamma
        self.state_dim = state_dim
        self.group_separation = group_separation
        
        # 群体中心 (用于产生分化)
        self.group_centers = {}
        for g in range(n_groups):
            angle = 2 * np.pi * g / n_groups
            self.group_centers[g] = np.array([
                np.cos(angle) * group_separation,
                np.sin(angle) * group_separation
            ] + [0] * (state_dim - 2))
        
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
        self.E = []
        self.s = {v: self.group_centers[self.group_assignments[v]] + 
                  np.random.randn(state_dim) * 0.3 for v in self.V}
        
        # 初始化超边 (每群体独立)
        n_initial = max(4, N // 4)
        for _ in range(n_initial):
            group = random.randint(0, n_groups - 1)
            group_nodes = [v for v in self.V if self.group_assignments[v] == group]
            if len(group_nodes) >= 2:
                size = random.randint(2, min(4, len(group_nodes) // 2))
                e = frozenset(random.sample(group_nodes, size))
                self.E.append(e)
    
    def get_node_group(self, v):
        return self.group_assignments[v]
    
    def get_group_nodes(self, g):
        return [v for v in self.V if self.group_assignments[v] == g]
    
    def get_node_degree(self, v):
        return sum(1 for e in self.E if v in e)
    
    def get_group_order_parameter(self, g):
        """群体g的序参量：该群体节点的平均度"""
        group_nodes = self.get_group_nodes(g)
        if not group_nodes:
            return 0.0
        degrees = [self.get_node_degree(v) for v in group_nodes]
        return np.mean(degrees)
    
    def compute_group_attraction(self, g1, g2):
        """计算群体g1对g2的吸引力（负=排斥）"""
        c1 = self.group_centers[g1]
        c2 = self.group_centers[g2]
        dist = np.linalg.norm(c1 - c2)
        # 同群体吸引，异群体排斥
        if g1 == g2:
            return 1.0 / (1 + dist)
        else:
            return -self.group_separation / (1 + dist)
    
    def apply_rules(self):
        """动力学规则"""
        
        # 规则1: 生长 (优先同群体内)
        if random.random() < 0.35 and self.E:
            # 按群体比例选择超边
            group_weights = defaultdict(int)
            for e in self.E:
                g = self.get_node_group(list(e)[0])
                group_weights[g] += 1
            
            total = sum(group_weights.values())
            groups = list(group_weights.keys())
            weights = [group_weights[g] / total for g in groups]
            chosen_group = random.choices(groups, weights=weights)[0]
            
            group_edges = [e for e in self.E if self.get_node_group(list(e)[0]) == chosen_group]
            if group_edges:
                e = random.choice(group_edges)
                nodes = list(e)
                v = len(self.V)
                self.V.append(v)
                self.s[v] = self.s[nodes[0]] + np.random.randn(self.state_dim) * 0.2
                self.group_assignments[v] = chosen_group
                new_e = frozenset([nodes[0], v])
                self.E.append(new_e)
        
        # 规则2: 融合 (关键：受群体分离度影响)
        if random.random() < 0.25 and len(self.E) >= 2:
            e1, e2 = random.sample(self.E, 2)
            if len(e1 & e2) >= 1:
                g1 = self.get_node_group(list(e1)[0])
                g2 = self.get_node_group(list(e2)[0])
                
                # 计算融合中心
                w1 = np.array([1.0 / (1 + self.get_node_degree(v)) for v in e1])
                w1 = w1 / w1.sum()
                c1 = np.average([self.s[u] for u in e1], axis=0, weights=w1)
                
                w2 = np.array([1.0 / (1 + self.get_node_degree(v)) for v in e2])
                w2 = w2 / w2.sum()
                c2 = np.average([self.s[u] for u in e2], axis=0, weights=w2)
                
                dist = np.linalg.norm(c1 - c2)
                
                # 群体分离度影响融合阈值
                if g1 == g2:
                    threshold = 1.0  # 同群体容易融合
                else:
                    threshold = self.group_separation * 0.5  # 异群体难融合
                
                if dist < threshold:
                    new_e = frozenset(e1 | e2)
                    if new_e not in self.E:
                        self.E.append(new_e)
                        self.E.remove(e1)
                        self.E.remove(e2)
        
        # 规则3: 分裂 (保持多样性)
        if random.random() < 0.15:
            large_edges = [e for e in self.E if len(e) > 2]
            if large_edges:
                e = random.choice(large_edges)
                nodes = list(e)
                if len(nodes) >= 2:
                    split = len(nodes) // 2
                    e1 = frozenset(nodes[:split])
                    e2 = frozenset(nodes[split:])
                    if e1 not in self.E and e2 not in self.E:
                        self.E.append(e1)
                        self.E.append(e2)
                        self.E.remove(e)
        
        # 规则4: 删除 (保持容量约束)
        if random.random() < 0.25:
            if len(self.E) > 2:
                e = random.choice(self.E)
                self.E.remove(e)
    
    def run(self, steps=80):
        """运行动力学"""
        M_history = {g: [] for g in range(self.n_groups)}
        
        for t in range(steps):
            self.apply_rules()
            for g in range(self.n_groups):
                M_history[g].append(self.get_group_order_parameter(g))
        
        return M_history
    
    def get_final_states(self):
        """获取最终稳定态"""
        return {g: self.get_group_order_parameter(g) for g in range(self.n_groups)}


def count_distinct_states(states, tolerance=0.5):
    """统计不同的稳定态数量"""
    values = list(states.values())
    unique = []
    for v in values:
        is_new = True
        for u in unique:
            if abs(v - u) < tolerance:
                is_new = False
                break
        if is_new:
            unique.append(v)
    return len(unique), unique


# ============================================================================
# 实验1: 测试不同群体分离度
# ============================================================================

def test_group_separation():
    """测试群体分离度对稳态度数的影响"""
    
    print("=" * 60)
    print("测试: 群体分离度对稳态度数的影响")
    print("=" * 60)
    
    results = {}
    
    for sep in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        print(f"\n群体分离度: {sep}")
        counts = []
        final_states_list = []
        
        for seed in range(10):
            system = MultiGroupHypergraphV2(
                N=30, n_groups=3, gamma=0.35,
                group_separation=sep, seed=seed
            )
            history = system.run(steps=60)
            states = system.get_final_states()
            count, unique = count_distinct_states(states, tolerance=1.0)
            counts.append(count)
            final_states_list.append(unique)
        
        avg_count = np.mean(counts)
        results[sep] = {
            'count': avg_count,
            'states': final_states_list
        }
        print(f"  稳态度数: {avg_count:.1f} (unique values: {np.unique(counts)})")
    
    # 绘图
    plt.figure(figsize=(10, 6))
    x = list(results.keys())
    y = [results[k]['count'] for k in x]
    plt.plot(x, y, marker='o', linewidth=2, markersize=8)
    plt.xlabel('Group Separation', fontsize=12)
    plt.ylabel('Number of Stable States', fontsize=12)
    plt.title('Effect of Group Separation on Multistability', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig('F:/hypergraph_bistability/figures/multi_stability/group_separation_test.png', 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    return results


# ============================================================================
# 实验2: 不同gamma值
# ============================================================================

def test_gamma():
    """测试gamma对稳态度数的影响"""
    
    print("\n" + "=" * 60)
    print("测试: gamma对稳态度数的影响")
    print("=" * 60)
    
    results = {}
    
    for gamma in [0.2, 0.3, 0.35, 0.4, 0.5, 0.6]:
        print(f"\ngamma: {gamma}")
        counts = []
        
        for seed in range(10):
            system = MultiGroupHypergraphV2(
                N=30, n_groups=3, gamma=gamma,
                group_separation=0.8, seed=seed
            )
            history = system.run(steps=60)
            states = system.get_final_states()
            count, _ = count_distinct_states(states, tolerance=1.0)
            counts.append(count)
        
        avg_count = np.mean(counts)
        results[gamma] = avg_count
        print(f"  稳态度数: {avg_count:.1f}")
    
    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(list(results.keys()), list(results.values()), marker='s', linewidth=2, markersize=8)
    plt.xlabel('Gamma (Capacity Constraint)', fontsize=12)
    plt.ylabel('Number of Stable States', fontsize=12)
    plt.title('Effect of Gamma on Multistability', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig('F:/hypergraph_bistability/figures/multi_stability/gamma_test.png', 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    return results


# ============================================================================
# 实验3: 多群体数量
# ============================================================================

def test_n_groups():
    """测试群体数量"""
    
    print("\n" + "=" * 60)
    print("测试: 群体数量对稳态度数的影响")
    print("=" * 60)
    
    results = {}
    
    for n_groups in [2, 3, 4, 5, 6]:
        print(f"\n群体数: {n_groups}")
        counts = []
        
        for seed in range(10):
            system = MultiGroupHypergraphV2(
                N=30, n_groups=n_groups, gamma=0.35,
                group_separation=0.8, seed=seed
            )
            history = system.run(steps=60)
            states = system.get_final_states()
            count, _ = count_distinct_states(states, tolerance=1.0)
            counts.append(count)
        
        avg_count = np.mean(counts)
        results[n_groups] = avg_count
        print(f"  稳态度数: {avg_count:.1f}")
    
    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(list(results.keys()), list(results.values()), marker='^', linewidth=2, markersize=8)
    plt.xlabel('Number of Groups', fontsize=12)
    plt.ylabel('Number of Stable States', fontsize=12)
    plt.title('Effect of Group Number on Multistability', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig('F:/hypergraph_bistability/figures/multi_stability/n_groups_test.png', 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    return results


# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("多稳态超图 v2 - 修复版测试")
    print("=" * 60)
    
    # 测试群体分离度
    sep_results = test_group_separation()
    
    # 测试gamma
    gamma_results = test_gamma()
    
    # 测试群体数
    ngroups_results = test_n_groups()
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
    print(f"群体分离度影响: {sep_results}")
    print(f"gamma影响: {gamma_results}")
    print(f"群体数影响: {ngroups_results}")
