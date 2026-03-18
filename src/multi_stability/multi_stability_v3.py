"""
多稳态超图 v3 - 重新定义稳定态
======================
核心问题：原来双稳态是指"同一系统不同初始条件→不同吸引子"
多群体应该看：不同初始条件下，群体分布是否呈现不同模式

新思路：
- 把多群体超图看作"多个独立的双稳态系统"
- 关键指标：群体间的序参量差异
- 如果某些群体→M≈1（高活跃），某些群体→M≈0.45（低活跃），就是多稳态
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import os

os.makedirs('F:/hypergraph_bistability/figures/multi_stability', exist_ok=True)


class MultiGroupV3:
    """改进版多群体超图"""
    
    def __init__(self, N=50, n_groups=3, gamma=0.35, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        
        self.N = N
        self.n_groups = n_groups
        self.gamma = gamma
        self.K = int(gamma * N)
        
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
        
        # 群体初始状态分化 (关键!)
        # 某些群体初始度高，某些初始度低
        for g in range(n_groups):
            group_nodes = [v for v in self.V if self.group_assignments[v] == g]
            # 交替设置初始度
            if g % 2 == 0:
                # 高初始度
                for v in group_nodes[:len(group_nodes)//2]:
                    self.E.append(frozenset([v, (v+1)%N]))
            else:
                # 低初始度 (只有少量超边)
                if group_nodes:
                    self.E.append(frozenset([group_nodes[0], group_nodes[0]]))
    
    def get_node_degree(self, v):
        return sum(1 for e in self.E if v in e)
    
    def get_group_M(self, g):
        """群体g的序参量：归一化平均度"""
        group_nodes = [v for v in self.V if self.group_assignments[v] == g]
        if not group_nodes:
            return 0.0
        degrees = [self.get_node_degree(v) for v in group_nodes]
        return np.mean(degrees) / max(self.K, 1)
    
    def apply_rules(self):
        """动力学规则 - 简化版"""
        avg_deg = len(self.E) / max(len(self.V), 1)
        
        # 生长
        if random.random() < 0.35:
            v = random.choice(self.V)
            w = len(self.V)
            self.V.append(w)
            self.group_assignments[w] = self.group_assignments[v]
            self.E.append(frozenset([v, w]))
        
        # 融合
        if random.random() < 0.3 and len(self.E) >= 2:
            e1, e2 = random.sample(self.E, 2)
            if len(e1 & e2) >= 1:
                g1 = self.group_assignments[list(e1)[0]]
                g2 = self.group_assignments[list(e2)[0]]
                
                # 同群体容易融合，异群体难
                if g1 == g2:
                    new_e = frozenset(e1 | e2)
                    if new_e not in self.E and len(new_e) <= self.K:
                        self.E.append(new_e)
                        self.E.remove(e1)
                        self.E.remove(e2)
                else:
                    if random.random() < 0.1:  # 异群体只有10%概率融合
                        new_e = frozenset(e1 | e2)
                        if new_e not in self.E and len(new_e) <= self.K:
                            self.E.append(new_e)
                            self.E.remove(e1)
                            self.E.remove(e2)
        
        # 分裂
        if random.random() < 0.15:
            large = [e for e in self.E if len(e) > 2]
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
            if len(self.E) > 2:
                self.E.pop(random.randrange(len(self.E)))
    
    def run(self, steps=80):
        """运行"""
        history = []
        for _ in range(steps):
            self.apply_rules()
            M = [self.get_group_M(g) for g in range(self.n_groups)]
            history.append(M)
        return np.array(history)


def analyze_multistability():
    """分析多稳态"""
    
    print("=" * 60)
    print("多稳态分析 v3")
    print("=" * 60)
    
    results = {}
    
    # 测试不同初始条件
    for n_groups in [2, 3, 4, 5]:
        print(f"\n群体数: {n_groups}")
        
        # 多次运行，看分布
        all_M = []
        for seed in range(20):
            sys = MultiGroupV3(N=40, n_groups=n_groups, gamma=0.35, seed=seed)
            history = sys.run(steps=60)
            final_M = history[-1]
            all_M.append(final_M)
        
        all_M = np.array(all_M)
        
        # 分析稳态分布
        print(f"  最终序参量分布:")
        for g in range(n_groups):
            vals = all_M[:, g]
            print(f"    群体{g}: mean={np.mean(vals):.2f}, std={np.std(vals):.2f}, range=[{np.min(vals):.2f}, {np.max(vals):.2f}]")
        
        # 检查是否有双峰分布（表示多稳态）
        combined = all_M.flatten()
        hist, bins = np.histogram(combined, bins=20)
        peaks = np.where((hist[1:-1] > hist[:-2]) & (hist[1:-1] > hist[2:]))[0]
        
        print(f"  直方图峰值数: {len(peaks)}")
        
        # 计算群体间差异
        between_group_var = np.var([np.mean(all_M[:, g]) for g in range(n_groups)])
        within_group_var = np.mean([np.var(all_M[:, g]) for g in range(n_groups)])
        
        print(f"  组间方差: {between_group_var:.3f}")
        print(f"  组内方差: {within_group_var:.3f}")
        
        results[n_groups] = {
            'all_M': all_M,
            'peaks': len(peaks),
            'between_var': between_group_var,
            'within_var': within_group_var
        }
    
    # 绘图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for idx, n_groups in enumerate([2, 3, 4, 5]):
        ax = axes[idx // 2, idx % 2]
        all_M = results[n_groups]['all_M']
        
        for g in range(n_groups):
            ax.hist(all_M[:, g], bins=15, alpha=0.5, label=f'Group {g}')
        
        ax.set_xlabel('Order Parameter M')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{n_groups} Groups')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('F:/hypergraph_bistability/figures/multi_stability/v3_distribution.png', dpi=150)
    plt.close()
    
    return results


def test_bistability_per_group():
    """测试每个群体内部是否有双稳态"""
    
    print("\n" + "=" * 60)
    print("测试：每个群体内部的双稳态")
    print("=" * 60)
    
    # 方法：固定群体数，改变gamma，看是否每个群体内部有双稳态
    for gamma in [0.2, 0.3, 0.35, 0.4, 0.5]:
        print(f"\ngamma: {gamma}")
        
        all_final_M = []
        for seed in range(30):
            sys = MultiGroupV3(N=40, n_groups=3, gamma=gamma, seed=seed)
            history = sys.run(steps=60)
            final_M = history[-1]
            all_final_M.append(np.mean(final_M))
        
        all_final_M = np.array(all_final_M)
        
        # 检查分布
        hist, bins = np.histogram(all_final_M, bins=20)
        
        # 找峰值
        peaks = []
        for i in range(1, len(hist) - 1):
            if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
                peaks.append((bins[i] + bins[i+1]) / 2)
        
        print(f"  峰值位置: {peaks}")
        print(f"  M分布: mean={np.mean(all_final_M):.2f}, std={np.std(all_final_M):.2f}")


if __name__ == "__main__":
    results = analyze_multistability()
    test_bistability_per_group()
    print("\n完成!")
