"""
超图多稳态相图 - 完整实现
============================

核心思路：
- k个群体 × L层 = k×L 维状态空间
- 每个群体有不同的 Kc（非对称）
- 使用聚类方法计算吸引子数量

参考：n_wells_arrangements.py 的聚类逻辑
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from collections import defaultdict
import os

os.makedirs('F:/hypergraph_bistability/figures/multistable_phase', exist_ok=True)


# ============================================================
# 核心模型：多群体超图动力学
# ============================================================

def hypergraph_multistable_dynamics(M, t, k, L, Kc_list, lambda_, mu):
    """
    超图多稳态动力学
    
    参数:
    - M: 状态向量，形状 (k, L) - k个群体，每个L层
    - Kc_list: 每个群体的容量约束 (k,)
    - lambda_: 群体间耦合强度
    - mu: 层间耦合强度
    """
    k, L = len(Kc_list), L
    dMdt = np.zeros_like(M)
    
    # 重组为 (k, L) 形状
    M = np.array(M).reshape(k, L)
    
    for i in range(k):  # 群体 i
        for l in range(L):  # 层 l
            # 基本动力学：容量约束项
            Mc = M[i, l]  # 当前值
            Kc_i = Kc_list[i]  # 群体 i 的容量
            
            # 1. 自身非线性项 (logistic-like)
            term1 = Mc * (1 - Mc / Kc_i)
            
            # 2. 群体间竞争项 (j != i)
            competition = 0
            for j in range(k):
                if j != i:
                    # 群体 j 对 i 的竞争
                    Mj_mean = np.mean(M[j, :])  # 群体 j 的平均活动
                    competition += lambda_ * (Mj_mean - Mc)
            
            # 3. 层间耦合项 (l' != l)
            layer_coupling = 0
            for l2 in range(L):
                if l2 != l:
                    layer_coupling += mu * (M[i, l2] - Mc)
            
            dMdt[i * L + l] = term1 + competition + layer_coupling
    
    return dMdt


def simulate(M0, t, k, L, Kc_list, lambda_, mu):
    """运行模拟"""
    M0 = np.array(M0).flatten()
    sol = odeint(hypergraph_multistable_dynamics, M0, t, 
                 args=(k, L, Kc_list, lambda_, mu))
    return sol[-1]  # 返回最终状态


def cluster_attractors(final_states, threshold=0.12):
    """
    聚类吸引子（参考 n_wells_arrangements.py）
    
    参数:
    - final_states: 所有最终状态的列表
    - threshold: 聚类阈值
    
    返回:
    - unique_attractors: 唯一的吸引子列表
    """
    unique = []
    
    for state in final_states:
        is_new = True
        for up in unique:
            # 计算两个状态之间的"距离"（使用均方根）
            dist = np.sqrt(np.mean((state - up)**2))
            if dist < threshold:
                is_new = False
                break
        if is_new:
            unique.append(state)
    
    return unique


def count_attractors(k, L, Kc_list, lambda_, mu, n_init=150, threshold=0.12, t_max=30):
    """
    计算给定参数下的吸引子数量
    """
    t = np.linspace(0, t_max, 300)
    
    # 生成随机初始条件
    finals = []
    for _ in range(n_init):
        # 每个 M[i,l] 在 [0.1, 0.9] 范围内随机
        M0 = np.random.uniform(0.1, 0.9, k * L)
        try:
            final = simulate(M0, t, k, L, Kc_list, lambda_, mu)
            finals.append(final)
        except:
            pass
    
    # 聚类
    unique = cluster_attractors(finals, threshold)
    
    return len(unique), unique, finals


# ============================================================
# 实验1：对称 vs 非对称 Kc
# ============================================================

def experiment_symmetry():
    """实验：对称 vs 非对称 Kc"""
    print("=" * 60)
    print("Experiment 1: Symmetry vs Asymmetry")
    print("=" * 60)
    
    k, L = 3, 2  # 3个群体，2层 = 6维状态空间
    lambda_ = 0.5
    mu = 0.0
    n_init = 150
    
    # 对称 Kc
    Kc_sym = [0.4, 0.4, 0.4]
    n_sym, _, _ = count_attractors(k, L, Kc_sym, lambda_, mu, n_init)
    print(f"\nSymmetric Kc={Kc_sym}: {n_sym} attractors")
    
    # 轻微非对称
    Kc_asym1 = [0.32, 0.40, 0.48]
    n_asym1, _, _ = count_attractors(k, L, Kc_asym1, lambda_, mu, n_init)
    print(f"Slightly asymmetric Kc={Kc_asym1}: {n_asym1} attractors")
    
    # 强非对称
    Kc_asym2 = [0.20, 0.50, 0.80]
    n_asym2, _, _ = count_attractors(k, L, Kc_asym2, lambda_, mu, n_init)
    print(f"Strongly asymmetric Kc={Kc_asym2}: {n_asym2} attractors")
    
    return {
        'symmetric': (Kc_sym, n_sym),
        'slight': (Kc_asym1, n_asym1),
        'strong': (Kc_asym2, n_asym2)
    }


# ============================================================
# 实验2：lambda 扫描
# ============================================================

def experiment_lambda_scan():
    """实验：lambda 对吸引子数量的影响"""
    print("\n" + "=" * 60)
    print("Experiment 2: Lambda Scan")
    print("=" * 60)
    
    k, L = 3, 2
    Kc_list = [0.32, 0.40, 0.48]  # 非对称
    mu = 0.0
    n_init = 150
    
    results = []
    for lambda_ in np.linspace(0, 1.0, 11):
        n_att, _, _ = count_attractors(k, L, Kc_list, lambda_, mu, n_init)
        results.append((lambda_, n_att))
        print(f"lambda={lambda_:.2f}: {n_att} attractors")
    
    return results


# ============================================================
# 实验3：2D相图（lambda vs Kc差异）
# ============================================================

def experiment_2d_phase():
    """实验：2D相图"""
    print("\n" + "=" * 60)
    print("Experiment 3: 2D Phase Diagram")
    print("=" * 60)
    
    k, L = 3, 2
    n_init = 100  # 减少点数加快速度
    mu = 0.0
    
    # 网格
    lambda_vals = np.linspace(0, 0.8, 9)
    delta_Kc_vals = np.linspace(0, 0.3, 7)
    
    phase_data = np.zeros((len(delta_Kc_vals), len(lambda_vals)))
    
    for i, delta_Kc in enumerate(delta_Kc_vals):
        for j, lambda_ in enumerate(lambda_vals):
            # Kc = [0.4-delta, 0.4, 0.4+delta]
            Kc_list = [0.4 - delta_Kc, 0.4, 0.4 + delta_Kc]
            n_att, _, _ = count_attractors(k, L, Kc_list, lambda_, mu, n_init)
            phase_data[i, j] = n_att
            print(f"delta_Kc={delta_Kc:.2f}, lambda={lambda_:.2f}: {n_att} attractors")
    
    # 绘图
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(phase_data, origin='lower', aspect='auto',
                   extent=[lambda_vals[0], lambda_vals[-1], 
                          delta_Kc_vals[0], delta_Kc_vals[-1]],
                   cmap='RdYlGn_r', vmin=0, vmax=10)
    
    plt.colorbar(im, ax=ax, label='Number of Attractors')
    
    ax.set_xlabel('λ (inter-group coupling)')
    ax.set_ylabel('ΔKc (asymmetry)')
    ax.set_title('Phase Diagram: Attractors vs λ and ΔKc\n(k=3, L=2, 6D state space)')
    
    plt.tight_layout()
    plt.savefig('F:/hypergraph_bistability/figures/multistable_phase/phase_2d.png', dpi=150)
    plt.close()
    
    print("\nFigure saved!")
    
    return phase_data, lambda_vals, delta_Kc_vals


# ============================================================
# 实验4：验证你的结果（25个吸引子）
# ============================================================

def experiment_verify():
    """验证你提到的25个吸引子结果"""
    print("\n" + "=" * 60)
    print("Experiment 4: Verification")
    print("=" * 60)
    
    k, L = 3, 2
    Kc_list = [0.32, 0.40, 0.48]
    lambda_ = 0.5
    mu = 0.0
    n_init = 200  # 更多初始点
    
    n_att, attractors, finals = count_attractors(k, L, Kc_list, lambda_, mu, 
                                                  n_init, threshold=0.15, t_max=40)
    
    print(f"\nKc={Kc_list}, lambda={lambda_}, mu={mu}")
    print(f"n_init={n_init}")
    print(f"Attractors found: {n_att}")
    
    # 统计每个吸引子的 basin 大小
    basin_sizes = defaultdict(int)
    for f in finals:
        matched = False
        for i, at in enumerate(attractors):
            dist = np.sqrt(np.mean((f - at)**2))
            if dist < 0.15:
                basin_sizes[i] += 1
                matched = True
                break
    
    print("\nBasin sizes:")
    for i, size in sorted(basin_sizes.items(), key=lambda x: -x[1])[:10]:
        print(f"  Basin {i}: {size} ({size/n_init*100:.1f}%)")
    
    return n_att, attractors, finals


# ============================================================
# 主函数
# ============================================================

if __name__ == "__main__":
    print("Starting experiments...\n")
    
    # 实验1：对称 vs 非对称
    exp1_results = experiment_symmetry()
    
    # 实验2：lambda扫描
    exp2_results = experiment_lambda_scan()
    
    # 实验3：2D相图
    exp3_data = experiment_2d_phase()
    
    # 实验4：验证
    exp4_result = experiment_verify()
    
    print("\n" + "=" * 60)
    print("All experiments completed!")
    print("=" * 60)
