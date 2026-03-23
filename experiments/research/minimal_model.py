"""
最小可行性模型
=============

目标：构造一个只保留"可行性 + 重建 + 阈值"三要素的极简系统

模型：
F(x) = a * x * (1-x) - b

x ∈ [0,1] = 凝聚度 (对应 M)
a = 竞争强度
b = 约束强度 (complexity)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import os

os.makedirs('F:/hypergraph_bistability/figures/model', exist_ok=True)


def feasibility_function(x, a, b):
    """
    可行性函数
    F(x) > 0: 可自洽
    F(x) < 0: 不可自洽
    """
    return a * x * (1 - x) - b


def find_fixed_points(a, b):
    """
    找固定点: F(x) = 0
    """
    # 解 ax(1-x) - b = 0
    # => ax - ax^2 - b = 0
    # => ax^2 - ax + b = 0
    
    discriminant = a**2 - 4*a*b  # 判别式
    
    if discriminant < 0:
        return []  # 无解
    
    x1 = (a - np.sqrt(discriminant)) / (2*a)
    x2 = (a + np.sqrt(discriminant)) / (2*a)
    
    # 只返回 [0,1] 区间内的
    roots = []
    if 0 <= x1 <= 1:
        roots.append(x1)
    if 0 <= x2 <= 1:
        roots.append(x2)
    
    return roots


def stability(x, a, b):
    """
    判断稳定性: dF/dx < 0 = 稳定
    """
    dFdx = a * (1 - 2*x)
    return dFdx


def model_simulation(a, b, x0, steps=100):
    """
    简单动力学模拟
    x_{t+1} = x_t + α * F(x_t)
    """
    x = x0
    trajectory = [x]
    
    for _ in range(steps):
        F = feasibility_function(x, a, b)
        # 简单的梯度上升
        x = x + 0.1 * F
        x = np.clip(x, 0, 1)  # 限制在 [0,1]
        trajectory.append(x)
    
    return trajectory


def plot_phase_diagram():
    """绘制相图"""
    print("=" * 60)
    print("1. 相图分析")
    print("=" * 60)
    
    a = 1.0
    
    # 扫描 b
    b_range = np.linspace(0, 0.25, 50)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 子图1: 不同b下的F(x)
    ax1 = axes[0, 0]
    x = np.linspace(0, 1, 100)
    
    for b in [0.05, 0.1, 0.15, 0.2, 0.25]:
        F = feasibility_function(x, a, b)
        ax1.plot(x, F, label=f'b={b}')
    
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax1.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5)
    ax1.set_xlabel('x (cohesion)')
    ax1.set_ylabel('F(x) (feasibility)')
    ax1.set_title('Feasibility Function F(x) = a*x*(1-x) - b')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 子图2: 固定点随b变化
    ax2 = axes[0, 1]
    
    b_fine = np.linspace(0, 0.25, 100)
    stable_points = []
    unstable_points = []
    
    for b in b_fine:
        roots = find_fixed_points(a, b)
        for root in roots:
            stab = stability(root, a, b)
            if stab < 0:
                stable_points.append((b, root))
            else:
                unstable_points.append((b, root))
    
    if stable_points:
        bs, xs = zip(*stable_points)
        ax2.plot(xs, bs, 'b-', label='Stable', linewidth=2)
    if unstable_points:
        bs, xs = zip(*unstable_points)
        ax2.plot(xs, bs, 'r--', label='Unstable', linewidth=2)
    
    ax2.set_xlabel('x (fixed point)')
    ax2.set_ylabel('b (constraint strength)')
    ax2.set_title('Fixed Points in (x, b) Space')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 子图3: 双稳态演示
    ax3 = axes[1, 0]
    
    b = 0.15  # 双稳态区域
    x0_range = np.linspace(0.05, 0.95, 20)
    
    final_states = []
    for x0 in x0_range:
        traj = model_simulation(a, b, x0, steps=50)
        final_states.append(traj[-1])
    
    ax3.plot(x0_range, final_states, 'o-', markersize=4)
    ax3.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='y=x')
    ax3.set_xlabel('Initial x₀')
    ax3.set_ylabel('Final x')
    ax3.set_title(f'Bistability: b={b}')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 子图4: P(HIGH) vs 初始条件
    ax4 = axes[1, 1]
    
    threshold = 0.5  # HIGH/LOW 分界
    
    # 对不同x0，统计最终状态
    results = []
    for n_init_ratio in np.linspace(0.1, 0.9, 30):
        # 模拟多次随机
        high_count = 0
        for _ in range(50):
            x0 = n_init_ratio + np.random.randn() * 0.1
            x0 = np.clip(x0, 0.01, 0.99)
            traj = model_simulation(a, b, x0, steps=50)
            if traj[-1] > threshold:
                high_count += 1
        
        results.append((n_init_ratio, high_count / 50))
    
    x0s, p_high = zip(*results)
    ax4.plot(x0s, p_high, 'o-')
    ax4.set_xlabel('Initial Condition (x₀)')
    ax4.set_ylabel('P(HIGH)')
    ax4.set_title('S-Curve: P(HIGH | x₀)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('F:/hypergraph_bistability/figures/model/phase_diagram.png', dpi=150)
    plt.close()


def plot_critical_slowing():
    """绘制临界减速"""
    print("\n" + "=" * 60)
    print("2. 临界减速分析")
    print("=" * 60)
    
    a = 1.0
    
    # 在b临界值附近分析
    # 临界点: b = a/4 = 0.25 时只有一个解
    # 双稳态区: b < 0.25
    
    b_values = [0.10, 0.15, 0.20, 0.22, 0.24]
    
    plt.figure(figsize=(12, 5))
    
    # 子图1: 收敛时间 vs b
    ax1 = plt.subplot(1, 2, 1)
    
    convergence_times = []
    b_eff = []
    
    for b in b_values:
        roots = find_fixed_points(a, b)
        
        # 计算每个根的收敛时间（梯度越小越慢）
        for root in roots:
            dFdx = abs(stability(root, a, b))
            if dFdx > 0.01:  # 避免除以0
                conv_time = 1 / dFdx
            else:
                conv_time = 100
            convergence_times.append(conv_time)
            b_eff.append(b)
    
    ax1.plot(b_eff, convergence_times, 'o-')
    ax1.set_xlabel('b (constraint)')
    ax1.set_ylabel('Convergence Time (∝ 1/|dF/dx|)')
    ax1.set_title('Critical Slowing: Convergence Time')
    ax1.grid(True, alpha=0.3)
    
    # 子图2: 固定点稳定性
    ax2 = plt.subplot(1, 2, 2)
    
    x = np.linspace(0, 1, 100)
    
    for b in [0.15, 0.20, 0.24]:
        F = feasibility_function(x, a, b)
        ax2.plot(x, F, label=f'b={b}')
    
    ax2.axhline(y=0, color='k', linestyle='--')
    ax2.set_xlabel('x')
    ax2.set_ylabel('F(x)')
    ax2.set_title('Approaching Critical Point')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('F:/hypergraph_bistability/figures/model/critical_slowing.png', dpi=150)
    plt.close()
    
    print("临界减速分析完成")


def fit_experimental_data():
    """用实验数据拟合模型参数"""
    print("\n" + "=" * 60)
    print("3. 拟合实验数据")
    print("=" * 60)
    
    # 实验数据 (从之前的critical_analysis)
    # n_init -> P(HIGH)
    # n_init = 10: P=0.07
    # n_init = 15: P=0.58
    # n_init = 20: P=0.96
    
    # 把 n_init 映射到 x0
    # 假设: x0 ∝ n_init / N
    # 临界区在 n_init ≈ 15
    
    # 简单映射
    def n_to_x0(n, scale=0.03):
        return scale * n
    
    # 实验数据
    exp_data = [
        (10, 0.07),
        (11, 0.19),
        (12, 0.19),
        (13, 0.33),
        (14, 0.48),
        (15, 0.58),
        (16, 0.76),
        (17, 0.88),
        (18, 0.85),
        (19, 0.92),
        (20, 0.96),
    ]
    
    # 模型函数: P(HIGH | x0)
    def model_p_high(x0, a, b, threshold=0.5):
        """计算P(HIGH) = P(x_final > threshold)"""
        # 模拟多次
        p_high = 0
        for _ in range(100):
            x0_noise = x0 + np.random.randn() * 0.05
            traj = model_simulation(a, b, x0_noise, steps=50)
            if traj[-1] > threshold:
                p_high += 1
        return p_high / 100
    
    # 拟合
    from scipy.optimize import minimize
    
    def loss(params):
        a, b = params
        if a <= 0 or b <= 0 or b >= a/4:  # 约束
            return 1e6
        
        total_error = 0
        for n, p_exp in exp_data:
            x0 = n_to_x0(n)
            p_model = model_p_high(x0, a, b)
            total_error += (p_model - p_exp)**2
        
        return total_error
    
    # 搜索最优参数
    best_params = None
    best_loss = float('inf')
    
    for a_try in np.linspace(0.5, 2.0, 10):
        for b_try in np.linspace(0.05, 0.2, 10):
            loss_val = loss((a_try, b_try))
            if loss_val < best_loss:
                best_loss = loss_val
                best_params = (a_try, b_try)
    
    a_opt, b_opt = best_params
    print(f"最优参数: a = {a_opt:.3f}, b = {b_opt:.3f}")
    print(f"拟合误差: {best_loss:.4f}")
    
    # 绘制拟合结果
    plt.figure(figsize=(10, 6))
    
    # 实验点
    n_exp, p_exp = zip(*exp_data)
    plt.scatter(n_exp, p_exp, s=100, label='Experiment', zorder=5)
    
    # 模型曲线
    n_model = range(5, 35)
    p_model = [model_p_high(n_to_x0(n), a_opt, b_opt) for n in n_model]
    plt.plot(n_model, p_model, '-', label='Model', linewidth=2)
    
    plt.xlabel('Initial Condition (n_init)')
    plt.ylabel('P(HIGH)')
    plt.title(f'Model Fit: a={a_opt:.2f}, b={b_opt:.2f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('F:/hypergraph_bistability/figures/model/fit.png', dpi=150)
    plt.close()


def analyze_gap():
    """分析中间gap"""
    print("\n" + "=" * 60)
    print("4. 中间Gap分析")
    print("=" * 60)
    
    a = 1.0
    
    # 不同b值
    b_values = [0.10, 0.15, 0.20, 0.24]
    
    plt.figure(figsize=(10, 6))
    
    for b in b_values:
        roots = find_fixed_points(a, b)
        print(f"\nb = {b}: 固定点 = {roots}")
        
        # 计算稳定性
        for root in roots:
            dFdx = stability(root, a, b)
            print(f"  x = {root:.3f}, dF/dx = {dFdx:.3f}, {'稳定' if dFdx < 0 else '不稳定'}")
        
        # 可行性区间
        x = np.linspace(0, 1, 100)
        F = feasibility_function(x, a, b)
        
        feasible = F > 0
        plt.fill_between(x, 0, feasible.astype(int), alpha=0.3, label=f'b={b}')
    
    plt.xlabel('x (cohesion)')
    plt.ylabel('Feasible (1) / Infeasible (0)')
    plt.title('Feasibility Regions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('F:/hypergraph_bistability/figures/model/gap.png', dpi=150)
    plt.close()


def theoretical_summary():
    """理论总结"""
    print("\n" + "=" * 60)
    print("理论总结")
    print("=" * 60)
    
    print("""
模型: F(x) = a * x * (1-x) - b

关键洞察:
1. x(1-x) 项意味着:
   - 低x: 结构简单，易满足
   - 高x: 凝聚优势
   - 中间: 最难自洽（两边不靠）

2. 双稳态条件: b < a/4
   - 两个稳定固定点
   - 一个不稳定分界点

3. 临界减速: 当 dF/dx → 0
   - 在b接近a/4时发生
   - 对应实验中的n_init≈15

4. Gap产生:
   - 中间x值对应F(x)最小
   - 当b较大时，中间变得不可行
   - 实验中观察到"弱gap"（10%概率）

对应实验观测:
- 双峰分布 ↔ 两个稳定固定点
- S曲线 ↔ 初始条件→不同吸引域
- 临界减速 ↔ b≈a/4时收敛变慢
- 中间稀疏 ↔ F(x)在中间最小
""")


if __name__ == "__main__":
    plot_phase_diagram()
    plot_critical_slowing()
    fit_experimental_data()
    analyze_gap()
    theoretical_summary()
    print("\n完成!")
