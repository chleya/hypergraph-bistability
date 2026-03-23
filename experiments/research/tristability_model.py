"""
三稳态可行性模型
================

目标：构造最小多峰模型，验证"解域数量 = 峰的数量"

模型升级：
F(x) = a1*x*(1-x) + a2*x*(1-x)*(x-c)^2 - b

或更干净的形式（双高斯势阱）：
F(x) = -k1*(x-μ1)^2 - k2*(x-μ2)^2 + k3*(x-μ1)*(x-μ2) - b
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import os

os.makedirs('F:/hypergraph_bistability/figures/tristability', exist_ok=True)


def feasibility_unimodal(x, a, b):
    """原始单峰模型: F(x) = a*x*(1-x) - b"""
    return a * x * (1 - x) - b


def feasibility_bimodal(x, a1, a2, c, b):
    """
    双峰模型: F(x) = a1*x*(1-x) + a2*x*(1-x)*(x-c)^2 - b
    
    参数:
    - a1: 基础竞争强度
    - a2: 额外竞争强度（产生第二峰）
    - c: 峰的位置
    - b: 约束强度
    """
    term1 = a1 * x * (1 - x)
    term2 = a2 * x * (1 - x) * (x - c)**2
    return term1 + term2 - b


def feasibility_gaussian(x, mu1, mu2, sigma, b):
    """
    双高斯势阱模型（更干净）
    F(x) = -exp(-(x-μ1)^2/2σ²) - exp(-(x-μ2)^2/2σ²) - b + c*x
    
    三个竞争中心：
    - LOW (接近μ1)
    - MID (中间)
    - HIGH (接近μ2)
    """
    term1 = -np.exp(-(x - mu1)**2 / (2 * sigma**2))
    term2 = -np.exp(-(x - mu2)**2 / (2 * sigma**2))
    # 添加线性项使中间下沉
    return term1 + term2 + 0.3 * x - b


def find_roots(func, x_range, n_points=1000):
    """找函数的零点"""
    x = np.linspace(x_range[0], x_range[1], n_points)
    y = func(x)
    
    # 找符号变化的点
    roots = []
    for i in range(len(x) - 1):
        if y[i] * y[i+1] < 0:
            # 二分法找精确根
            x_root = (x[i] + x[i+1]) / 2
            for _ in range(20):
                x_root = (x[i] + x[i+1]) / 2
                if y[i] * func(x_root) < 0:
                    x[i+1] = x_root
                else:
                    i = x_root
            roots.append(x_root)
    
    return roots


def stability_analysis(roots, func):
    """分析根的稳定性: dF/dx < 0 = 稳定"""
    results = []
    for root in roots:
        # 数值导数
        h = 0.001
        dFdx = (func(root + h) - func(root - h)) / (2 * h)
        results.append({
            'x': root,
            'dFdx': dFdx,
            'stable': dFdx < 0
        })
    return results


def scan_b_for_tristability():
    """扫描 b 找三稳态"""
    print("=" * 60)
    print("扫描 b 寻找三稳态")
    print("=" * 60)
    
    # 使用双高斯模型
    mu1, mu2 = 0.25, 0.75
    sigma = 0.15
    
    results = []
    
    for b in np.linspace(0.1, 0.6, 30):
        def F(x):
            return feasibility_gaussian(x, mu1, mu2, sigma, b)
        
        roots = find_roots(F, [0, 1])
        stability = stability_analysis(roots, F)
        
        n_stable = sum(1 for s in stability if s['stable'])
        
        results.append({
            'b': b,
            'roots': len(roots),
            'n_stable': n_stable,
            'stability': stability
        })
        
        if n_stable >= 3:
            print(f"\n找到三稳态! b = {b:.3f}")
            print(f"  稳定点: {[s['x'] for s in stability if s['stable']]}")
    
    # 绘图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 子图1: 不同b的F(x)
    ax1 = axes[0, 0]
    x = np.linspace(0, 1, 200)
    
    for b in [0.2, 0.35, 0.5]:
        F = feasibility_gaussian(x, mu1, mu2, sigma, b)
        ax1.plot(x, F, label=f'b={b}')
    
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax1.set_xlabel('x')
    ax1.set_ylabel('F(x)')
    ax1.set_title('Feasibility Function F(x) with Different b')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 子图2: 稳态数量 vs b
    ax2 = axes[0, 1]
    bs = [r['b'] for r in results]
    n_stables = [r['n_stable'] for r in results]
    ax2.plot(bs, n_stables, 'o-')
    ax2.set_xlabel('b (constraint)')
    ax2.set_ylabel('Number of Stable States')
    ax2.set_title('Number of Stable States vs b')
    ax2.grid(True, alpha=0.3)
    
    # 子图3: 相图 - 不同b的最终状态
    ax3 = axes[1, 0]
    
    # 模拟从不同初始条件出发
    for b in [0.2, 0.35, 0.5]:
        final_states = []
        for x0 in np.linspace(0.05, 0.95, 30):
            traj = dynamics_simulation(x0, b, mu1, mu2, sigma, steps=50)
            final_states.append(traj[-1])
        
        ax3.plot(np.linspace(0.05, 0.95, 30), final_states, 'o-', 
                 label=f'b={b}', alpha=0.7)
    
    ax3.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax3.set_xlabel('Initial x₀')
    ax3.set_ylabel('Final x')
    ax3.set_title('Basin Attraction')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 子图4: 关键案例 - 三稳态
    ax4 = axes[1, 1]
    
    # 找一个三稳态的b
    b_tri = 0.35
    x = np.linspace(0, 1, 200)
    F = feasibility_gaussian(x, mu1, mu2, sigma, b_tri)
    ax4.plot(x, F, 'b-', linewidth=2)
    ax4.axhline(y=0, color='k', linestyle='--')
    
    # 标记稳态点
    roots = find_roots(lambda x: feasibility_gaussian(x, mu1, mu2, sigma, b_tri), [0, 1])
    stability = stability_analysis(roots, lambda x: feasibility_gaussian(x, mu1, mu2, sigma, b_tri))
    
    for s in stability:
        if s['stable']:
            ax4.plot(s['x'], 0, 'go', markersize=10, label='Stable')
        else:
            ax4.plot(s['x'], 0, 'rx', markersize=10, label='Unstable')
    
    ax4.set_xlabel('x')
    ax4.set_ylabel('F(x)')
    ax4.set_title(f'Three Stable States (b={b_tri})')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('F:/hypergraph_bistability/figures/tristability/tristability_search.png', dpi=150)
    plt.close()
    
    return results


def dynamics_simulation(x0, b, mu1, mu2, sigma, steps=50):
    """简单动力学模拟"""
    x = x0
    trajectory = [x]
    
    for _ in range(steps):
        F = feasibility_gaussian(x, mu1, mu2, sigma, b)
        # 梯度上升
        x = x + 0.1 * F
        x = np.clip(x, 0.01, 0.99)
        trajectory.append(x)
    
    return trajectory


def hysteresis_test():
    """Hysteresis 实验"""
    print("\n" + "=" * 60)
    print("Hysteresis 实验")
    print("=" * 60)
    
    mu1, mu2 = 0.25, 0.75
    sigma = 0.15
    
    # 正向扫描: b: 0.1 -> 0.6
    forward_results = []
    for b in np.linspace(0.1, 0.6, 20):
        final_states = []
        for x0 in [0.2, 0.5, 0.8]:  # 三个初始条件
            traj = dynamics_simulation(x0, b, mu1, mu2, sigma)
            final_states.append(traj[-1])
        
        forward_results.append({
            'b': b,
            'final_states': final_states
        })
    
    # 反向扫描: b: 0.6 -> 0.1
    backward_results = []
    for b in np.linspace(0.6, 0.1, 20):
        final_states = []
        for x0 in [0.2, 0.5, 0.8]:
            traj = dynamics_simulation(x0, b, mu1, mu2, sigma)
            final_states.append(traj[-1])
        
        backward_results.append({
            'b': b,
            'final_states': final_states
        })
    
    # 绘图
    plt.figure(figsize=(12, 5))
    
    # 只画一个初始条件
    plt.subplot(1, 2, 1)
    b_fwd = [r['b'] for r in forward_results]
    x_fwd = [r['final_states'][1] for r in forward_results]  # x0=0.5
    plt.plot(b_fwd, x_fwd, 'b-o', label='Forward', markersize=6)
    
    b_bwd = [r['b'] for r in backward_results]
    x_bwd = [r['final_states'][1] for r in backward_results]
    plt.plot(b_bwd, x_bwd, 'r--s', label='Backward', markersize=6)
    
    plt.xlabel('b (constraint)')
    plt.ylabel('Final x')
    plt.title('Hysteresis Test (x₀=0.5)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 画三个初始条件
    plt.subplot(1, 2, 2)
    colors = ['blue', 'green', 'red']
    markers = ['o', 's', '^']

    for idx, x0 in enumerate([0.2, 0.5, 0.8]):
        x_fwd = [r['final_states'][idx] for r in forward_results]
        plt.plot(b_fwd, x_fwd, color=colors[idx], marker=markers[idx], 
                 label=f'x0={x0}', alpha=0.7, linestyle='-')
    
    plt.xlabel('b (constraint)')
    plt.ylabel('Final x')
    plt.title('All Initial Conditions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('F:/hypergraph_bistability/figures/tristability/hysteresis.png', dpi=150)
    plt.close()
    
    # 检查 hysteresis
    print("\nHysteresis Analysis:")
    for idx, x0 in enumerate([0.2, 0.5, 0.8]):
        diff = abs(forward_results[0]['final_states'][idx] - backward_results[-1]['final_states'][idx])
        print(f"  x0 = {x0}: diff = {diff:.3f}")
    
    return forward_results, backward_results


def phase_diagram():
    """相图 - 关键图"""
    print("\n" + "=" * 60)
    print("相图分析")
    print("=" * 60)
    
    mu1, mu2 = 0.25, 0.75
    sigma = 0.15
    
    # 扫描 b 和初始条件
    b_range = np.linspace(0.1, 0.6, 30)
    x0_range = np.linspace(0.05, 0.95, 30)
    
    final_grid = np.zeros((len(b_range), len(x0_range)))
    
    for i, b in enumerate(b_range):
        for j, x0 in enumerate(x0_range):
            traj = dynamics_simulation(x0, b, mu1, mu2, sigma)
            final_grid[i, j] = traj[-1]
    
    # 绘图
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(final_grid, aspect='auto', origin='lower',
               extent=[0, 1, 0.1, 0.6], cmap='RdYlBu_r')
    plt.colorbar(label='Final x')
    plt.xlabel('Initial x₀')
    plt.ylabel('b (constraint)')
    plt.title('Phase Diagram: Final State')
    
    plt.subplot(1, 2, 2)
    # 计算每个b的三个区域占比
    regions = {'LOW': [], 'MID': [], 'HIGH': []}
    
    for i, b in enumerate(b_range):
        low = sum(1 for j in range(len(x0_range)) if final_grid[i, j] < 0.35)
        mid = sum(1 for j in range(len(x0_range)) if 0.35 <= final_grid[i, j] < 0.65)
        high = sum(1 for j in range(len(x0_range)) if final_grid[i, j] >= 0.65)
        
        total = len(x0_range)
        regions['LOW'].append(low / total)
        regions['MID'].append(mid / total)
        regions['HIGH'].append(high / total)
    
    plt.plot(b_range, regions['LOW'], 'b-', label='LOW', linewidth=2)
    plt.plot(b_range, regions['MID'], 'g-', label='MID', linewidth=2)
    plt.plot(b_range, regions['HIGH'], 'r-', label='HIGH', linewidth=2)
    
    plt.xlabel('b (constraint)')
    plt.ylabel('Fraction')
    plt.title('Basin Fraction vs b')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('F:/hypergraph_bistability/figures/tristability/phase_diagram.png', dpi=150)
    plt.close()


if __name__ == "__main__":
    # Step 1: 找三稳态
    results = scan_b_for_tristability()
    
    # Step 2: Hysteresis
    forward, backward = hysteresis_test()
    
    # Step 3: 相图
    phase_diagram()
    
    print("\n完成!")
