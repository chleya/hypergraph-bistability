"""
更直接的三稳态模型
==================

使用组合函数：基础竞争 + 周期调制
"""

import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('F:/hypergraph_bistability/figures/tristability', exist_ok=True)


def combined_F(x, a, b, c):
    """
    组合函数: F(x) = a*x*(1-x) + b*sin(3*pi*x) - c
    
    思想：
    - a*x*(1-x): 基础竞争（单峰）
    - b*sin(3*pi*x): 添加3个周期
    - c: 约束强度
    """
    return a * x * (1 - x) + b * np.sin(3 * np.pi * x) - c


def find_roots(F, x_range, n=2000):
    """找零点"""
    x = np.linspace(x_range[0], x_range[1], n)
    y = F(x)
    
    roots = []
    for i in range(len(x) - 1):
        if y[i] * y[i+1] < 0:
            # 二分法精确定位
            xl, xr = x[i], x[i+1]
            for _ in range(30):
                xm = (xl + xr) / 2
                if y[i] * F(xm) < 0:
                    xr = xm
                else:
                    xl = xm
            roots.append(xm)
    return roots


def check_stability(root, F):
    """检查稳定性"""
    h = 0.0001
    dF = (F(root + h) - F(root - h)) / (2 * h)
    return dF < 0


def scan_combined():
    """扫描组合函数参数"""
    print("=" * 60)
    print("Scanning Combined Function Parameters")
    print("=" * 60)
    
    results = []
    
    for a in np.linspace(0.5, 3.0, 15):
        for b in np.linspace(0.3, 2.0, 15):
            for c in np.linspace(0.1, 0.8, 20):
                def F(x):
                    return combined_F(x, a, b, c)
                
                roots = find_roots(F, [0, 1])
                stable = [r for r in roots if check_stability(r, F)]
                
                # 只关心在[0.05, 0.95]范围内的稳定点
                in_range = [r for r in stable if 0.05 < r < 0.95]
                
                if len(in_range) >= 3:
                    results.append({
                        'a': a, 'b': b, 'c': c,
                        'stable': in_range,
                        'n': len(in_range)
                    })
    
    print(f"\nFound {len(results)} parameter sets with 3+ stable states")
    
    if results:
        # 按稳定点数量排序
        results.sort(key=lambda x: x['n'], reverse=True)
        
        # 打印前5个
        for i, r in enumerate(results[:5]):
            print(f"\n{i+1}. a={r['a']:.2f}, b={r['b']:.2f}, c={r['c']:.2f}")
            print(f"   Stable points: {r['stable']}")
        
        # 画前3个
        for i, r in enumerate(results[:3]):
            plot_combined(r, i+1)
    
    return results


def plot_combined(params, idx):
    """绘制组合函数"""
    a, b, c = params['a'], params['b'], params['c']
    
    x = np.linspace(0, 1, 500)
    y = combined_F(x, a, b, c)
    
    def F(x):
        return combined_F(x, a, b, c)
    
    roots = find_roots(F, [0, 1])
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', linewidth=2)
    plt.axhline(y=0, color='k', linestyle='--')
    plt.axvline(x=0, color='gray', linestyle=':', alpha=0.3)
    plt.axvline(x=1, color='gray', linestyle=':', alpha=0.3)
    
    # 标记稳态点
    for r in roots:
        if check_stability(r, F):
            plt.plot(r, 0, 'go', markersize=15, markeredgecolor='black', markeredgewidth=2)
        else:
            plt.plot(r, 0, 'rx', markersize=15, markeredgecolor='black', markeredgewidth=2)
    
    plt.xlabel('x (cohesion)', fontsize=12)
    plt.ylabel('F(x) (feasibility)', fontsize=12)
    plt.title(f'Tristability: a={a:.2f}, b={b:.2f}, c={c:.2f}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xlim(-0.02, 1.02)
    plt.savefig(f'F:/hypergraph_bistability/figures/tristability/combined_{idx}.png', dpi=150)
    plt.close()


def phase_diagram_for_best(params):
    """为最佳参数绘制相图"""
    a, b, c = params['a'], params['b'], params['c']
    
    print(f"\nDrawing phase diagram for best params: a={a}, b={b}, c={c}")
    
    # 扫描c (constraint) 作为控制参数
    c_values = np.linspace(0.1, 0.8, 25)
    x0_values = np.linspace(0.02, 0.98, 30)
    
    final_grid = np.zeros((len(c_values), len(x0_values)))
    
    def simulate(x0, c, steps=80):
        x = x0
        for _ in range(steps):
            F = combined_F(x, a, b, c)
            x = x + 0.15 * F  # 步长
            x = np.clip(x, 0.001, 0.999)
        return x
    
    for i, c in enumerate(c_values):
        for j, x0 in enumerate(x0_values):
            final_grid[i, j] = simulate(x0, c)
    
    # 绘制相图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 相图
    ax1 = axes[0]
    im = ax1.imshow(final_grid, aspect='auto', origin='lower',
                    extent=[0, 1, c_values[0], c_values[-1]], 
                    cmap='RdYlBu_r')
    ax1.set_xlabel('Initial x₀', fontsize=12)
    ax1.set_ylabel('c (constraint)', fontsize=12)
    ax1.set_title('Phase Diagram: Final State', fontsize=14)
    plt.colorbar(im, ax=ax1, label='Final x')
    
    # 统计各区域占比
    ax2 = axes[1]
    
    low_frac = []
    mid_frac = []
    high_frac = []
    
    for i, c in enumerate(c_values):
        low = sum(1 for j in range(len(x0_values)) if final_grid[i, j] < 0.3)
        mid = sum(1 for j in range(len(x0_values)) if 0.3 <= final_grid[i, j] < 0.7)
        high = sum(1 for j in range(len(x0_values)) if final_grid[i, j] >= 0.7)
        
        total = len(x0_values)
        low_frac.append(low / total)
        mid_frac.append(mid / total)
        high_frac.append(high / total)
    
    ax2.plot(c_values, low_frac, 'b-o', label='LOW', markersize=4)
    ax2.plot(c_values, mid_frac, 'g-s', label='MID', markersize=4)
    ax2.plot(c_values, high_frac, 'r-^', label='HIGH', markersize=4)
    ax2.set_xlabel('c (constraint)', fontsize=12)
    ax2.set_ylabel('Fraction', fontsize=12)
    ax2.set_title('Basin Fraction vs c', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('F:/hypergraph_bistability/figures/tristability/phase_best.png', dpi=150)
    plt.close()


if __name__ == "__main__":
    # 扫描参数
    results = scan_combined()
    
    if results:
        # 为最佳结果画相图
        phase_diagram_for_best(results[0])
    
    print("\nDone!")
