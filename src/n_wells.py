"""
N势阱模型 - 推广到N个势阱
=========================

测试假设：N个势阱 → N个吸引子
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import os

os.makedirs('F:/hypergraph_bistability/figures/n_wells', exist_ok=True)


def n_wells_potential(x, y, centers, h, s):
    """
    N个高斯势阱
    centers: list of (x, y) tuples
    """
    V = np.zeros_like(x, dtype=float)
    
    for (cx, cy) in centers:
        V = V - h * np.exp(-((x - cx)**2 + (y - cy)**2) / s)
    
    return V


def n_wells_force(x, y, centers, h, s):
    """计算力 F = -grad(V)"""
    fx = np.zeros_like(x, dtype=float)
    fy = np.zeros_like(x, dtype=float)
    
    for (cx, cy) in centers:
        r2 = (x - cx)**2 + (y - cy)**2
        exp_term = h * np.exp(-r2 / s)
        fx = fx + (2 * (x - cx) / s) * exp_term
        fy = fy + (2 * (y - cy) / s) * exp_term
    
    return fx, fy


def dynamics(state, t, centers, h, s, damping):
    x, y = state
    fx, fy = n_wells_force(x, y, centers, h, s)
    return [damping * fx, damping * fy]


def find_attractors(centers, h, s, damping, n_init=64, t_max=10):
    """找到所有吸引子"""
    # 初始点网格
    initials = []
    for x0 in np.linspace(0.1, 0.9, int(np.sqrt(n_init))):
        for y0 in np.linspace(0.1, 0.9, int(np.sqrt(n_init))):
            initials.append((x0, y0))
    
    t = np.linspace(0, t_max, 500)
    
    final_positions = []
    for x0, y0 in initials:
        try:
            sol = odeint(dynamics, [x0, y0], t, args=(centers, h, s, damping))
            final = (sol[-1, 0], sol[-1, 1])
            # 检查是否在有效范围内
            if 0 <= final[0] <= 1 and 0 <= final[1] <= 1:
                final_positions.append(final)
        except:
            pass
    
    # 聚类找到唯一吸引子
    unique_attractors = []
    threshold = 0.15  # 聚类阈值
    
    for fp in final_positions:
        is_new = True
        for up in unique_attractors:
            dist = np.sqrt((fp[0] - up[0])**2 + (fp[1] - up[1])**2)
            if dist < threshold:
                is_new = False
                break
        if is_new:
            unique_attractors.append(fp)
    
    return unique_attractors, len(initials), len(final_positions)


def generate_circle_centers(n, radius=0.35, center=(0.5, 0.5)):
    """生成圆形分布的势阱中心"""
    centers = []
    for i in range(n):
        angle = 2 * np.pi * i / n - np.pi / 2  # 从顶部开始
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        centers.append((x, y))
    return centers


def test_n_wells():
    """测试 N = 2 到 10"""
    print("=" * 60)
    print("N-Wells Systematic Test")
    print("=" * 60)
    
    h = 1.0  # 势阱深度
    s = 0.08  # 势阱宽度
    damping = 0.5
    
    results = []
    
    for n in range(2, 11):
        centers = generate_circle_centers(n)
        attractors, n_init, n_final = find_attractors(centers, h, s, damping)
        
        n_attractors = len(attractors)
        efficiency = n_attractors / n if n > 0 else 0
        
        results.append({
            'n': n,
            'n_attractors': n_attractors,
            'efficiency': efficiency,
            'attractors': attractors
        })
        
        print(f"\nN = {n}:")
        print(f"  Centers: {n}")
        print(f"  Attractors found: {n_attractors}")
        print(f"  Efficiency: {efficiency:.2%}")
        
        if n_attractors > 0:
            for i, at in enumerate(attractors):
                print(f"    #{i+1}: ({at[0]:.2f}, {at[1]:.2f})")
    
    return results


def visualize_results(results):
    """可视化结果"""
    # 1. N vs Attractors 曲线
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    ax1 = axes[0, 0]
    n_values = [r['n'] for r in results]
    n_attractors = [r['n_attractors'] for r in results]
    efficiencies = [r['efficiency'] for r in results]
    
    ax1.plot(n_values, n_values, 'k--', label='Ideal (N attractors)', alpha=0.5)
    ax1.scatter(n_values, n_attractors, s=100, c='blue', label='Actual', zorder=5)
    ax1.set_xlabel('Number of Wells (N)')
    ax1.set_ylabel('Number of Attractors')
    ax1.set_title('N Wells → N Attractors?')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(n_values)
    
    # 2. Efficiency 曲线
    ax2 = axes[0, 1]
    ax2.bar(n_values, efficiencies, color='steelblue', edgecolor='black')
    ax2.axhline(y=1.0, color='red', linestyle='--', label='100% efficiency')
    ax2.set_xlabel('Number of Wells (N)')
    ax2.set_ylabel('Efficiency (Attractors / N)')
    ax2.set_title('Attractor Efficiency')
    ax2.set_xticks(n_values)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.2)
    
    # 3. 关键案例可视化
    cases_to_show = [3, 5, 7, 9]
    
    for idx, n in enumerate(cases_to_show):
        if idx < 4:
            ax = axes[1, idx // 2] if idx < 2 else axes[1, idx - 2]
            if idx >= 2:
                ax = axes[1, 1]
        
    # 重新布局
    for idx, n in enumerate(cases_to_show):
        ax = axes[1, idx % 2]
        
        centers = generate_circle_centers(n)
        
        # 绘制势阱位置
        for cx, cy in centers:
            ax.add_patch(plt.Circle((cx, cy), 0.03, color='red', alpha=0.5))
        
        # 绘制吸引子
        r = results[n - 2]
        for at in r['attractors']:
            ax.plot(at[0], at[1], 'go', markersize=12, markeredgecolor='black')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_title(f'N={n}: {r["n_attractors"]} attractors')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('F:/hypergraph_bistability/figures/n_wells/n_wells_summary.png', dpi=150)
    plt.close()
    
    print("\nFigure saved!")


def test_parameter_variations():
    """测试不同参数"""
    print("\n" + "=" * 60)
    print("Parameter Variations")
    print("=" * 60)
    
    variations = [
        {'h': 0.5, 's': 0.08, 'name': 'Shallow'},
        {'h': 2.0, 's': 0.08, 'name': 'Deep'},
        {'h': 1.0, 's': 0.05, 'name': 'Narrow'},
        {'h': 1.0, 's': 0.15, 'name': 'Wide'},
    ]
    
    n = 5  # 测试 N=5
    
    for var in variations:
        centers = generate_circle_centers(n)
        attractors, _, _ = find_attractors(centers, var['h'], var['s'], 0.5)
        
        print(f"\n{var['name']} (h={var['h']}, s={var['s']}): {len(attractors)} attractors")


def explore_limits():
    """探索极限"""
    print("\n" + "=" * 60)
    print("Exploring Limits")
    print("=" * 60)
    
    # 测试更大的 N
    for n in [10, 15, 20, 30]:
        centers = generate_circle_centers(n, radius=0.4)
        attractors, _, _ = find_attractors(centers, 1.0, 0.06, 0.5, n_init=100)
        print(f"N={n}: {len(attractors)} attractors")


if __name__ == "__main__":
    results = test_n_wells()
    visualize_results(results)
    test_parameter_variations()
    explore_limits()
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
