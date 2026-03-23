"""
N势阱 - 不同的排列方式
======================

发现：圆形排列 → 所有收敛到中心
需要不同的排列方式
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import random
import os

os.makedirs('F:/hypergraph_bistability/figures/n_wells', exist_ok=True)


def n_wells_force(x, y, centers, h, s):
    """计算力"""
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


def find_attractors(centers, h, s, damping, n_init=100, t_max=15):
    """找到所有吸引子"""
    initials = []
    for x0 in np.linspace(0.05, 0.95, int(np.sqrt(n_init))):
        for y0 in np.linspace(0.05, 0.95, int(np.sqrt(n_init))):
            initials.append((x0, y0))
    
    t = np.linspace(0, t_max, 600)
    
    final_positions = []
    for x0, y0 in initials:
        try:
            sol = odeint(dynamics, [x0, y0], t, args=(centers, h, s, damping))
            final = (sol[-1, 0], sol[-1, 1])
            if 0 <= final[0] <= 1 and 0 <= final[1] <= 1:
                final_positions.append(final)
        except:
            pass
    
    # 聚类
    unique_attractors = []
    threshold = 0.12
    
    for fp in final_positions:
        is_new = True
        for up in unique_attractors:
            dist = np.sqrt((fp[0] - up[0])**2 + (fp[1] - up[1])**2)
            if dist < threshold:
                is_new = False
                break
        if is_new:
            unique_attractors.append(fp)
    
    return unique_attractors


def test_arrangement(n, arrangement_type):
    """测试不同的排列方式"""
    
    if arrangement_type == 'circle':
        # 圆形 - 之前试过，只能得到1个
        centers = []
        radius = 0.35
        for i in range(n):
            angle = 2 * np.pi * i / n - np.pi / 2
            x = 0.5 + radius * np.cos(angle)
            y = 0.5 + radius * np.sin(angle)
            centers.append((x, y))
    
    elif arrangement_type == 'grid':
        # 网格排列
        centers = []
        side = int(np.ceil(np.sqrt(n)))
        spacing = 0.8 / side
        offset = 0.1 + spacing / 2
        count = 0
        for i in range(side):
            for j in range(side):
                if count < n:
                    x = offset + i * spacing
                    y = offset + j * spacing
                    centers.append((x, y))
                    count += 1
    
    elif arrangement_type == 'random':
        # 随机分布
        centers = []
        for _ in range(n):
            x = 0.2 + random.random() * 0.6
            y = 0.2 + random.random() * 0.6
            centers.append((x, y))
    
    elif arrangement_type == 'ring':
        # 双环 - 内环和外环
        centers = []
        n_inner = n // 2
        n_outer = n - n_inner
        for i in range(n_inner):
            angle = 2 * np.pi * i / n_inner
            x = 0.5 + 0.15 * np.cos(angle)
            y = 0.5 + 0.15 * np.sin(angle)
            centers.append((x, y))
        for i in range(n_outer):
            angle = 2 * np.pi * i / n_outer + np.pi / n_outer
            x = 0.5 + 0.35 * np.cos(angle)
            y = 0.5 + 0.35 * np.sin(angle)
            centers.append((x, y))
    
    elif arrangement_type == 'triangle':
        # 三角形网格
        centers = []
        rows = int(np.ceil(np.sqrt(2*n)))
        for row in range(rows):
            for col in range(row + 1):
                if len(centers) < n:
                    x = 0.15 + col * 0.7 / row + 0.5 if row > 0 else 0.5
                    y = 0.15 + row * 0.7 / rows
                    if row > 0:
                        x = 0.15 + col * (0.7 / row)
                    centers.append((x, y))
    
    elif arrangement_type == 'line':
        # 直线排列
        centers = []
        for i in range(n):
            x = 0.15 + i * 0.7 / (n - 1) if n > 1 else 0.5
            y = 0.5
            centers.append((x, y))
    
    elif arrangement_type == 'vertices':
        # 正多边形顶点
        centers = []
        radius = 0.35
        for i in range(n):
            angle = 2 * np.pi * i / n - np.pi / 2
            # 稍微随机化半径，避免完美对称
            r = radius * (0.9 + 0.2 * random.random())
            x = 0.5 + r * np.cos(angle)
            y = 0.5 + r * np.sin(angle)
            centers.append((x, y))
    
    else:
        centers = []
    
    # 运行模拟
    h = 1.0
    s = 0.06
    damping = 0.5
    
    attractors = find_attractors(centers, h, s, damping)
    
    return centers, attractors


def main():
    print("=" * 60)
    print("Testing Different Arrangements")
    print("=" * 60)
    
    # 测试不同的 N 和排列方式
    test_cases = [
        (3, 'circle'),
        (3, 'vertices'),
        (3, 'random'),
        (5, 'circle'),
        (5, 'grid'),
        (5, 'random'),
        (5, 'ring'),
        (6, 'grid'),
        (6, 'ring'),
        (7, 'vertices'),
        (9, 'vertices'),
        (10, 'grid'),
    ]
    
    random.seed(42)  # 可重复
    
    results = []
    
    for n, arrangement in test_cases:
        centers, attractors = test_arrangement(n, arrangement)
        
        results.append({
            'n': n,
            'arrangement': arrangement,
            'n_attractors': len(attractors),
            'efficiency': len(attractors) / n if n > 0 else 0,
            'centers': centers,
            'attractors': attractors
        })
        
        print(f"\nN={n}, {arrangement}:")
        print(f"  Attractors: {len(attractors)}")
        
        # 打印吸引子位置
        for i, at in enumerate(attractors):
            print(f"    #{i+1}: ({at[0]:.2f}, {at[1]:.2f})")
    
    # 找到最佳组合
    best = max(results, key=lambda x: x['efficiency'])
    print(f"\n{'='*60}")
    print(f"Best: N={best['n']}, {best['arrangement']} with {best['n_attractors']} attractors ({best['efficiency']:.1%})")
    
    # 详细可视化最佳案例
    visualize_best(results)
    
    return results


def visualize_best(results):
    """可视化最佳案例"""
    # 找到有多个吸引子的案例
    multi_attractor = [r for r in results if r['n_attractors'] >= 2]
    
    if not multi_attractor:
        print("No multi-attractor cases found!")
        return
    
    n_cases = min(4, len(multi_attractor))
    fig, axes = plt.subplots(1, n_cases, figsize=(5*n_cases, 5))
    if n_cases == 1:
        axes = [axes]
    
    for idx, r in enumerate(multi_attractor[:n_cases]):
        ax = axes[idx]
        
        # 绘制势阱
        for cx, cy in r['centers']:
            ax.add_patch(plt.Circle((cx, cy), 0.025, color='red', alpha=0.6))
        
        # 绘制吸引子
        for at in r['attractors']:
            ax.plot(at[0], at[1], 'go', markersize=15, markeredgecolor='black', markeredgewidth=2)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_title(f"N={r['n']}, {r['arrangement']}\n{r['n_attractors']} attractors")
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('F:/hypergraph_bistability/figures/n_wells/best_arrangements.png', dpi=150)
    plt.close()
    
    print(f"\nFigure saved!")


def systematic_search():
    """系统性搜索最佳排列"""
    print("\n" + "=" * 60)
    print("Systematic Search for 5 Wells")
    print("=" * 60)
    
    # 对于 N=5，尝试不同的随机种子
    random.seed(None)
    
    best_result = None
    best_count = 0
    
    for trial in range(50):
        centers = []
        for _ in range(5):
            x = 0.2 + random.random() * 0.6
            y = 0.2 + random.random() * 0.6
            centers.append((x, y))
        
        h, s, damping = 1.0, 0.05, 0.5
        attractors = find_attractors(centers, h, s, damping, n_init=150)
        
        if len(attractors) > best_count:
            best_count = len(attractors)
            best_result = {
                'centers': centers,
                'attractors': attractors,
                'trial': trial
            }
            print(f"Trial {trial}: {len(attractors)} attractors - NEW BEST!")
    
    print(f"\nBest result: {best_count} attractors")
    
    if best_result and best_count > 1:
        # 可视化最佳结果
        fig, ax = plt.subplots(figsize=(8, 8))
        
        for cx, cy in best_result['centers']:
            ax.add_patch(plt.Circle((cx, cy), 0.025, color='red', alpha=0.6, label='Wells'))
        
        for at in best_result['attractors']:
            ax.plot(at[0], at[1], 'go', markersize=15, markeredgecolor='black', 
                   markeredgewidth=2, label='Attractors')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_title(f"Best Random Arrangement: {best_count} Attractors")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('F:/hypergraph_bistability/figures/n_wells/best_random.png', dpi=150)
        plt.close()


if __name__ == "__main__":
    results = main()
    systematic_search()
    print("\nDone!")
