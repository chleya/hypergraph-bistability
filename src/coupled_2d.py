"""
更强的2D模型：耦合双稳态系统
===========================

尝试两个耦合的双稳态系统来产生三稳态
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import os

os.makedirs('F:/hypergraph_bistability/figures/2d_model', exist_ok=True)


def coupled_bistable(state, t, a, b, c):
    """
    耦合双稳态系统:
    dx/dt = x*(1-x)*(x - a) + c*(y - x)
    dy/dt = y*(1-y)*(y - b) + c*(x - y)
    
    每个变量x, y都有自己的双稳态势阱
    c是耦合强度
    """
    x, y = state
    
    dxdt = x * (1 - x) * (x - a) + c * (y - x)
    dydt = y * (1 - y) * (y - b) + c * (x - y)
    
    return [dxdt, dydt]


def find_fixed_points_brute(a, b, c, resolution=20):
    """暴力搜索固定点"""
    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)
    
    fixed = []
    
    for x0 in x:
        for y0 in y:
            dxdt, dydt = coupled_bistable([x0, y0], 0, a, b, c)
            if abs(dxdt) < 0.01 and abs(dydt) < 0.01:
                # 检查是否已存在
                is_new = True
                for fx, fy in fixed:
                    if abs(fx - x0) < 0.05 and abs(fy - y0) < 0.05:
                        is_new = False
                        break
                if is_new:
                    fixed.append((x0, y0))
    
    return fixed


def analyze_jacobian(fp, a, b, c):
    """分析Jacobian矩阵"""
    x, y = fp
    
    # 偏导数
    # dx/dt = x(1-x)(x-a) + c(y-x)
    # d(dx/dt)/dx = (1-2x)(x-a) + x(1-x)
    # = (1-2x)(x-a) + x - x²
    # = (x-a) - 2x(x-a) + x - x²
    # = x - a - 2x² + 2ax + x - x²
    # = -a - 3x² + (2a+2)x
    J11 = (1 - 2*x) * (x - a) + x * (1 - x)
    J12 = c
    J21 = c
    J22 = (1 - 2*y) * (y - b) + y * (1 - y)
    
    trace = J11 + J22
    det = J11 * J22 - J12 * J21
    
    # 特征值
    if trace**2 - 4*det >= 0:
        lambda1 = (trace + np.sqrt(max(0, trace**2 - 4*det))) / 2
        lambda2 = (trace - np.sqrt(max(0, trace**2 - 4*det))) / 2
        eigenvals = [lambda1, lambda2]
    else:
        real = trace / 2
        imag = np.sqrt(max(0, 4*det - trace**2)) / 2
        eigenvals = [complex(real, imag), complex(real, -imag)]
    
    # 稳定性
    if all(np.real(e) < 0 for e in eigenvals):
        stability = "stable"
    elif any(np.real(e) > 0 for e in eigenvals):
        stability = "unstable"
    else:
        stability = "saddle"
    
    return stability, trace, det, eigenvals


def scan_coupled():
    """扫描耦合双稳态系统"""
    print("=" * 60)
    print("Scanning Coupled Bistable System")
    print("=" * 60)
    
    results = []
    
    # 搜索参数空间
    for a in np.linspace(-0.3, 0.3, 8):
        for b in np.linspace(-0.3, 0.3, 8):
            for c in np.linspace(0.05, 0.5, 10):
                fps = find_fixed_points_brute(a, b, c, resolution=15)
                
                n_stable = 0
                stabilities = []
                for fp in fps:
                    stab, trace, det, ev = analyze_jacobian(fp, a, b, c)
                    stabilities.append((fp, stab))
                    if stab == "stable":
                        n_stable += 1
                
                if n_stable >= 2:
                    results.append({
                        'a': a, 'b': b, 'c': c,
                        'n_fp': len(fps),
                        'n_stable': n_stable,
                        'fps': stabilities
                    })
    
    print(f"\nFound {len(results)} parameter sets with 2+ stable points")
    
    # 按稳定点数量排序
    results.sort(key=lambda x: x['n_stable'], reverse=True)
    
    # 打印前10个
    for r in results[:10]:
        print(f"\na={r['a']:.2f}, b={r['b']:.2f}, c={r['c']:.2f}: {r['n_stable']} stable out of {r['n_fp']} fixed points")
        for fp, stab in r['fps']:
            print(f"  ({fp[0]:.2f}, {fp[1]:.2f}): {stab}")
    
    return results


def plot_phase_portrait(a, b, c):
    """绘制相图"""
    x = np.linspace(-0.1, 1.1, 40)
    y = np.linspace(-0.1, 1.1, 40)
    X, Y = np.meshgrid(x, y)
    
    U = np.zeros_like(X)
    V = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            dxdt, dydt = coupled_bistable([X[i,j], Y[i,j]], 0, a, b, c)
            U[i,j] = dxdt
            V[i,j] = dydt
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    speed = np.sqrt(U**2 + V**2)
    stream = ax.streamplot(x, y, U, V, color=speed, cmap='viridis', density=1.5)
    
    # 固定点
    fps = find_fixed_points_brute(a, b, c, resolution=25)
    for fp in fps:
        stab, trace, det, ev = analyze_jacobian(fp, a, b, c)
        if stab == "stable":
            ax.plot(fp[0], fp[1], 'go', markersize=15, markeredgecolor='black', 
                   markeredgewidth=2, zorder=10)
        elif stab == "saddle":
            ax.plot(fp[0], fp[1], 'rx', markersize=15, markeredgecolor='black',
                   markeredgewidth=2, zorder=10)
    
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Coupled Bistable: a={a:.2f}, b={b:.2f}, c={c:.2f}')
    ax.grid(True, alpha=0.3)
    plt.colorbar(stream.lines, ax=ax, label='Speed')
    
    return fig


def plot_trajectories(a, b, c):
    """绘制轨迹"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 初始条件
    initials = []
    for x0 in np.linspace(0.05, 0.95, 6):
        for y0 in np.linspace(0.05, 0.95, 6):
            initials.append((x0, y0))
    
    t = np.linspace(0, 30, 500)
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(initials)))
    
    for i, (x0, y0) in enumerate(initials):
        sol = odeint(coupled_bistable, [x0, y0], t, args=(a, b, c))
        ax.plot(sol[:, 0], sol[:, 1], color=colors[i], alpha=0.6, linewidth=1.5)
        ax.plot(sol[0, 0], sol[0, 1], 'o', color=colors[i], markersize=6)
        ax.plot(sol[-1, 0], sol[-1, 1], 's', color=colors[i], markersize=8)
    
    # 标记稳定点
    fps = find_fixed_points_brute(a, b, c, resolution=25)
    for fp in fps:
        stab, _, _, _ = analyze_jacobian(fp, a, b, c)
        if stab == "stable":
            ax.plot(fp[0], fp[1], 'go', markersize=15, markeredgecolor='black', 
                   markeredgewidth=2, zorder=10)
    
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Trajectories: a={a:.2f}, b={b:.2f}, c={c:.2f}')
    ax.grid(True, alpha=0.3)
    
    return fig


if __name__ == "__main__":
    # 扫描找多稳态
    results = scan_coupled()
    
    # 绘制最佳结果的相图
    if results:
        best = results[0]
        print(f"\n\nBest result: a={best['a']}, b={best['b']}, c={best['c']}, n_stable={best['n_stable']}")
        
        fig1 = plot_phase_portrait(best['a'], best['b'], best['c'])
        fig1.savefig('F:/hypergraph_bistability/figures/2d_model/coupled_best.png', dpi=150)
        plt.close()
        
        fig2 = plot_trajectories(best['a'], best['b'], best['c'])
        fig2.savefig('F:/hypergraph_bistability/figures/2d_model/coupled_trajectories.png', dpi=150)
        plt.close()
    
    # 也测试一些典型参数
    test_cases = [
        (0.0, 0.0, 0.1),
        (0.0, 0.0, 0.2),
        (-0.1, 0.1, 0.15),
    ]
    
    for a, b, c in test_cases:
        n_fp = len(find_fixed_points_brute(a, b, c))
        print(f"\na={a}, b={b}, c={c}: {n_fp} fixed points")
        
        if n_fp >= 3:
            fig = plot_phase_portrait(a, b, c)
            fig.savefig(f'F:/hypergraph_bistability/figures/2d_model/test_a{a}_b{b}_c{c}.png'.replace('.', 'p').replace('-', 'm'), dpi=150)
            plt.close()
    
    print("\nDone!")
