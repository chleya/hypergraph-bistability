"""
2D 双稳态/三稳态模型实验
========================

模型:
dx/dt = x*(1-x)*(alpha - y)
dy/dt = beta*(x - y)

这个系统在 x 方向有双稳态机制 (x*(1-x))，y 是全局约束反馈
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import os

os.makedirs('F:/hypergraph_bistability/figures/2d_model', exist_ok=True)


def model_2d(state, t, alpha, beta):
    """2D dynamics"""
    x, y = state
    
    dxdt = x * (1 - x) * (alpha - y)
    dydt = beta * (x - y)
    
    return [dxdt, dydt]


def find_fixed_points(alpha, beta):
    """找固定点"""
    # 固定点条件: dx/dt = 0, dy/dt = 0
    # x*(1-x)*(alpha-y) = 0 => x=0, x=1, 或 y=alpha
    # beta*(x-y) = 0 => x=y
    
    fixed_points = []
    
    # 情况1: x = 0, x = y => (0, 0)
    fixed_points.append((0.0, 0.0))
    
    # 情况2: x = 1, x = y => (1, 1)
    fixed_points.append((1.0, 1.0))
    
    # 情况3: y = alpha, x = y => (alpha, alpha)
    if 0 <= alpha <= 1:
        fixed_points.append((alpha, alpha))
    
    return fixed_points


def analyze_stability(fp, alpha, beta):
    """分析固定点稳定性"""
    x, y = fp
    
    # Jacobian矩阵
    # dx/dt = x*(1-x)*(alpha-y)
    # d(dx/dt)/dx = (1-2x)*(alpha-y)
    # d(dx/dt)/dy = -x*(1-x)
    # dy/dt = beta*(x-y)
    # d(dy/dt)/dx = beta
    # d(dy/dt)/dy = -beta
    
    J11 = (1 - 2*x) * (alpha - y)
    J12 = -x * (1 - x)
    J21 = beta
    J22 = -beta
    
    # 特征值
    trace = J11 + J22
    det = J11 * J22 - J12 * J21
    
    eigenvalues = []
    if trace**2 - 4*det >= 0:
        lambda1 = (trace + np.sqrt(trace**2 - 4*det)) / 2
        lambda2 = (trace - np.sqrt(trace**2 - 4*det)) / 2
        eigenvalues = [lambda1, lambda2]
    else:
        # 复数特征值
        real = trace / 2
        imag = np.sqrt(4*det - trace**2) / 2
        eigenvalues = [complex(real, imag), complex(real, -imag)]
    
    # 稳定性判断
    # 稳定: 两个特征值的实部都 < 0
    # 不稳定: 至少一个特征值实部 > 0
    # 鞍点: 一个稳定一个不稳定
    
    if all(np.real(e) < 0 for e in eigenvalues):
        stability = "stable"
    elif any(np.real(e) > 0 for e in eigenvalues):
        stability = "unstable"
    else:
        stability = "saddle"
    
    return stability, trace, det


def parameter_scan():
    """参数扫描找三稳态"""
    print("=" * 60)
    print("2D Model Parameter Scan")
    print("=" * 60)
    
    results = []
    
    for alpha in np.linspace(0.1, 2.0, 20):
        for beta in np.linspace(0.1, 2.0, 20):
            fps = find_fixed_points(alpha, beta)
            
            stabilities = []
            for fp in fps:
                stab, trace, det = analyze_stability(fp, alpha, beta)
                stabilities.append((fp, stab))
            
            # 统计稳定点数量
            n_stable = sum(1 for _, s in stabilities if s == "stable")
            
            if n_stable >= 3:
                results.append({
                    'alpha': alpha,
                    'beta': beta,
                    'fixed_points': fps,
                    'stabilities': stabilities,
                    'n_stable': n_stable
                })
                print(f"\nFound {n_stable} stable points: alpha={alpha:.2f}, beta={beta:.2f}")
                for fp, stab in stabilities:
                    print(f"  {fp}: {stab}")
    
    print(f"\nTotal found: {len(results)}")
    return results


def phase_portrait(alpha, beta, title="Phase Portrait"):
    """绘制相图"""
    # 创建网格
    x = np.linspace(0, 1.2, 30)
    y = np.linspace(0, 1.2, 30)
    X, Y = np.meshgrid(x, y)
    
    # 计算速度场
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            dxdt, dydt = model_2d([X[i,j], Y[i,j]], 0, alpha, beta)
            U[i,j] = dxdt
            V[i,j] = dydt
    
    # 绘制
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 流场
    speed = np.sqrt(U**2 + V**2)
    stream = ax.streamplot(x, y, U, V, color=speed, cmap='viridis', density=1.5)
    
    # 固定点
    fps = find_fixed_points(alpha, beta)
    for fp in fps:
        stab, _, _ = analyze_stability(fp, alpha, beta)
        if stab == "stable":
            ax.plot(fp[0], fp[1], 'go', markersize=15, markeredgecolor='black', 
                   markeredgewidth=2, label=f'Stable ({fp[0]:.2f}, {fp[1]:.2f})')
        elif stab == "saddle":
            ax.plot(fp[0], fp[1], 'rx', markersize=15, markeredgecolor='black',
                   markeredgewidth=2, label=f'Saddle ({fp[0]:.2f}, {fp[1]:.2f})')
        else:
            ax.plot(fp[0], fp[1], 'b^', markersize=10)
    
    ax.set_xlim(0, 1.2)
    ax.set_ylim(0, 1.2)
    ax.set_xlabel('x (local cohesion)')
    ax.set_ylabel('y (global constraint)')
    ax.set_title(f'{title}\nα={alpha:.2f}, β={beta:.2f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.colorbar(stream.lines, ax=ax, label='Speed')
    
    return fig


def trajectory_simulation(alpha, beta, initial_conditions):
    """轨迹模拟"""
    t = np.linspace(0, 20, 500)
    
    trajectories = []
    for x0, y0 in initial_conditions:
        sol = odeint(model_2d, [x0, y0], t, args=(alpha, beta))
        trajectories.append(sol)
    
    return t, trajectories


def plot_trajectories(alpha, beta):
    """绘制多条轨迹"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 初始条件网格
    x0_vals = np.linspace(0.01, 0.99, 8)
    y0_vals = np.linspace(0.01, 0.99, 8)
    initial_conditions = [(x0, y0) for x0 in x0_vals for y0 in y0_vals]
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(initial_conditions)))
    
    t, trajectories = trajectory_simulation(alpha, beta, initial_conditions)
    
    for i, traj in enumerate(trajectories):
        ax.plot(traj[:, 0], traj[:, 1], color=colors[i], alpha=0.6, linewidth=1.5)
        ax.plot(traj[0, 0], traj[0, 1], 'o', color=colors[i], markersize=8)  # start
        ax.plot(traj[-1, 0], traj[-1, 1], 's', color=colors[i], markersize=10)  # end
    
    # 标记固定点
    fps = find_fixed_points(alpha, beta)
    for fp in fps:
        stab, _, _ = analyze_stability(fp, alpha, beta)
        if stab == "stable":
            ax.plot(fp[0], fp[1], 'go', markersize=15, markeredgecolor='black', 
                   markeredgewidth=2, zorder=10)
    
    ax.set_xlim(0, 1.1)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel('x (local cohesion)')
    ax.set_ylabel('y (global constraint)')
    ax.set_title(f'Trajectories from Multiple Initial Conditions\nα={alpha:.2f}, β={beta:.2f}')
    ax.grid(True, alpha=0.3)
    
    return fig


def find_tristability():
    """系统地找三稳态参数"""
    print("\n" + "=" * 60)
    print("Finding Parameters with 3+ Stable Fixed Points")
    print("=" * 60)
    
    # 先分析几个典型参数
    test_cases = [
        (0.5, 0.5),
        (0.5, 1.0),
        (1.0, 0.5),
        (1.0, 1.0),
        (1.5, 0.5),
        (1.5, 1.0),
        (0.8, 0.8),
    ]
    
    for alpha, beta in test_cases:
        print(f"\nα={alpha}, β={beta}:")
        fps = find_fixed_points(alpha, beta)
        
        for fp in fps:
            stab, trace, det = analyze_stability(fp, alpha, beta)
            print(f"  ({fp[0]:.2f}, {fp[1]:.2f}): {stab} (trace={trace:.2f}, det={det:.2f})")
    
    # 找三固定点的情况（当0<α<1时，有3个潜在固定点）
    print("\n" + "=" * 60)
    print("Cases with 3 Fixed Points (0 < alpha < 1)")
    print("=" * 60)
    
    for alpha in np.linspace(0.2, 0.9, 10):
        for beta in np.linspace(0.1, 2.0, 10):
            fps = find_fixed_points(alpha, beta)
            
            if len(fps) == 3:
                n_stable = 0
                for fp in fps:
                    stab, _, _ = analyze_stability(fp, alpha, beta)
                    if stab == "stable":
                        n_stable += 1
                
                if n_stable >= 2:
                    print(f"\nα={alpha:.2f}, β={beta:.2f}: {len(fps)} fixed points, {n_stable} stable")
                    for fp in fps:
                        stab, _, _ = analyze_stability(fp, alpha, beta)
                        print(f"  ({fp[0]:.2f}, {fp[1]:.2f}): {stab}")


if __name__ == "__main__":
    # 先找三稳态参数
    find_tristability()
    
    # 绘制典型相图
    print("\n" + "=" * 60)
    print("Generating Phase Portraits")
    print("=" * 60)
    
    # 情况1: alpha > 1, 只有2个固定点
    phase_portrait(1.5, 0.5, "Case 1: α>1, 2 Fixed Points").savefig(
        'F:/hypergraph_bistability/figures/2d_model/case1.png', dpi=150)
    plt.close()
    
    # 情况2: 0<alpha<1, 有3个固定点
    phase_portrait(0.5, 0.5, "Case 2: 0<α<1, 3 Fixed Points").savefig(
        'F:/hypergraph_bistability/figures/2d_model/case2.png', dpi=150)
    plt.close()
    
    # 情况3: 另一个3固定点情况
    phase_portrait(0.8, 1.0, "Case 3: 3 Fixed Points").savefig(
        'F:/hypergraph_bistability/figures/2d_model/case3.png', dpi=150)
    plt.close()
    
    # 绘制轨迹
    plot_trajectories(0.5, 0.5).savefig(
        'F:/hypergraph_bistability/figures/2d_model/trajectories.png', dpi=150)
    plt.close()
    
    print("\nDone! Figures saved.")
