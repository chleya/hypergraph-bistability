"""
2D三势阱模型 - 直接构造3个稳定吸引子
====================================

在2D平面上构造一个势能函数 V(x,y)，有三个局部极小值点
每个极小值点都是一个稳定吸引子
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import os

os.makedirs('F:/hypergraph_bistability/figures/2d_model', exist_ok=True)


def three_wells_potential(x, y, h1, h2, h3, s1, s2, s3):
    """
    三个高斯势阱:
    V(x,y) = -h1*exp(-((x-x1)^2 + (y-y1)^2)/s1) 
           - h2*exp(-((x-x2)^2 + (y-y2)^2)/s2)
           - h3*exp(-((x-x3)^2 + (y-y3)^2)/s3)
    
    力 = -grad(V)
    """
    # 三个势阱中心位置 (三角形分布)
    x1, y1 = 0.2, 0.2
    x2, y2 = 0.8, 0.2
    x3, y3 = 0.5, 0.8
    
    V = (-h1 * np.exp(-((x-x1)**2 + (y-y1)**2) / s1)
         - h2 * np.exp(-((x-x2)**2 + (y-y2)**2) / s2)
         - h3 * np.exp(-((x-x3)**2 + (y-y3)**2) / s3))
    
    return V


def three_wells_force(x, y, h1, h2, h3, s1, s2, s3):
    """计算力 F = -grad(V)"""
    x1, y1 = 0.2, 0.2
    x2, y2 = 0.8, 0.2
    x3, y3 = 0.5, 0.8
    
    # dV/dx
    dVdx = (-h1 * np.exp(-((x-x1)**2 + (y-y1)**2) / s1) * (-2*(x-x1)/s1)
            - h2 * np.exp(-((x-x2)**2 + (y-y2)**2) / s2) * (-2*(x-x2)/s2)
            - h3 * np.exp(-((x-x3)**2 + (y-y3)**2) / s3) * (-2*(x-x3)/s3))
    
    # dV/dy
    dVdy = (-h1 * np.exp(-((x-x1)**2 + (y-y1)**2) / s1) * (-2*(y-y1)/s1)
            - h2 * np.exp(-((x-x2)**2 + (y-y2)**2) / s2) * (-2*(y-y2)/s2)
            - h3 * np.exp(-((x-x3)**2 + (y-y3)**2) / s3) * (-2*(y-y3)/s3))
    
    # F = -grad(V)
    fx = -dVdx
    fy = -dVdy
    
    return fx, fy


def dynamics(state, t, h1, h2, h3, s1, s2, s3, damping):
    """带阻尼的动力学"""
    x, y = state
    fx, fy = three_wells_force(x, y, h1, h2, h3, s1, s2, s3)
    
    dxdt = damping * fx
    dydt = damping * fy
    
    return [dxdt, dydt]


def plot_potential_and_phase(h1, h2, h3, s1, s2, s3, damping, name):
    """绘制势能和相图"""
    fig = plt.figure(figsize=(16, 6))
    
    # 1. 势能等高线
    ax1 = fig.add_subplot(1, 3, 1)
    
    x = np.linspace(0, 1, 60)
    y = np.linspace(0, 1, 60)
    X, Y = np.meshgrid(x, y)
    
    Z = three_wells_potential(X, Y, h1, h2, h3, s1, s2, s3)
    
    contour = ax1.contourf(X, Y, Z, levels=25, cmap='RdYlBu_r')
    ax1.contour(X, Y, Z, levels=15, colors='black', alpha=0.3, linewidths=0.5)
    plt.colorbar(contour, ax=ax1, label='Potential V')
    
    # 标记势阱
    ax1.plot(0.2, 0.2, 'go', markersize=15, markeredgecolor='black', label='Well 1')
    ax1.plot(0.8, 0.2, 'bo', markersize=15, markeredgecolor='black', label='Well 2')
    ax1.plot(0.5, 0.8, 'ro', markersize=15, markeredgecolor='black', label='Well 3')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title(f'{name}\nPotential Landscape')
    ax1.legend()
    
    # 2. 流场
    ax2 = fig.add_subplot(1, 3, 2)
    
    U = np.zeros_like(X)
    V = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            fx, fy = three_wells_force(X[i,j], Y[i,j], h1, h2, h3, s1, s2, s3)
            U[i,j] = damping * fx
            V[i,j] = damping * fy
    
    speed = np.sqrt(U**2 + V**2)
    ax2.streamplot(x, y, U, V, color=speed, cmap='viridis', density=1.5)
    
    ax2.plot(0.2, 0.2, 'go', markersize=15, markeredgecolor='black')
    ax2.plot(0.8, 0.2, 'bo', markersize=15, markeredgecolor='black')
    ax2.plot(0.5, 0.8, 'ro', markersize=15, markeredgecolor='black')
    
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Phase Flow')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    
    # 3. 轨迹
    ax3 = fig.add_subplot(1, 3, 3)
    
    # 初始点网格
    initials = []
    for x0 in np.linspace(0.05, 0.95, 8):
        for y0 in np.linspace(0.05, 0.95, 8):
            initials.append((x0, y0))
    
    t = np.linspace(0, 5, 300)
    colors = plt.cm.tab10(np.linspace(0, 1, len(initials)))
    
    # 找到最终吸引子
    final_positions = []
    
    for i, (x0, y0) in enumerate(initials):
        try:
            sol = odeint(dynamics, [x0, y0], t, args=(h1,h2,h3,s1,s2,s3,damping))
            ax3.plot(sol[:, 0], sol[:, 1], color=colors[i], alpha=0.5, linewidth=1)
            ax3.plot(sol[0, 0], sol[0, 1], 'o', color=colors[i], markersize=4)
            ax3.plot(sol[-1, 0], sol[-1, 1], 's', color=colors[i], markersize=8)
            final_positions.append((sol[-1, 0], sol[-1, 1]))
        except:
            pass
    
    ax3.plot(0.2, 0.2, 'go', markersize=15, markeredgecolor='black')
    ax3.plot(0.8, 0.2, 'bo', markersize=15, markeredgecolor='black')
    ax3.plot(0.5, 0.8, 'ro', markersize=15, markeredgecolor='black')
    
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_title('Trajectories (Squares = End Points)')
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    
    plt.tight_layout()
    
    # 统计吸引子
    unique_attractors = []
    for fp in final_positions:
        is_new = True
        for up in unique_attractors:
            if abs(fp[0] - up[0]) < 0.15 and abs(fp[1] - up[1]) < 0.15:
                is_new = False
                break
        if is_new:
            unique_attractors.append(fp)
    
    print(f"\n{name}:")
    print(f"  Unique attractors: {len(unique_attractors)}")
    for at in unique_attractors:
        print(f"    ({at[0]:.2f}, {at[1]:.2f})")
    
    return fig, unique_attractors


if __name__ == "__main__":
    print("=" * 60)
    print("2D Three-Well Potential Model")
    print("=" * 60)
    
    # 测试不同参数组合
    test_cases = [
        # (h1, h2, h3, s1, s2, s3, damping, name)
        (1.0, 1.0, 1.0, 0.08, 0.08, 0.08, 0.5, "Equal Wells"),
        (1.5, 1.5, 1.5, 0.06, 0.06, 0.06, 0.5, "Deep Wells"),
        (1.0, 1.0, 1.2, 0.07, 0.07, 0.07, 0.5, "Asymmetric Depth"),
        (1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.8, "Wide Wells"),
        (2.0, 2.0, 2.0, 0.05, 0.05, 0.05, 0.3, "Very Deep"),
    ]
    
    all_results = []
    
    for h1, h2, h3, s1, s2, s3, damping, name in test_cases:
        try:
            fig, attractors = plot_potential_and_phase(h1, h2, h3, s1, s2, s3, damping, name)
            fig.savefig(f'F:/hypergraph_bistability/figures/2d_model/three_wells_{name.replace(" ", "_")}.png'.lower(), dpi=150)
            plt.close()
            all_results.append((name, len(attractors), attractors))
        except Exception as e:
            print(f"Error with {name}: {e}")
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, n_attractors, _ in all_results:
        print(f"{name}: {n_attractors} attractors")
    
    print("\nDone!")
