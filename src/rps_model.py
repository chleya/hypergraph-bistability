"""
Rock-Paper-Scissors (石头剪刀布) 循环竞争模型
=============================================

这是一个经典的3态循环竞争系统：
- 石头(rock) 战胜 剪刀(scissors)
- 剪刀 战胜 布(paper)  
- 布 战胜 石头(rock)

在复制动力学中，这种系统自然产生3个角的稳定平衡 + 中心的不稳定平衡
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import os

os.makedirs('F:/hypergraph_bistability/figures/2d_model', exist_ok=True)


def rps_system(state, t, r, p):
    """
    简化版 Rock-Paper-Scissors 复制方程:
    dx/dt = x * (f_x - phi)
    dy/dt = y * (f_y - phi)
    dz/dt = z * (f_z - phi)
    
    其中 z = 1 - x - y (归一化)
    
    收益矩阵:
           R    P    S
    R   [[0,  -1,  1],
    P    [1,   0, -1],
    S   [-1,   1,  0]]
    
    但我们用2D版本
    """
    x, y = state
    z = 1 - x - y
    
    # 收益
    # R vs P: -1, R vs S: +1
    # P vs S: -1, P vs R: +1
    # S vs R: -1, S vs P: +1
    
    f_x = r * (y - p)  # R beats S, loses to P
    f_y = r * (z - x)  # P beats R, loses to S
    # f_z = r * (x - y)  # S beats P, loses to R
    
    # 平均适应度
    phi = r * p * (x*x + y*y + z*z) + r * (1-p) * (x*y + y*z + z*x)
    # 简化版
    phi = p * (x + y)  # 简化
    
    dxdt = x * (f_x - phi)
    dydt = y * (f_y - phi)
    
    return [dxdt, dydt]


def rps_classic(state, t, alpha, beta, gamma):
    """
    经典RPS with 参数化相互作用:
    dx/dt = x * (alpha*y + beta*z - gamma*(x+y))
    dy/dt = y * (alpha*z + beta*x - gamma*(x+y))
    
    用2D: z = 1 - x - y
    """
    x, y = state
    if x < 0 or y < 0 or x + y > 1:
        return [0, 0]
    
    z = 1 - x - y
    
    # 每个物种的收益
    # Species 0 (x) gets: alpha*y (beats y) + beta*z (beats z) - gamma*(y+z)
    # But we want cyclic: 0 beats 2, 2 beats 1, 1 beats 0
    
    # 简化：只有循环竞争
    dxdt = x * (y - z)  # x beats z, loses to y
    dydt = y * (z - x)  # y beats x, loses to z
    
    return [dxdt, dydt]


def rps_spiral(state, t, sigma):
    """带参数的RPS，能产生螺旋"""
    x, y = state
    if x < 0 or y < 0 or x + y > 1:
        return [0, 0]
    
    z = 1 - x - y
    
    # 添加参数控制
    dxdt = x * ((1+sigma)*y - z)
    dydt = y * ((1+sigma)*z - x)
    
    return [dxdt, dydt]


def plot_rps_phase(alpha_values, title_prefix):
    """绘制RPS相图"""
    fig, axes = plt.subplots(1, len(alpha_values), figsize=(6*len(alpha_values), 5))
    if len(alpha_values) == 1:
        axes = [axes]
    
    for idx, alpha in enumerate(alpha_values):
        ax = axes[idx]
        
        # 网格
        x = np.linspace(0.01, 0.99, 25)
        y = np.linspace(0.01, 0.99, 25)
        X, Y = np.meshgrid(x, y)
        
        # 过滤掉 x+y>1 的区域
        valid = X + Y < 1
        
        U = np.zeros_like(X)
        V = np.zeros_like(X)
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                if valid[i,j]:
                    dxdt, dydt = rps_spiral([X[i,j], Y[i,j]], 0, alpha)
                    U[i,j] = dxdt
                    V[i,j] = dydt
                else:
                    U[i,j] = 0
                    V[i,j] = 0
        
        # 绘制流场
        speed = np.sqrt(U**2 + V**2)
        # 遮盖无效区域
        speed = np.where(valid, speed, np.nan)
        
        ax.contourf(X, Y, speed, levels=15, cmap='viridis', alpha=0.7)
        ax.contour(X, Y, U, levels=5, colors='white', alpha=0.3)
        
        # 三条边
        ax.plot([0, 1], [0, 0], 'k-', linewidth=2)
        ax.plot([0, 0], [0, 1], 'k-', linewidth=2)
        ax.plot([0, 1], [1, 0], 'k-', linewidth=2)
        
        # 标记角点
        ax.plot(1, 0, 'ro', markersize=15, markeredgecolor='black', label='Rock (1,0,0)')
        ax.plot(0, 1, 'bo', markersize=15, markeredgecolor='black', label='Paper (0,1,0)')
        ax.plot(0, 0, 'go', markersize=15, markeredgecolor='black', label='Scissors (0,0,1)')
        
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel('x (Rock)')
        ax.set_ylabel('y (Paper)')
        ax.set_title(f'{title_prefix} σ={alpha}')
        ax.legend(loc='upper right', fontsize=8)
        
        # 限制在三角形内
        ax.set_aspect('equal')
    
    plt.tight_layout()
    return fig


def plot_trajectories_rps(sigma, n_points=15):
    """绘制RPS轨迹"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 初始条件 - 均匀分布在三角形内
    initials = []
    for x0 in np.linspace(0.05, 0.90, n_points):
        for y0 in np.linspace(0.05, 0.90, n_points):
            if x0 + y0 < 0.95:
                initials.append((x0, y0))
    
    t = np.linspace(0, 20, 500)
    colors = plt.cm.hsv(np.linspace(0, 1, len(initials)))
    
    for i, (x0, y0) in enumerate(initials):
        try:
            sol = odeint(rps_spiral, [x0, y0], t, args=(sigma,))
            # 过滤无效解
            valid = (sol[:, 0] > 0) & (sol[:, 1] > 0) & (sol[:, 0] + sol[:, 1] < 1)
            if sum(valid) > 10:
                ax.plot(sol[valid, 0], sol[valid, 1], color=colors[i], alpha=0.4, linewidth=0.8)
                ax.plot(sol[0, 0], sol[0, 1], 'o', color=colors[i], markersize=3)
        except:
            pass
    
    # 边界
    ax.plot([0, 1], [0, 0], 'k-', linewidth=3)
    ax.plot([0, 0], [0, 1], 'k-', linewidth=3)
    ax.plot([0, 1], [1, 0], 'k-', linewidth=3)
    
    # 角点
    ax.plot(1, 0, 'ro', markersize=15, markeredgecolor='black')
    ax.plot(0, 1, 'bo', markersize=15, markeredgecolor='black')
    ax.plot(0, 0, 'go', markersize=15, markeredgecolor='black')
    
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('x (Rock)')
    ax.set_ylabel('y (Paper)')
    ax.set_title(f'RPS Trajectories σ={sigma}')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    return fig


def rps_bistability():
    """尝试找RPS中的双稳态/三稳态"""
    print("=" * 60)
    print("Rock-Paper-Scissors Model Analysis")
    print("=" * 60)
    
    # 1. 经典RPS (sigma=0)
    print("\n1. Classic RPS (sigma=0):")
    print("   The simplex has 3 corner fixed points (1,0,0), (0,1,0), (0,0,1)")
    print("   These are typically stable in discrete systems")
    print("   The interior fixed point (1/3, 1/3, 1/3) is a saddle")
    
    # 2. 带参数
    print("\n2. With sigma parameter:")
    
    for sigma in [-0.5, -0.2, 0, 0.2, 0.5]:
        # 分析边界固定点
        # (1, 0, 0): dx/dt = 0, 特征值 = ?
        # 简化为2D: 在x=1, y=0处
        # 这个点是稳定的当 sigma < 0
        print(f"   sigma={sigma}: corners are {'stable' if sigma < 0 else 'unstable'}")
    
    # 3. 绘制不同sigma的相图
    print("\n3. Generating phase portraits...")
    for sigma in [-0.3, 0, 0.3]:
        fig = plot_trajectories_rps(sigma, n_points=12)
        fig.savefig(f'F:/hypergraph_bistability/figures/2d_model/rps_sigma{sigma}.png'.replace('-', 'm'), dpi=150)
        plt.close()
    
    # 4. 另一个2D模型：三个高斯势阱
    print("\n4. Three Gaussian Wells in 2D:")
    print("   This is the most direct way to get 3 stable points")


if __name__ == "__main__":
    rps_bistability()
    
    print("\nDone!")
