"""
手工设计的三稳态模型
===================

直接构造一个三势阱势能函数
"""


import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('F:/hypergraph_bistability/figures/tristability', exist_ok=True)


def triple_well_potential(x, h1, h2, h3, w1, w2, w3):
    """
    三高斯势阱
    V(x) = -h1*exp(-(x-w1)^2) - h2*exp(-(x-w2)^2) - h3*exp(-(x-w3)^2)
    
    F(x) = -dV/dx
    """
    term1 = -h1 * np.exp(-((x - w1)**2) / 0.02)
    term2 = -h2 * np.exp(-((x - w2)**2) / 0.02)
    term3 = -h3 * np.exp(-((x - w3)**2) / 0.02)
    
    # F是势能的负梯度
    return term1 + term2 + term3


def F_from_potential(x, h1, h2, h3, w1, w2, w3):
    """从势能计算力"""
    # 数值导数
    dx = 0.001
    V_plus = triple_well_potential(x + dx, h1, h2, h3, w1, w2, w3)
    V_minus = triple_well_potential(x - dx, h1, h2, h3, w1, w2, w3)
    return -(V_plus - V_minus) / (2 * dx)


def find_roots_numerical(F, x_range, n=3000):
    """数值找零点"""
    x = np.linspace(x_range[0], x_range[1], n)
    y = F(x)
    
    roots = []
    for i in range(len(x) - 1):
        if y[i] * y[i+1] < 0:
            roots.append((x[i] + x[i+1]) / 2)
    return roots


def check_stability(root, F):
    """检查稳定性"""
    h = 0.0001
    dF = (F(root + h) - F(root - h)) / (2 * h)
    return dF < 0


def manual_triple_well():
    """手工设计三势阱"""
    print("=" * 60)
    print("Manual Triple Well Design")
    print("=" * 60)
    
    # 三个峰位置
    w1, w2, w3 = 0.2, 0.5, 0.8
    
    # 搜索高度
    for h1 in np.linspace(0.5, 2.0, 10):
        for h2 in np.linspace(0.5, 2.0, 10):
            for h3 in np.linspace(0.5, 2.0, 10):
                def F(x):
                    return F_from_potential(x, h1, h2, h3, w1, w2, w3)
                
                roots = find_roots_numerical(F, [0, 1])
                stable = [r for r in roots if check_stability(r, F)]
                
                # 只关心在 [0.05, 0.95] 范围内的
                in_range = [r for r in stable if 0.05 < r < 0.95]
                
                if len(in_range) >= 3:
                    print(f"\nFound! h1={h1:.2f}, h2={h2:.2f}, h3={h3:.2f}")
                    print(f"  Stable: {in_range}")
                    
                    plot_triple_well(h1, h2, h3, w1, w2, w3)
                    return
    
    print("\nNo triple stable states found with uniform heights")
    print("Trying asymmetric heights...")
    
    # 尝试非对称
    for h1 in np.linspace(0.3, 2.5, 15):
        for h2 in np.linspace(0.1, 1.5, 15):
            for h3 in np.linspace(0.3, 2.5, 15):
                def F(x):
                    return F_from_potential(x, h1, h2, h3, w1, w2, w3)
                
                roots = find_roots_numerical(F, [0, 1])
                stable = [r for r in roots if check_stability(r, F)]
                in_range = [r for r in stable if 0.05 < r < 0.95]
                
                if len(in_range) >= 3:
                    print(f"\nFound! h1={h1:.2f}, h2={h2:.2f}, h3={h3:.2f}")
                    print(f"  Stable: {in_range}")
                    plot_triple_well(h1, h2, h3, w1, w2, w3)
                    return
    
    print("\nStill no luck. Let's try a simpler approach...")


def simplest_triple_well():
    """最简单的三稳态：使用分段函数或多项式"""
    print("\n" + "=" * 60)
    print("Simplest Triple Well: 4th order polynomial")
    print("=" * 60)
    
    # x^4 - 2x^3 + x^2 = x^2(x^2 - 2x + 1) = x^2(x-1)^2
    # 这个在0和1处有平台
    
    # 尝试：F(x) = k*(x-0.2)*(x-0.5)*(x-0.8) - c
    # 这是一个三次函数，最多3个根
    
    def F(x, k, c):
        return k * (x - 0.2) * (x - 0.5) * (x - 0.8) - c
    
    for k in np.linspace(3, 10, 20):
        for c in np.linspace(0.1, 0.5, 20):
            roots = find_roots_numerical(lambda x: F(x, k, c), [0, 1])
            stable = [r for r in roots if check_stability(lambda x: F(x, k, c), r)]
            in_range = [r for r in stable if 0.05 < r < 0.95]
            
            if len(in_range) >= 3:
                print(f"\nFound! k={k:.2f}, c={c:.2f}")
                print(f"  Stable: {in_range}")
                plot_quartic(k, c)
                return
    
    print("\nCubic doesn't work. Trying quartic...")
    
    # 尝试更高阶: F(x) = k*(x-0.15)*(x-0.4)*(x-0.6)*(x-0.85) - c
    for k in np.linspace(20, 50, 15):
        for c in np.linspace(0.1, 0.8, 20):
            try:
                def F(x):
                    return k * (x - 0.15) * (x - 0.4) * (x - 0.6) * (x - 0.85) - c
                
                roots = find_roots_numerical(F, [0, 1])
                stable = [r for r in roots if check_stability(F, r)]
                in_range = [r for r in stable if 0.05 < r < 0.95]
                
                if len(in_range) >= 3:
                    print(f"\nFound! k={k:.2f}, c={c:.2f}")
                    print(f"  Stable: {in_range}")
                    plot_quartic(k, c, mode='quartic')
                    return
            except:
                pass
    
    print("\nNeed different approach. Let's manually verify one case...")
    
    # 手动验证一个四次多项式
    k, c = 30, 0.4
    def F(x):
        return k * (x - 0.15) * (x - 0.4) * (x - 0.6) * (x - 0.85) - c
    
    roots = find_roots_numerical(F, [0, 1])
    print(f"\nRoots at k={k}, c={c}: {roots}")
    
    for r in roots:
        d = check_stability(F, r)
        print(f"  x={r:.3f}, stable={d}")
    
    # 画这个
    plot_quartic_manual()


def plot_triple_well(h1, h2, h3, w1, w2, w3):
    """绘制三势阱"""
    x = np.linspace(0, 1, 500)
    V = triple_well_potential(x, h1, h2, h3, w1, w2, w3)
    F = np.gradient(-V, x)  # 负梯度
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 势能
    ax1 = axes[0]
    ax1.plot(x, V, 'b-', linewidth=2)
    ax1.set_xlabel('x')
    ax1.set_ylabel('V(x)')
    ax1.set_title('Triple Well Potential')
    ax1.grid(True, alpha=0.3)
    
    # 力
    ax2 = axes[1]
    ax2.plot(x, F, 'r-', linewidth=2)
    ax2.axhline(y=0, color='k', linestyle='--')
    
    # 标记稳定点
    roots = find_roots_numerical(lambda x: F_from_potential(x, h1, h2, h3, w1, w2, w3), [0, 1])
    for r in roots:
        if check_stability(r, lambda x: F_from_potential(x, h1, h2, h3, w1, w2, w3)):
            ax2.plot(r, 0, 'go', markersize=12)
    
    ax2.set_xlabel('x')
    ax2.set_ylabel('F(x)')
    ax2.set_title('Force (F = -dV/dx)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('F:/hypergraph_bistability/figures/tristability/triple_well.png', dpi=150)
    plt.close()


def plot_quartic(k, c, mode='cubic'):
    """绘制四次/三次多项式"""
    x = np.linspace(0, 1, 500)
    
    if mode == 'quartic':
        y = k * (x - 0.15) * (x - 0.4) * (x - 0.6) * (x - 0.85) - c
    else:
        y = k * (x - 0.2) * (x - 0.5) * (x - 0.8) - c
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', linewidth=2)
    plt.axhline(y=0, color='k', linestyle='--')
    
    # 找根
    if mode == 'quartic':
        F = lambda x: k * (x - 0.15) * (x - 0.4) * (x - 0.6) * (x - 0.85) - c
    else:
        F = lambda x: k * (x - 0.2) * (x - 0.5) * (x - 0.8) - c
    
    roots = find_roots_numerical(F, [0, 1])
    for r in roots:
        if check_stability(F, r):
            plt.plot(r, 0, 'go', markersize=12)
        else:
            plt.plot(r, 0, 'rx', markersize=12)
    
    plt.xlabel('x')
    plt.ylabel('F(x)')
    plt.title(f'{mode.capitalize()}: k={k:.2f}, c={c:.2f}')
    plt.grid(True, alpha=0.3)
    plt.xlim(-0.02, 1.02)
    plt.savefig(f'F:/hypergraph_bistability/figures/tristability/{mode}.png', dpi=150)
    plt.close()


def plot_quartet_manual():
    """手动绘制验证"""
    k, c = 30, 0.4
    
    x = np.linspace(0, 1, 500)
    y = k * (x - 0.15) * (x - 0.4) * (x - 0.6) * (x - 0.85) - c
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', linewidth=2)
    plt.axhline(y=0, color='k', linestyle='--')
    
    def F(x):
        return k * (x - 0.15) * (x - 0.4) * (x - 0.6) * (x - 0.85) - c
    
    roots = find_roots_numerical(F, [0, 1])
    print(f"Roots: {roots}")
    
    for r in roots:
        stable = check_stability(F, r)
        print(f"  x={r:.3f}, stable={stable}")
        if stable:
            plt.plot(r, 0, 'go', markersize=15, markeredgecolor='black')
        else:
            plt.plot(r, 0, 'rx', markersize=15, markeredgecolor='black')
    
    plt.xlabel('x (cohesion)')
    plt.ylabel('F(x)')
    plt.title('Quartic: F(x) = k*(x-0.15)(x-0.4)(x-0.6)(x-0.85) - c')
    plt.grid(True, alpha=0.3)
    plt.xlim(-0.02, 1.02)
    plt.savefig('F:/hypergraph_bistability/figures/tristability/quartic_manual.png', dpi=150)
    plt.close()


if __name__ == "__main__":
    simplest_triple_well()
    print("\nDone!")
