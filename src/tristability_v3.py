"""
真正的三稳态模型
================

使用更高阶多项式或三角函数来产生3个稳定状态
"""

import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('F:/hypergraph_bistability/figures/tristability', exist_ok=True)


def cubic_FE(x, a, b, c, d):
    """
    三次多项式: F(x) = ax^3 + bx^2 + cx + d
    """
    return a * x**3 + b * x**2 + c * x + d


def find_roots_analytical(a, b, c, d):
    """解三次方程 ax^3 + bx^2 + cx + d = 0"""
    # 使用numpy的roots函数
    coeffs = [a, b, c, d]
    roots = np.roots(coeffs)
    # 只保留实根且在[0,1]区间
    real_roots = []
    for r in roots:
        if np.isreal(r) or abs(r.imag) < 1e-6:
            r_real = float(r.real)
            if -0.1 <= r_real <= 1.1:
                real_roots.append(r_real)
    return real_roots


def cubic_stability(root, a, b, c):
    """检查三次多项式的稳定性"""
    # dF/dx = 3ax^2 + 2bx + c
    dFdx = 3*a*root**2 + 2*b*root + c
    return dFdx < 0


def find_cubic_tristability():
    """找到产生3个稳定状态的参数"""
    print("=" * 60)
    print("Finding Cubic Tristability")
    print("=" * 60)
    
    found = []
    
    # 搜索参数空间
    for a in np.linspace(-5, 5, 20):
        for b in np.linspace(-5, 5, 20):
            for c in np.linspace(-5, 5, 20):
                for d in np.linspace(-1, 1, 10):
                    try:
                        roots = find_roots_analytical(a, b, c, d)
                        stable = [r for r in roots if cubic_stability(r, a, b, c)]
                        
                        if len(stable) >= 3:
                            found.append({
                                'a': a, 'b': b, 'c': c, 'd': d,
                                'stable': stable
                            })
                    except:
                        pass
    
    print(f"\nFound {len(found)} combinations with 3+ stable states")
    
    if found:
        # 打印前几个
        for i, f in enumerate(found[:5]):
            print(f"\n{i+1}. a={f['a']:.2f}, b={f['b']:.2f}, c={f['c']:.2f}, d={f['d']:.2f}")
            print(f"   Stable: {f['stable']}")
        
        # 画第一个
        plot_cubic_example(found[0])
    
    return found


def plot_cubic_example(params):
    """绘制三次多项式示例"""
    a, b, c, d = params['a'], params['b'], params['c'], params['d']
    
    x = np.linspace(-0.1, 1.1, 200)
    y = cubic_FE(x, a, b, c, d)
    
    roots = find_roots_analytical(a, b, c, d)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', linewidth=2)
    plt.axhline(y=0, color='k', linestyle='--')
    plt.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    plt.axvline(x=1, color='gray', linestyle=':', alpha=0.5)
    
    # 标记稳态点
    for r in roots:
        if cubic_stability(r, a, b, c):
            plt.plot(r, 0, 'go', markersize=12)
        else:
            plt.plot(r, 0, 'rx', markersize=12)
    
    plt.xlabel('x')
    plt.ylabel('F(x)')
    plt.title(f'Cubic Tristability: a={a:.2f}, b={b:.2f}, c={c:.2f}, d={d:.2f}')
    plt.grid(True, alpha=0.3)
    plt.xlim(-0.1, 1.1)
    plt.savefig('F:/hypergraph_bistability/figures/tristability/cubic_tristability.png', dpi=150)
    plt.close()


def sine_tristability():
    """使用sin函数构造三稳态"""
    print("\n" + "=" * 60)
    print("Sine-based Tristability")
    print("=" * 60)
    
    # F(x) = A*sin(3*pi*x) + B*(x-0.5) - C
    # 这应该产生3个峰/谷
    
    def F(x, A, B, C):
        return A * np.sin(3 * np.pi * x) + B * (x - 0.5) - C
    
    def find_roots_sine(A, B, C):
        x = np.linspace(0, 1, 1000)
        y = F(x, A, B, C)
        
        roots = []
        for i in range(len(x) - 1):
            if y[i] * y[i+1] < 0:
                roots.append((x[i] + x[i+1]) / 2)
        return roots
    
    def check_stability_sine(root, A, B):
        h = 0.001
        d = (F(root + h, A, B, 0) - F(root - h, A, B, 0)) / (2 * h)
        return d < 0
    
    # 扫描参数
    found = []
    
    for A in np.linspace(0.5, 3.0, 15):
        for B in np.linspace(0.1, 1.0, 10):
            for C in np.linspace(0.1, 0.8, 15):
                roots = find_roots_sine(A, B, C)
                stable = [r for r in roots if check_stability_sine(r, A, B)]
                
                # 检查是否在[0,1]区间有3个稳定点
                in_range = [r for r in stable if 0 < r < 1]
                if len(in_range) >= 3:
                    found.append({
                        'A': A, 'B': B, 'C': C,
                        'stable': in_range
                    })
    
    print(f"\nFound {len(found)} combinations")
    
    if found:
        # 打印前3个
        for i, f in enumerate(found[:3]):
            print(f"\n{i+1}. A={f['A']:.2f}, B={f['B']:.2f}, C={f['C']:.2f}")
            print(f"   Stable: {f['stable']}")
        
        # 画第一个
        plot_sine_example(found[0])
    
    return found


def plot_sine_example(params):
    """绘制sin模型示例"""
    A, B, C = params['A'], params['B'], params['C']
    
    x = np.linspace(0, 1, 200)
    y = A * np.sin(3 * np.pi * x) + B * (x - 0.5) - C
    
    def F(x):
        return A * np.sin(3 * np.pi * x) + B * (x - 0.5) - C
    
    def find_roots_sine():
        roots = []
        for i in range(len(x) - 1):
            if y[i] * y[i+1] < 0:
                roots.append((x[i] + x[i+1]) / 2)
        return roots
    
    roots = find_roots_sine()
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', linewidth=2)
    plt.axhline(y=0, color='k', linestyle='--')
    
    # 标记稳态点
    for r in roots:
        h = 0.001
        d = (F(r + h) - F(r - h)) / (2 * h)
        if d < 0:
            plt.plot(r, 0, 'go', markersize=12)
        else:
            plt.plot(r, 0, 'rx', markersize=12)
    
    plt.xlabel('x')
    plt.ylabel('F(x)')
    plt.title(f'Sine Tristability: A={A:.2f}, B={B:.2f}, C={C:.2f}')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.savefig('F:/hypergraph_bistability/figures/tristability/sine_tristability.png', dpi=150)
    plt.close()


if __name__ == "__main__":
    # 尝试三次多项式
    cubic_results = find_cubic_tristability()
    
    # 尝试sin函数
    sine_results = sine_tristability()
    
    print("\nDone!")
