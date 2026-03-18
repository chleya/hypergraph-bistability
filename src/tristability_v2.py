"""
改进的三稳态模型
================

使用三势阱来产生3个稳定状态
"""

import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('F:/hypergraph_bistability/figures/tristability', exist_ok=True)


def triple_well(x, a, b):
    """
    三势阱模型
    F(x) = a * sin(3*pi*x) * (x - 0.5) - b
    
    或者更简单：三余弦
    F(x) = -cos(3*pi*x) - b + c*x
    """
    # 产生三个峰
    return -np.cos(3 * np.pi * x) - b + 0.5 * x


def polynomial_tristability(x, c1, c2, b):
    """
    多项式形式 - 更易控
    F(x) = c1*x*(1-x) + c2*x*(1-x)*(x-0.5)^2 - b
    """
    return c1 * x * (1 - x) + c2 * x * (1 - x) * (x - 0.5)**2 - b


def find_roots(func, x_range, n=1000):
    """找零点"""
    x = np.linspace(x_range[0], x_range[1], n)
    y = func(x)
    
    roots = []
    for i in range(len(x) - 1):
        if y[i] * y[i+1] < 0:
            roots.append((x[i] + x[i+1]) / 2)
    return roots


def check_stability(root, func):
    """检查稳定性"""
    h = 0.001
    d = (func(root + h) - func(root - h)) / (2 * h)
    return d < 0


def scan_parameters():
    """参数扫描找三稳态"""
    print("=" * 60)
    print("Parameter Scan for Tristability")
    print("=" * 60)
    
    found = []
    
    for c1 in np.linspace(0.5, 2.0, 8):
        for c2 in np.linspace(0.5, 3.0, 8):
            for b in np.linspace(0.1, 0.4, 10):
                def F(x):
                    return polynomial_tristability(x, c1, c2, b)
                
                roots = find_roots(F, [0, 1])
                stable_count = sum(1 for r in roots if check_stability(r, F))
                
                if stable_count >= 3:
                    found.append({
                        'c1': c1, 'c2': c2, 'b': b,
                        'stable_count': stable_count,
                        'roots': roots
                    })
    
    print(f"\nFound {len(found)} parameter combinations with 3+ stable states")
    
    if found:
        print("\nFirst example:")
        f = found[0]
        print(f"  c1={f['c1']:.2f}, c2={f['c2']:.2f}, b={f['b']:.2f}")
        print(f"  Stable roots: {f['roots']}")
        
        # 绘制这个例子
        plot_example(f['c1'], f['c2'], f['b'])
    
    return found


def plot_example(c1, c2, b):
    """绘制示例"""
    x = np.linspace(0, 1, 200)
    y = polynomial_tristability(x, c1, c2, b)
    
    roots = find_roots(lambda x: polynomial_tristability(x, c1, c2, b), [0, 1])
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', linewidth=2)
    plt.axhline(y=0, color='k', linestyle='--')
    
    # 标记稳态点
    for r in roots:
        stable = check_stability(r, lambda x: polynomial_tristability(x, c1, c2, b))
        if stable:
            plt.plot(r, 0, 'go', markersize=12)
        else:
            plt.plot(r, 0, 'rx', markersize=12)
    
    plt.xlabel('x')
    plt.ylabel('F(x)')
    plt.title(f'Tristability: c1={c1:.2f}, c2={c2:.2f}, b={b:.2f}')
    plt.grid(True, alpha=0.3)
    plt.savefig('F:/hypergraph_bistability/figures/tristability/example_tristability.png', dpi=150)
    plt.close()


def simple_tristability():
    """简单测试：固定c1=1, c2=2, 扫描b"""
    print("\n" + "=" * 60)
    print("Simple Scan: c1=1.5, c2=2.5, vary b")
    print("=" * 60)
    
    c1, c2 = 1.5, 2.5
    
    results = []
    for b in np.linspace(0.05, 0.35, 20):
        def F(x):
            return polynomial_tristability(x, c1, c2, b)
        
        roots = find_roots(F, [0, 1])
        stable = [r for r in roots if check_stability(r, F)]
        
        results.append({
            'b': b,
            'n_stable': len(stable),
            'stable': stable
        })
        
        if len(stable) >= 3:
            print(f"\nFound 3+ stable states at b={b:.3f}")
            print(f"  Stable: {stable}")
    
    # 绘图
    plt.figure(figsize=(12, 8))
    
    x = np.linspace(0, 1, 200)
    
    # 画几个b值
    for i, b in enumerate([0.1, 0.15, 0.2, 0.25, 0.3]):
        y = polynomial_tristability(x, c1, c2, b)
        plt.subplot(2, 3, i+1)
        plt.plot(x, y)
        plt.axhline(y=0, color='k', linestyle='--')
        plt.title(f'b={b:.2f}')
        
        # 标记稳态
        def F(x):
            return polynomial_tristability(x, c1, c2, b)
        roots = find_roots(F, [0, 1])
        for r in roots:
            if check_stability(r, F):
                plt.plot(r, 0, 'go')
    
    plt.tight_layout()
    plt.savefig('F:/hypergraph_bistability/figures/tristability/simple_scan.png', dpi=150)
    plt.close()
    
    # 绘制相图
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    bs = [r['b'] for r in results]
    n_stables = [r['n_stable'] for r in results]
    plt.plot(bs, n_stables, 'o-')
    plt.xlabel('b')
    plt.ylabel('# Stable States')
    plt.title('Number of Stable States vs b')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    # 相图
    b_range = np.linspace(0.05, 0.35, 20)
    x0_range = np.linspace(0.02, 0.98, 30)
    
    final_grid = np.zeros((len(b_range), len(x0_range)))
    
    def simulate(x0, b, steps=50):
        x = x0
        for _ in range(steps):
            F = polynomial_tristability(x, c1, c2, b)
            x = x + 0.1 * F
            x = np.clip(x, 0.01, 0.99)
        return x
    
    for i, b in enumerate(b_range):
        for j, x0 in enumerate(x0_range):
            final_grid[i, j] = simulate(x0, b)
    
    plt.imshow(final_grid, aspect='auto', origin='lower',
               extent=[0, 1, 0.05, 0.35], cmap='RdYlBu_r')
    plt.colorbar(label='Final x')
    plt.xlabel('Initial x0')
    plt.ylabel('b')
    plt.title('Phase Diagram')
    
    plt.tight_layout()
    plt.savefig(r'F:/hypergraph_bistability/figures/tristability/phase_simple.png', dpi=150)
    plt.close()


if __name__ == "__main__":
    # 先做简单扫描
    simple_tristability()
    
    # 再做全参数扫描
    found = scan_parameters()
    
    print("\nDone!")
