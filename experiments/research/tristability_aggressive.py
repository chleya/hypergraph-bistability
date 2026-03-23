"""
三稳态模型 - 强化版
==================

使用更强的非线性组合来产生3个稳定状态
"""


import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('F:/hypergraph_bistability/figures/tristability', exist_ok=True)


def find_roots(F, x_range, n=5000):
    """找零点"""
    x = np.linspace(x_range[0], x_range[1], n)
    y = F(x)
    
    roots = []
    for i in range(len(x) - 1):
        if y[i] * y[i+1] < 0:
            # 二分法精确化
            xl, xr = x[i], x[i+1]
            for _ in range(30):
                xm = (xl + xr) / 2
                if y[i] * F(xm) < 0:
                    xr = xm
                else:
                    xl = xm
            roots.append(xm)
    return roots


def check_stability(F, root):
    """检查稳定性"""
    h = 0.0001
    dF = (F(root + h) - F(root - h)) / (2 * h)
    return dF < 0


def plot_F_and_phase(F, name):
    """绘制F(x)和相图"""
    x = np.linspace(0, 1, 500)
    y = F(x)
    
    roots = find_roots(F, [0, 1])
    stable = [r for r in roots if 0.02 < r < 0.98 and check_stability(F, r)]
    
    print(f"\n{name}:")
    print(f"  Roots: {[f'{r:.3f}' for r in roots]}")
    print(f"  Stable in range: {[f'{r:.3f}' for r in stable]}")
    print(f"  N stable: {len(stable)}")
    
    if len(stable) >= 3:
        print(f"  *** FOUND 3+ STABLE STATES! ***")
        
        # 绘制
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # F(x)
        axes[0].plot(x, y, 'b-', linewidth=2)
        axes[0].axhline(y=0, color='k', linestyle='--')
        for r in stable:
            axes[0].plot(r, 0, 'go', markersize=15, markeredgecolor='black')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('F(x)')
        axes[0].set_title(f'{name}: F(x)')
        axes[0].grid(True, alpha=0.3)
        
        # 相图
        c_values = np.linspace(0.1, 0.8, 20)
        x0_values = np.linspace(0.02, 0.98, 25)
        
        final_grid = np.zeros((len(c_values), len(x0_values)))
        
        for i, c in enumerate(c_values):
            for j, x0 in enumerate(x0_values):
                x = x0
                for _ in range(100):
                    F_val = F(x) - c  # c as parameter
                    x = x + 0.08 * F_val
                    x = np.clip(x, 0.001, 0.999)
                final_grid[i, j] = x
        
        axes[1].imshow(final_grid, aspect='auto', origin='lower',
                      extent=[0, 1, c_values[0], c_values[-1]], cmap='RdYlBu_r')
        axes[1].set_xlabel('Initial x0')
        axes[1].set_ylabel('c')
        axes[1].set_title('Phase Diagram')
        
        # 区域占比
        low = [sum(1 for j in range(len(x0_values)) if final_grid[i,j] < 0.3) / len(x0_values) for i in range(len(c_values))]
        mid = [sum(1 for j in range(len(x0_values)) if 0.3 <= final_grid[i,j] < 0.7) / len(x0_values) for i in range(len(c_values))]
        high = [sum(1 for j in range(len(x0_values)) if final_grid[i,j] >= 0.7) / len(x0_values) for i in range(len(c_values))]
        
        axes[2].plot(c_values, low, 'b-o', label='LOW', markersize=3)
        axes[2].plot(c_values, mid, 'g-s', label='MID', markersize=3)
        axes[2].plot(c_values, high, 'r-^', label='HIGH', markersize=3)
        axes[2].set_xlabel('c')
        axes[2].set_ylabel('Fraction')
        axes[2].set_title('Basin Fraction')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'F:/hypergraph_bistability/figures/tristability/{name}.png', dpi=150)
        plt.close()
        
        return True
    return False


# ========== 尝试不同的函数形式 ==========

print("=" * 60)
print("Searching for Tristability")
print("=" * 60)

# 方法1: 3次谐波 + 二次项
# F(x) = A*sin(3*pi*x) + B*sin(6*pi*x) + C*x*(1-x) - D
print("\n1. Testing: F = A*sin(3πx) + B*sin(6πx) + C*x*(1-x) - D")

found = False
for A in np.linspace(0.5, 4.0, 12):
    for B in np.linspace(0.2, 2.0, 8):
        for C in np.linspace(0.5, 3.0, 10):
            for D in np.linspace(0.1, 0.9, 15):
                def F(x):
                    return A*np.sin(3*np.pi*x) + B*np.sin(6*np.pi*x) + C*x*(1-x) - D
                
                if plot_F_and_phase(F, f"A{A:.1f}_B{B:.1f}_C{C:.1f}_D{D:.1f}"):
                    found = True
                    break
            if found:
                break
        if found:
            break
    if found:
        break

if not found:
    print("\n  No tristability found with method 1")

# 方法2: 5次多项式
# F(x) = a*x^5 + b*x^4 + c*x^3 + d*x^2 + e*x + f
print("\n2. Testing: 5th degree polynomial")

found = False
for a in np.linspace(-10, 10, 8):
    for b in np.linspace(-10, 10, 8):
        for c in np.linspace(-10, 10, 8):
            for d in np.linspace(-5, 5, 8):
                for e in np.linspace(-5, 5, 8):
                    for f in np.linspace(-2, 2, 8):
                        def F(x):
                            return a*x**5 + b*x**4 + c*x**3 + d*x**2 + e*x + f
                        
                        if plot_F_and_phase(F, f"poly_{a:.1f}_{b:.1f}_{c:.1f}"):
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                break
        if found:
            break
    if found:
        break

if not found:
    print("\n  No tristability found with method 2")

# 方法3: 组合多个高斯
print("\n3. Testing: Multiple Gaussian wells")

# 使用数值导数
def gaussian_well(x, h, mu, sigma=0.08):
    return -h * np.exp(-(x-mu)**2 / (2*sigma**2))

def gaussian_force(x, h, mu, sigma=0.08):
    """F = -dV/dx"""
    return -h * np.exp(-(x-mu)**2 / (2*sigma**2)) * (-(x-mu) / sigma**2)

found = False
for h1 in np.linspace(0.3, 2.0, 8):
    for h2 in np.linspace(0.3, 2.0, 8):
        for h3 in np.linspace(0.3, 2.0, 8):
            for shift in np.linspace(-0.3, 0.3, 6):
                def F(x):
                    # 三个峰 + 线性位移
                    return (gaussian_force(x, h1, 0.2 + shift) + 
                            gaussian_force(x, h2, 0.5) + 
                            gaussian_force(x, h3, 0.8 - shift))
                
                if plot_F_and_phase(F, f"Gauss_h{h1:.1f}_{h2:.1f}_{h3:.1f}_s{shift:.1f}"):
                    found = True
                    break
            if found:
                break
        if found:
            break
    if found:
        break

if not found:
    print("\n  No tristability found with method 3")

# 方法4: 简单的分段设计
# 直接构造: F(x) 在 0.2, 0.5, 0.8 附近为负(吸引)，中间为正(排斥)
print("\n4. Testing: Designed triple-well")

# F(x) = k * (x-0.2)*(0.5-x)*(x-0.8) - c
# 这是一个三次函数，在0.2和0.8附近为正，在0.5附近为负

found = False
for k in np.linspace(5, 30, 15):
    for c in np.linspace(0.1, 1.0, 20):
        def F(x):
            # (x-0.2)*(0.5-x)*(x-0.8) = -(x-0.2)(x-0.5)(x-0.8)
            return -k * (x-0.2) * (x-0.5) * (x-0.8) - c
        
        if plot_F_and_phase(F, f"triple_k{k:.1f}_c{c:.1f}"):
            found = True
            break
    
    if found:
        break

if not found:
    print("\n  No tristability found")

# 方法5: 尝试不同的三次多项式
# F(x) = k*(x-a)*(x-b)*(x-c) - d
# 三个根 a,b,c 在 [0,1] 内
print("\n5. Testing: Cubic with roots in [0,1]")

found = False
for a in np.linspace(0.05, 0.25, 6):
    for b in np.linspace(0.35, 0.55, 6):
        for c in np.linspace(0.70, 0.90, 6):
            for k in np.linspace(8, 25, 10):
                for d in np.linspace(0.1, 0.8, 15):
                    def F(x):
                        return k * (x-a) * (x-b) * (x-c) - d
                    
                    if plot_F_and_phase(F, f"cubic_a{a:.2f}_b{b:.2f}_c{c:.2f}"):
                        found = True
                        break
                if found:
                    break
            if found:
                break
        if found:
            break
    if found:
        break

if not found:
    print("\n  No tristability found with any method")

print("\n" + "=" * 60)
print("Search Complete")
print("=" * 60)
