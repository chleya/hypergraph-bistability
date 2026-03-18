"""
三稳态模型 - 简化和最终版本
"""


import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('F:/hypergraph_bistability/figures/tristability', exist_ok=True)


def F_quartic(x, k=30, c=0.4):
    """四次多项式: F(x) = k*(x-a)(x-b)(x-c)(x-d) - e"""
    return k * (x - 0.15) * (x - 0.4) * (x - 0.6) * (x - 0.85) - c


def find_roots(F, x_range, n=3000):
    """找零点"""
    x = np.linspace(x_range[0], x_range[1], n)
    y = F(x)
    
    roots = []
    for i in range(len(x) - 1):
        if y[i] * y[i+1] < 0:
            roots.append((x[i] + x[i+1]) / 2)
    return roots


def check_stability(F, root):
    """检查稳定性"""
    h = 0.0001
    dF = (F(root + h) - F(root - h)) / (2 * h)
    return dF < 0


print("=" * 60)
print("Quartic Polynomial Analysis")
print("=" * 60)

# 分析四次多项式
x = np.linspace(0, 1, 500)
y = F_quartic(x)

plt.figure(figsize=(12, 8))

# 子图1: F(x)
plt.subplot(2, 2, 1)
plt.plot(x, y, 'b-', linewidth=2)
plt.axhline(y=0, color='k', linestyle='--')

# 找根
roots = find_roots(F_quartic, [0, 1])
print(f"\nRoots: {roots}")

for r in roots:
    stable = check_stability(F_quartic, r)
    print(f"  x={r:.4f}, stable={stable}")
    if stable:
        plt.plot(r, 0, 'go', markersize=15, markeredgecolor='black', markeredgewidth=2)
    else:
        plt.plot(r, 0, 'rx', markersize=15, markeredgecolor='black', markeredgewidth=2)

plt.xlabel('x')
plt.ylabel('F(x)')
plt.title('F(x) = k*(x-0.15)(x-0.4)(x-0.6)(x-0.85) - c')
plt.grid(True, alpha=0.3)
plt.xlim(-0.02, 1.02)

# 子图2: 相图
plt.subplot(2, 2, 2)

# 固定k=30, 扫描c
c_values = np.linspace(0.1, 0.8, 25)
x0_values = np.linspace(0.02, 0.98, 30)

final_grid = np.zeros((len(c_values), len(x0_values)))

for i, c in enumerate(c_values):
    for j, x0 in enumerate(x0_values):
        # 动力学模拟
        x = x0
        for _ in range(80):
            F_val = 30 * (x - 0.15) * (x - 0.4) * (x - 0.6) * (x - 0.85) - c
            x = x + 0.1 * F_val
            x = np.clip(x, 0.001, 0.999)
        final_grid[i, j] = x

plt.imshow(final_grid, aspect='auto', origin='lower',
           extent=[0, 1, c_values[0], c_values[-1]], cmap='RdYlBu_r')
plt.colorbar(label='Final x')
plt.xlabel('Initial x0')
plt.ylabel('c')
plt.title('Phase Diagram')

# 子图3: 区域占比
plt.subplot(2, 2, 3)

low_frac = []
mid_frac = []
high_frac = []

for i, c in enumerate(c_values):
    low = sum(1 for j in range(len(x0_values)) if final_grid[i, j] < 0.3)
    mid = sum(1 for j in range(len(x0_values)) if 0.3 <= final_grid[i, j] < 0.7)
    high = sum(1 for j in range(len(x0_values)) if final_grid[i, j] >= 0.7)
    total = len(x0_values)
    low_frac.append(low / total)
    mid_frac.append(mid / total)
    high_frac.append(high / total)

plt.plot(c_values, low_frac, 'b-o', label='LOW', markersize=4)
plt.plot(c_values, mid_frac, 'g-s', label='MID', markersize=4)
plt.plot(c_values, high_frac, 'r-^', label='HIGH', markersize=4)
plt.xlabel('c')
plt.ylabel('Fraction')
plt.title('Basin Fraction vs c')
plt.legend()
plt.grid(True, alpha=0.3)

# 子图4: F(x)随c变化
plt.subplot(2, 2, 4)

for c in [0.2, 0.3, 0.4, 0.5, 0.6]:
    y = 30 * (x - 0.15) * (x - 0.4) * (x - 0.6) * (x - 0.85) - c
    plt.plot(x, y, label=f'c={c}', alpha=0.7)

plt.axhline(y=0, color='k', linestyle='--')
plt.xlabel('x')
plt.ylabel('F(x)')
plt.title('F(x) with Different c')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('F:/hypergraph_bistability/figures/tristability/final_analysis.png', dpi=150)
plt.close()

print("\nFigure saved!")

# 检查是否有三稳态
print("\n" + "=" * 60)
print("Stability Analysis Summary")
print("=" * 60)

for c in [0.2, 0.3, 0.4, 0.5]:
    def F(x):
        return 30 * (x - 0.15) * (x - 0.4) * (x - 0.6) * (x - 0.85) - c
    
    roots = find_roots(F, [0, 1])
    stable = [r for r in roots if check_stability(F, r)]
    print(f"\nc={c}:")
    print(f"  All roots: {[f'{r:.3f}' for r in roots]}")
    print(f"  Stable: {[f'{r:.3f}' for r in stable]}")

print("\nDone!")
