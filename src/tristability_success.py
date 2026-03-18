"""绘制三稳态相图"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import os

os.makedirs('F:/hypergraph_bistability/figures/2d_model', exist_ok=True)


def coupled_bistable(state, t, a, b, c):
    x, y = state
    dxdt = x * (1 - x) * (x - a) + c * (y - x)
    dydt = y * (1 - y) * (y - b) + c * (x - y)
    return [dxdt, dydt]


def find_fixed_points_brute(a, b, c, resolution=25):
    x = np.linspace(-0.1, 1.1, resolution)
    y = np.linspace(-0.1, 1.1, resolution)
    
    fixed = []
    for x0 in x:
        for y0 in y:
            dxdt, dydt = coupled_bistable([x0, y0], 0, a, b, c)
            if abs(dxdt) < 0.005 and abs(dydt) < 0.005:
                is_new = True
                for fx, fy in fixed:
                    if abs(fx - x0) < 0.05 and abs(fy - y0) < 0.05:
                        is_new = False
                        break
                if is_new:
                    fixed.append((x0, y0))
    return fixed


def analyze(fp, a, b, c):
    x, y = fp
    J11 = (1 - 2*x) * (x - a) + x * (1 - x)
    J12 = c
    J21 = c
    J22 = (1 - 2*y) * (y - b) + y * (1 - y)
    
    trace = J11 + J22
    det = J11 * J22 - J12 * J21
    
    if trace**2 - 4*det >= 0:
        l1 = (trace + np.sqrt(max(0, trace**2 - 4*det))) / 2
        l2 = (trace - np.sqrt(max(0, trace**2 - 4*det))) / 2
    else:
        real = trace / 2
        l1 = complex(real, np.sqrt(max(0, 4*det - trace**2)) / 2)
        l2 = complex(real, -np.sqrt(max(0, 4*det - trace**2)) / 2)
    
    if np.real(l1) < 0 and np.real(l2) < 0:
        return "stable"
    elif np.real(l1) > 0 or np.real(l2) > 0:
        return "unstable"
    else:
        return "saddle"


# 最佳参数
a, b, c = 0.21, 0.21, 0.05

# 相图
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

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# 流场
ax1 = axes[0]
speed = np.sqrt(U**2 + V**2)
stream = ax1.streamplot(x, y, U, V, color=speed, cmap='viridis', density=1.5)

# 固定点
fps = find_fixed_points_brute(a, b, c)
for fp in fps:
    stab = analyze(fp, a, b, c)
    if stab == "stable":
        ax1.plot(fp[0], fp[1], 'go', markersize=15, markeredgecolor='black', 
                markeredgewidth=2, zorder=10)
    elif stab == "saddle":
        ax1.plot(fp[0], fp[1], 'rx', markersize=12, markeredgecolor='black',
                markeredgewidth=2, zorder=10)

ax1.set_xlim(-0.1, 1.1)
ax1.set_ylim(-0.1, 1.1)
ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('y', fontsize=12)
ax1.set_title(f'Phase Portrait: 3 Stable Fixed Points Found!\na={a}, b={b}, c={c}', fontsize=14)
ax1.grid(True, alpha=0.3)
plt.colorbar(stream.lines, ax=ax1, label='Speed')

# 轨迹
ax2 = axes[1]
initials = []
for x0 in np.linspace(0.05, 0.95, 7):
    for y0 in np.linspace(0.05, 0.95, 7):
        initials.append((x0, y0))

t = np.linspace(0, 40, 600)
colors = plt.cm.tab20(np.linspace(0, 1, len(initials)))

for i, (x0, y0) in enumerate(initials):
    sol = odeint(coupled_bistable, [x0, y0], t, args=(a, b, c))
    ax2.plot(sol[:, 0], sol[:, 1], color=colors[i], alpha=0.5, linewidth=1.2)
    ax2.plot(sol[0, 0], sol[0, 1], 'o', color=colors[i], markersize=5)
    ax2.plot(sol[-1, 0], sol[-1, 1], 's', color=colors[i], markersize=7)

# 标记稳定点
for fp in fps:
    stab = analyze(fp, a, b, c)
    if stab == "stable":
        ax2.plot(fp[0], fp[1], 'go', markersize=15, markeredgecolor='black', 
                markeredgewidth=2, zorder=10)

ax2.set_xlim(-0.1, 1.1)
ax2.set_ylim(-0.1, 1.1)
ax2.set_xlabel('x', fontsize=12)
ax2.set_ylabel('y', fontsize=12)
ax2.set_title(f'Trajectories from Multiple Initial Conditions\nGreen circles = Stable Fixed Points', fontsize=14)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('F:/hypergraph_bistability/figures/2d_model/tristability_success.png', dpi=150)
plt.close()

# 打印结果
print("=" * 60)
print("2D TRISTABILITY SUCCESS!")
print("=" * 60)
print(f"\nParameters: a={a}, b={b}, c={c}")
print(f"\nFound {len(fps)} fixed points:")
for fp in fps:
    stab = analyze(fp, a, b, c)
    print(f"  ({fp[0]:.3f}, {fp[1]:.3f}): {stab}")

# 统计
n_stable = sum(1 for fp in fps if analyze(fp, a, b, c) == "stable")
print(f"\n*** {n_stable} STABLE FIXED POINTS! ***")

# 分析稳定点
stable_pts = [fp for fp in fps if analyze(fp, a, b, c) == "stable"]
print(f"\nStable Points:")
for pt in stable_pts:
    print(f"  ({pt[0]:.3f}, {pt[1]:.3f})")

print("\nDone!")
