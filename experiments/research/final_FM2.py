"""
高分辨率 F(M) 测量 + 可视化
"""

import numpy as np
import random
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

def run_system(N, K, initial_M, steps=500, seed=42):
    """运行系统并返回最终的 M"""
    random.seed(seed)
    np.random.seed(seed)
    
    V = list(range(N))
    E = []
    faction = {}
    n_edges = N // 3
    
    # 创建初始状态
    if initial_M >= 0.8:
        for i in range(n_edges):
            e = frozenset(random.sample(V, min(3, len(V))))
            E.append(e)
            faction[e] = 0
    elif initial_M >= 0.5:
        for i in range(n_edges):
            e = frozenset(random.sample(V, min(3, len(V))))
            E.append(e)
            faction[e] = i % 2
    else:
        for i in range(n_edges):
            e = frozenset(random.sample(V, min(3, len(V))))
            E.append(e)
            faction[e] = i % 3
    
    s = {v: np.random.randn(4) for v in V}
    
    for t in range(steps):
        avg_dist = 0.5 + 0.5 * random.random()
        
        # 支化
        if random.random() < 0.35 and E:
            e = random.choice(E)
            x = random.choice(list(e))
            w = len(V)
            V.append(w)
            s[w] = s[x] + np.random.randn(4) * 0.2
            new_e = frozenset([x, w])
            E.append(new_e)
            faction[new_e] = faction.get(e, random.randint(0, 2))
        
        # 融合
        if random.random() < 0.3 and len(E) >= 2:
            e1, e2 = random.sample(E, 2)
            if len(e1 & e2) >= 1:
                f1, f2 = faction.get(e1, 0), faction.get(e2, 0)
                fp = 0.3 if f1 != f2 else 0.0
                c1 = np.mean([s[u] for u in e1], axis=0)
                c2 = np.mean([s[u] for u in e2], axis=0)
                if np.linalg.norm(c1 - c2) > 0.8 * avg_dist - fp:
                    for v in e1: s[v] += np.random.randn(4) * 0.3
                    for v in e2: s[v] -= np.random.randn(4) * 0.3
                else:
                    new_e = frozenset(e1 | e2)
                    if new_e not in E:
                        E.remove(e1); E.remove(e2); E.append(new_e)
                        faction[new_e] = (f1 + f2) % 3
        
        # 分裂
        if random.random() < 0.25 and E:
            e = random.choice(E)
            if len(e) >= 5:
                lst = list(e)
                mid = len(lst) // 2
                e_left, e_right = frozenset(lst[:mid]), frozenset(lst[mid:])
                f = faction.get(e, 0)
                E.remove(e); E.append(e_left); E.append(e_right)
                faction[e_left] = f; faction[e_right] = (f + 1) % 3
        
        # 消除
        if random.random() < 0.15 and E:
            e = random.choice(E)
            if len(e) <= 2 and random.random() < 0.5:
                E.remove(e)
                if e in faction: del faction[e]
        
        # K约束
        if t % 50 == 0:
            for v in V:
                d = sum(1 for e in E if v in e)
                if d > K:
                    excess = d - K
                    edges = [e for e in E if v in e]
                    edges.sort(key=lambda e: len(e), reverse=True)
                    for _ in range(min(excess, len(edges))):
                        if edges:
                            e = edges.pop()
                            if e in E:
                                E.remove(e)
                                if e in faction: del faction[e]
    
    if E:
        counts = Counter(faction.values())
        return max(counts.values()) / len(E)
    return 0


# 测量 F(M)
print("=" * 60)
print("高分辨率 F(M) 测量")
print("=" * 60)

N = 50
K = int(N * 0.35)
initial_Ms = [0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
F_data = {}

for M0 in initial_Ms:
    results = []
    for run in range(25):
        M_start = run_system(N, K, M0, steps=400, seed=42+run*100+int(M0*1000))
        results.append(M_start)
    avg_M = np.mean(results)
    delta = avg_M - M0
    F_data[M0] = delta / 400
    print(f"M0={M0:.2f}: F = {F_data[M0]:+.6f}, final M = {avg_M:.2f}")

# 插值和平滑
Ms = sorted(F_data.keys())
Fs = [F_data[M] for M in Ms]
M_dense = np.linspace(0.2, 1.0, 100)
F_interp = np.interp(M_dense, Ms, Fs)
F_smooth = gaussian_filter1d(F_interp, sigma=3)

# 找不动点
zeros = []
for i in range(1, len(F_smooth)):
    if F_smooth[i-1] > 0 and F_smooth[i] < 0:
        zeros.append(M_dense[i])
    elif F_smooth[i-1] < 0 and F_smooth[i] > 0:
        zeros.append(M_dense[i])

print(f"\n不动点: {[f'{z:.2f}' for z in zeros]}")

# 画图1: F(M)
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(M_dense, F_smooth, 'b-', lw=2, label='F(M)')
ax.scatter(Ms, Fs, c='red', s=80, zorder=5, label='Data')
ax.axhline(0, color='gray', ls='--', alpha=0.5)
for z in zeros:
    ax.axvline(z, color='green', ls=':', alpha=0.7)
ax.set_xlabel('M', fontsize=12)
ax.set_ylabel('F(M)', fontsize=12)
ax.set_title('Drift Function F(M)', fontsize=14)
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('F_M_curve.png', dpi=150)
print("Saved: F_M_curve.png")

# 画图2: Basin
initial_range = np.linspace(0.1, 1.0, 15)
final_range = []
for M0 in initial_range:
    results = [run_system(N, K, M0, steps=600, seed=42+r*100) for r in range(15)]
    final_range.append(np.mean(results))

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(initial_range, final_range, 'bo-', lw=2, ms=8)
ax.plot([0,1], [0,1], 'gray', ls='--', alpha=0.5)
ax.axvline(0.6, color='red', ls=':', label='Boundary ~0.6')
ax.axhline(0.45, color='green', ls=':', alpha=0.5)
ax.axhline(0.95, color='green', ls=':', alpha=0.5)
ax.set_xlabel('Initial M', fontsize=12)
ax.set_ylabel('Final M', fontsize=12)
ax.set_title('Basin of Attraction', fontsize=14)
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('basin.png', dpi=150)
print("Saved: basin.png")

# 画图3: V(M)
V = np.zeros(len(M_dense))
for i in range(1, len(M_dense)):
    V[i] = V[i-1] - F_smooth[i] * (M_dense[i] - M_dense[i-1])
V = (V - V.min()) / (V.max() - V.min() + 1e-10)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(M_dense, V, 'purple', lw=2)
ax.fill_between(M_dense, 0, V, alpha=0.3, color='purple')
ax.set_xlabel('M', fontsize=12)
ax.set_ylabel('V(M)', fontsize=12)
ax.set_title('Effective Potential V(M)', fontsize=14)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('V_M.png', dpi=150)
print("Saved: V_M.png")

print("\n" + "=" * 60)
print("完成！三张图已保存")
print("=" * 60)
