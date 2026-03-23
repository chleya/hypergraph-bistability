"""
噪声驱动盆地跳跃 - 严格按照陈雷阳指导
=====================================
目标：量化噪声如何"选择性抹平"小盆地
"""
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('F:/hypergraph_bistability/figures/noise/', exist_ok=True)
np.random.seed(42)

def gen_hypergraph(N, E):
    H = np.zeros((N, E))
    for e in range(E):
        s = np.random.randint(2, 5)
        nodes = np.random.choice(N, size=s, replace=False)
        H[nodes, e] = 1
    return H

def micro_dynamics(m, H, ga, la, Kc, lam, sigma):
    """微观动力学 + 噪声"""
    N = len(m)
    L, k = 2, 3
    
    # 群体均值 M
    M = np.zeros((L, k))
    cnt = np.zeros((L, k))
    for v in range(N):
        i, l = ga[v], la[v]
        M[l, i] += m[v]
        cnt[l, i] += 1
    M = M / (cnt + 1e-8)
    
    # dM
    a = [-3.0/x for x in Kc]
    b = [4.5/x for x in Kc]
    c = [-1.5/x for x in Kc]
    
    dM = np.zeros((L, k))
    for l in range(L):
        for i in range(k):
            Mi = M[l, i]
            st = a[i]*Mi**3 + b[i]*Mi**2 + c[i]*Mi
            cr = -lam * Mi * (np.sum(M[l, :]) - Mi)
            dM[l, i] = st + cr
    
    # dm + 噪声
    dm = np.zeros(N)
    for v in range(N):
        i, l = ga[v], la[v]
        dm[v] = dM[l, i] + np.random.normal(0, sigma)
    
    return dm

def cluster_states(states, th=0.12):
    """聚类所有状态"""
    uniq = []
    for s in states:
        is_new = True
        for u in uniq:
            if np.sqrt(np.mean((s-u)**2)) < th:
                is_new = False
                break
        if is_new:
            uniq.append(s)
    return len(uniq), uniq

def run_long_trajectory(H, ga, la, Kc, lam, sigma, T=3000, burn_in=500, sample_interval=30):
    """跑长轨迹，返回采样状态"""
    N = len(ga)
    m = np.random.uniform(0.1, 0.9, N)
    
    # burn-in
    for _ in range(burn_in):
        dm = micro_dynamics(m, H, ga, la, Kc, lam, sigma)
        m = np.clip(m + 0.05 * dm, 0.01, 0.99)
    
    # 采样
    samples = []
    for _ in range(T // sample_interval):
        for _ in range(sample_interval):
            dm = micro_dynamics(m, H, ga, la, Kc, lam, sigma)
            m = np.clip(m + 0.05 * dm, 0.01, 0.99)
        samples.append(m.copy())
    
    return np.array(samples)

# 实验参数
print("=" * 50)
print("噪声盆地跳跃实验")
print("=" * 50)

L, k = 2, 3
Kc = [0.32, 0.40, 0.48]
N, E = 60, 100
ga = np.random.randint(0, k, N)
la = np.random.randint(0, L, N)
H = gen_hypergraph(N, E)
print(f"N={N}, E={E}")

# 图1: σ vs 吸引子数 @ λ=0
print("\n[图1] σ vs 吸引子数 (λ=0):")
sigmas = [0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
lam = 0.0
results1 = []

for sigma in sigmas:
    all_states = []
    for traj in range(8):
        samples = run_long_trajectory(H, ga, la, Kc, lam, sigma)
        all_states.extend(samples)
    
    n_attractors, _ = cluster_states(np.array(all_states))
    results1.append((sigma, n_attractors))
    print(f"  σ={sigma}: {n_attractors}个吸引子")

# 绘图
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot([r[0] for r in results1], [r[1] for r in results1], 'o-', 
        lw=2.5, ms=8, color='steelblue')
ax.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Single state')
ax.set_xlabel('σ (Noise Intensity)', fontsize=12)
ax.set_ylabel('Effective # Attractors', fontsize=12)
ax.set_xscale('log')
ax.set_title(f'Noise vs Attractor Survival (λ={lam}, N={N})', fontsize=14)
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('F:/hypergraph_bistability/figures/noise/fig1_sigma_vs_attractors.png', dpi=150)
plt.close()
print("\nSaved: fig1_sigma_vs_attractors.png")

# 图2: 热力图 λ vs σ
print("\n[图2] 热力图 λ vs σ:")
lam_vals = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
sigma_vals = [0.0, 0.005, 0.01, 0.02, 0.05]
heatmap = np.zeros((len(lam_vals), len(sigma_vals)))

for i, lam in enumerate(lam_vals):
    for j, sigma in enumerate(sigma_vals):
        all_states = []
        for traj in range(6):
            samples = run_long_trajectory(H, ga, la, Kc, lam, sigma)
            all_states.extend(samples)
        n = cluster_states(np.array(all_states))[0]
        heatmap[i, j] = n
        print(f"  λ={lam}, σ={sigma}: {n}")

fig, ax = plt.subplots(figsize=(9, 6))
im = ax.imshow(heatmap, cmap='viridis', aspect='auto', origin='lower', vmin=0)
ax.set_xticks(range(len(sigma_vals)))
ax.set_xticklabels([f'{s}' for s in sigma_vals])
ax.set_yticks(range(len(lam_vals)))
ax.set_yticklabels([f'{l}' for l in lam_vals])
ax.set_xlabel('σ (Noise)', fontsize=12)
ax.set_ylabel('λ (Coupling)', fontsize=12)
ax.set_title('Effective Attractors: λ vs σ', fontsize=14)
plt.colorbar(im, ax=ax, label='# Attractors')
plt.tight_layout()
plt.savefig('F:/hypergraph_bistability/figures/noise/fig2_heatmap.png', dpi=150)
plt.close()
print("\nSaved: fig2_heatmap.png")

# 图3: 不同λ下的σ曲线
print("\n[图3] 多条σ曲线:")
fig, ax = plt.subplots(figsize=(9, 5))
colors = plt.cm.viridis(np.linspace(0, 1, len(lam_vals)))

for i, lam in enumerate(lam_vals):
    curve = []
    for sigma in sigmas:
        all_states = []
        for traj in range(5):
            samples = run_long_trajectory(H, ga, la, Kc, lam, sigma)
            all_states.extend(samples)
        n = cluster_states(np.array(all_states))[0]
        curve.append(n)
    ax.plot(sigmas, curve, 'o-', lw=2, ms=6, color=colors[i], label=f'λ={lam}')

ax.set_xlabel('σ (Noise)', fontsize=12)
ax.set_ylabel('# Attractors', fontsize=12)
ax.set_xscale('log')
ax.set_title('Noise Effect at Different Coupling', fontsize=14)
ax.legend(loc='upper right')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('F:/hypergraph_bistability/figures/noise/fig3_multi_curves.png', dpi=150)
plt.close()
print("\nSaved: fig3_multi_curves.png")

print("\n" + "=" * 50)
print("完成! 3张图已保存")
print("=" * 50)
