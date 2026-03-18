"""
Dwell Time Analysis - 精确驻留时间分析
========================================
目标：在临界区(λ=0.2-0.35)测量驻留时间 τ(σ)
预期：log τ ∝ 1/σ² (Kramers形式)
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

def micro_update(m, H, ga, la, Kc, lam, sigma):
    N = len(m)
    L, k = 2, 3
    
    M = np.zeros((L, k))
    cnt = np.zeros((L, k))
    for v in range(N):
        i, l = ga[v], la[v]
        M[l, i] += m[v]
        cnt[l, i] += 1
    M = M / (cnt + 1e-8)
    
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
    
    dm = np.zeros(N)
    for v in range(N):
        i, l = ga[v], la[v]
        dm[v] = dM[l, i] + np.random.normal(0, sigma)
    return dm

def get_cluster_centers(states, th=0.12):
    """获取聚类中心"""
    centers = []
    for s in states:
        is_new = True
        for c in centers:
            if np.sqrt(np.mean((s-c)**2)) < th:
                is_new = False
                break
        if is_new:
            centers.append(s)
    return np.array(centers)

def assign_labels(states, centers, th=0.12):
    """将状态分配到最近的中心"""
    labels = []
    for s in states:
        min_dist = float('inf')
        label = -1
        for i, c in enumerate(centers):
            d = np.sqrt(np.mean((s-c)**2))
            if d < min_dist:
                min_dist = d
                label = i
        labels.append(label if min_dist < th else -1)
    return labels

def compute_dwell_times(labels):
    """Run-length encoding计算驻留时间"""
    if len(labels) < 2:
        return [], 0
    
    dwells = []
    current_label = labels[0]
    count = 1
    
    for lab in labels[1:]:
        if lab == current_label:
            count += 1
        else:
            if current_label != -1:
                dwells.append(count)
            current_label = lab
            count = 1
    if current_label != -1:
        dwells.append(count)
    
    return dwells, len(np.unique([l for l in labels if l != -1]))

# 参数
L, k = 2, 3
Kc = [0.32, 0.40, 0.48]
N, E = 100, 150
ga = np.random.randint(0, k, N)
la = np.random.randint(0, L, N)
H = gen_hypergraph(N, E)
print(f"N={N}, E={E}")

# 临界区参数
lam_vals = [0.2, 0.25, 0.3, 0.35]
sigma_vals = [0.0, 0.002, 0.005, 0.01, 0.02, 0.05]
sigmas_nonzero = [s for s in sigma_vals if s > 0]

burn = 800
T = 3000
sample_interval = 5

print("\n" + "="*60)
print("Dwell Time Analysis - 临界区")
print("="*60)

# 存储结果
all_results = {lam: {'mean': [], 'max': [], 'n_attractors': []} for lam in lam_vals}

for lam in lam_vals:
    print(f"\nλ = {lam}")
    centers_cache = None
    
    for sigma in sigma_vals:
        all_dwells = []
        n_attractors_list = []
        
        for traj in range(10):
            m = np.random.uniform(0.1, 0.9, N)
            
            # burn-in
            for _ in range(burn):
                dm = micro_update(m, H, ga, la, Kc, lam, sigma)
                m = np.clip(m + 0.05*dm, 0.01, 0.99)
            
            # 采样轨迹
            trajectory = []
            for _ in range(T):
                dm = micro_update(m, H, ga, la, Kc, lam, sigma)
                m = np.clip(m + 0.05*dm, 0.01, 0.99)
                if _ % sample_interval == 0:
                    trajectory.append(m.copy())
            
            trajectory = np.array(trajectory)
            
            # 聚类
            if centers_cache is None or sigma == 0.0:
                centers_cache = get_cluster_centers(trajectory)
            
            labels = assign_labels(trajectory, centers_cache)
            dwells, n_att = compute_dwell_times(labels)
            
            if len(dwells) > 0:
                all_dwells.extend(dwells)
            n_attractors_list.append(n_att)
        
        mean_dwell = np.mean(all_dwells) if all_dwells else 0
        max_dwell = np.max(all_dwells) if all_dwells else 0
        mean_n = np.mean(n_attractors_list)
        
        all_results[lam]['mean'].append(mean_dwell)
        all_results[lam]['max'].append(max_dwell)
        all_results[lam]['n_attractors'].append(mean_n)
        
        print(f"  σ={sigma}: mean_dwell={mean_dwell:.1f}, max_dwell={max_dwell:.1f}, n={mean_n:.1f}")

# 图1: Mean Dwell Time vs σ
print("\n[图1] Mean Dwell Time vs σ")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

colors = ['steelblue', 'coral', 'green', 'purple']
for idx, lam in enumerate(lam_vals):
    means = all_results[lam]['mean']
    axes[0].plot(sigma_vals, means, 'o-', lw=2.5, ms=7, 
                 color=colors[idx], label=f'λ={lam}')
    axes[1].plot(sigma_vals, all_results[lam]['n_attractors'], 's--', lw=2, ms=6,
                 color=colors[idx], label=f'λ={lam}')

axes[0].set_xlabel('σ (Noise)', fontsize=12)
axes[0].set_ylabel('Mean Dwell Time', fontsize=12)
axes[0].set_xscale('log')
axes[0].set_yscale('log')
axes[0].set_title('Mean Dwell Time vs Noise', fontsize=14)
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].set_xlabel('σ (Noise)', fontsize=12)
axes[1].set_ylabel('# Attractors', fontsize=12)
axes[1].set_title('Attractor Count vs Noise', fontsize=14)
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('F:/hypergraph_bistability/figures/noise/dwell_vs_sigma.png', dpi=150)
plt.close()
print("Saved: dwell_vs_sigma.png")

# 图2: Kramers-like fitting: log(τ) vs σ²
print("\n[图2] Kramers Fitting")
fig, ax = plt.subplots(figsize=(9, 6))

delta_V_estimates = []
for idx, lam in enumerate(lam_vals):
    means = all_results[lam]['mean']
    
    # 只用非零sigma
    x_data = []
    y_data = []
    for i, s in enumerate(sigma_vals):
        if s > 0 and means[i] > 0:
            x_data.append(s**2)
            y_data.append(np.log(means[i] + 1))
    
    ax.scatter(x_data, y_data, s=80, color=colors[idx], label=f'λ={lam}', zorder=5)
    
    # 线性拟合
    if len(x_data) > 2:
        z = np.polyfit(x_data, y_data, 1)
        p = np.poly1d(z)
        x_fit = np.linspace(min(x_data), max(x_data), 50)
        ax.plot(x_fit, p(x_fit), '-', alpha=0.6, color=colors[idx])
        
        # ΔV ≈ -slope (from log(τ) ~ ΔV/σ²)
        delta_V_estimates.append(-z[0])
        print(f"  λ={lam}: ΔV ≈ {-z[0]:.4f}")

ax.set_xlabel('σ²', fontsize=12)
ax.set_ylabel('log(Mean Dwell Time)', fontsize=12)
ax.set_title('Kramers Relation: log(τ) ~ σ²', fontsize=14)
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('F:/hypergraph_bistability/figures/noise/kramers_fit.png', dpi=150)
plt.close()
print("Saved: kramers_fit.png")

# 图3: ΔV vs λ
print("\n[图3] ΔV vs λ")
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(lam_vals, delta_V_estimates, 'o-', lw=2.5, ms=10, color='steelblue')
ax.set_xlabel('λ (Coupling)', fontsize=12)
ax.set_ylabel('ΔV (Effective Barrier)', fontsize=12)
ax.set_title('Barrier Height vs Coupling', fontsize=14)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('F:/hypergraph_bistability/figures/noise/delta_V.png', dpi=150)
plt.close()
print("Saved: delta_V.png")

print("\n" + "="*60)
print("完成! 3张图已保存")
print("="*60)
