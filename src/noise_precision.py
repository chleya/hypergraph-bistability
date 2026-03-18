"""
噪声精确量化 - Dwell Time + Kramers拟合
======================================
目标：量化噪声敏感性，建立 ΔV(λ) 理论
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

def cluster_and_label(states, th=0.12):
    """聚类并返回中心和标签"""
    centers = []
    labels = []
    for s in states:
        label = -1
        for idx, c in enumerate(centers):
            if np.sqrt(np.mean((s-c)**2)) < th:
                label = idx
                break
        if label == -1:
            centers.append(s)
            labels.append(len(centers)-1)
        else:
            labels.append(label)
    return centers, labels

def analyze_dwell_times(labels, sample_interval=10):
    """计算驻留时间 (run-length encoding)"""
    if len(labels) < 2:
        return 0, 0, 0
    
    dwells = []
    current_label = labels[0]
    count = 1
    
    for lab in labels[1:]:
        if lab == current_label:
            count += 1
        else:
            dwells.append(count * sample_interval)
            current_label = lab
            count = 1
    dwells.append(count * sample_interval)
    
    return np.mean(dwells), np.max(dwells), len(np.unique(labels))

# 实验
print("=" * 60)
print("噪声精确量化 - Dwell Time + Kramers拟合")
print("=" * 60)

L, k = 2, 3
Kc = [0.32, 0.40, 0.48]
N, E = 80, 120  # 增大N
ga = np.random.randint(0, k, N)
la = np.random.randint(0, L, N)
H = gen_hypergraph(N, E)
print(f"N={N}, E={E}")

# 参数
sample_interval = 10
burn_in = 800
T = 5000

# 图1: Dwell Time vs σ @ 不同λ
print("\n[图1] 平均驻留时间 vs σ:")
sigmas = [0.0, 0.002, 0.005, 0.01, 0.02, 0.05]
lam_vals = [0.0, 0.2, 0.4]
colors = ['steelblue', 'coral', 'green']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, lam in enumerate(lam_vals):
    mean_dwells = []
    max_dwells = []
    n_visited_list = []
    
    for sigma in sigmas:
        all_states = []
        all_labels = []
        
        for traj in range(6):
            m = np.random.uniform(0.1, 0.9, N)
            
            # burn-in
            for _ in range(burn_in):
                dm = micro_update(m, H, ga, la, Kc, lam, sigma)
                m = np.clip(m + 0.05*dm, 0.01, 0.99)
            
            # 采样
            for _ in range(T // sample_interval):
                for _ in range(sample_interval):
                    dm = micro_update(m, H, ga, la, Kc, lam, sigma)
                    m = np.clip(m + 0.05*dm, 0.01, 0.99)
                all_states.append(m.copy())
        
        centers, labels = cluster_and_label(np.array(all_states))
        mean_d, max_d, n_vis = analyze_dwell_times(labels, sample_interval)
        
        mean_dwells.append(mean_d)
        max_dwells.append(max_d)
        n_visited_list.append(n_vis)
        
        print(f"  λ={lam}, σ={sigma}: mean_dwell={mean_d:.1f}, max_dwell={max_d}, n={n_vis}")
    
    axes[0].plot(sigmas, mean_dwells, 'o-', lw=2, ms=7, 
                 color=colors[idx], label=f'λ={lam}')
    axes[1].plot(sigmas, max_dwells, 's--', lw=2, ms=7, 
                 color=colors[idx], label=f'λ={lam}')

axes[0].set_xlabel('σ (Noise)', fontsize=12)
axes[0].set_ylabel('Mean Dwell Time', fontsize=12)
axes[0].set_xscale('log')
axes[0].set_yscale('log')
axes[0].set_title('Mean Dwell Time vs Noise', fontsize=14)
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].set_xlabel('σ (Noise)', fontsize=12)
axes[1].set_ylabel('Max Dwell Time', fontsize=12)
axes[1].set_xscale('log')
axes[1].set_yscale('log')
axes[1].set_title('Max Dwell Time vs Noise', fontsize=14)
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('F:/hypergraph_bistability/figures/noise/dwell_vs_sigma.png', dpi=150)
plt.close()
print("\nSaved: dwell_vs_sigma.png")

# 图2: log(dwell) vs σ² 曲线 (Kramers验证)
print("\n[图2] Kramers拟合:")
fig, ax = plt.subplots(figsize=(9, 6))

for idx, lam in enumerate(lam_vals):
    mean_dwells = []
    
    for sigma in sigmas:
        all_states = []
        for traj in range(6):
            m = np.random.uniform(0.1, 0.9, N)
            for _ in range(burn_in):
                dm = micro_update(m, H, ga, la, Kc, lam, sigma)
                m = np.clip(m + 0.05*dm, 0.01, 0.99)
            for _ in range(T // sample_interval):
                for _ in range(sample_interval):
                    dm = micro_update(m, H, ga, la, Kc, lam, sigma)
                    m = np.clip(m + 0.05*dm, 0.01, 0.99)
                all_states.append(m.copy())
        
        centers, labels = cluster_and_label(np.array(all_states))
        mean_d, _, _ = analyze_dwell_times(labels, sample_interval)
        mean_dwells.append(mean_d)
    
    # Kramers: log(τ) vs σ²
    sigma_sq = [s**2 for s in sigmas if s > 0]
    log_dwell = [np.log(d+1) for d, s in zip(mean_dwells, sigmas) if s > 0]
    
    ax.plot(sigma_sq, log_dwell, 'o-', lw=2.5, ms=8, 
            color=colors[idx], label=f'λ={lam}')
    
    # 线性拟合
    if len(sigma_sq) > 2:
        z = np.polyfit(sigma_sq, log_dwell, 1)
        p = np.poly1d(z)
        ax.plot(sigma_sq, p(sigma_sq), '--', alpha=0.5, color=colors[idx])

ax.set_xlabel('σ²', fontsize=12)
ax.set_ylabel('log(Mean Dwell Time)', fontsize=12)
ax.set_title('Kramers Relation: log(τ) ~ σ²', fontsize=14)
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('F:/hypergraph_bistability/figures/noise/kramers_fit.png', dpi=150)
plt.close()
print("Saved: kramers_fit.png")

# 图3: ΔV(λ) 估计 (从拟合斜率)
print("\n[图3] ΔV(λ) 估计:")
delta_V = []
for idx, lam in enumerate(lam_vals):
    mean_dwells = []
    for sigma in sigmas:
        all_states = []
        for traj in range(6):
            m = np.random.uniform(0.1, 0.9, N)
            for _ in range(burn_in):
                dm = micro_update(m, H, ga, la, Kc, lam, sigma)
                m = np.clip(m + 0.05*dm, 0.01, 0.99)
            for _ in range(T // sample_interval):
                for _ in range(sample_interval):
                    dm = micro_update(m, H, ga, la, Kc, lam, sigma)
                    m = np.clip(m + 0.05*dm, 0.01, 0.99)
                all_states.append(m.copy())
        centers, labels = cluster_and_label(np.array(all_states))
        mean_d, _, _ = analyze_dwell_times(labels, sample_interval)
        mean_dwells.append(mean_d)
    
    sigma_sq = [s**2 for s in sigmas if s > 0]
    log_dwell = [np.log(d+1) for d, s in zip(mean_dwells, sigmas) if s > 0]
    
    if len(sigma_sq) > 2:
        z = np.polyfit(sigma_sq, log_dwell, 1)
        # Kramers: log(τ) ≈ ΔV/σ² → slope = -ΔV
        delta_V.append(-z[0])
        print(f"  λ={lam}: ΔV ≈ {-z[0]:.3f}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(lam_vals, delta_V, 'o-', lw=2.5, ms=10, color='steelblue')
ax.set_xlabel('λ (Coupling)', fontsize=12)
ax.set_ylabel('ΔV (Barrier Height)', fontsize=12)
ax.set_title('Effective Barrier Height vs Coupling', fontsize=14)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('F:/hypergraph_bistability/figures/noise/delta_V.png', dpi=150)
plt.close()
print("Saved: delta_V.png")

print("\n" + "=" * 60)
print("完成! 3张图已保存")
print("=" * 60)
