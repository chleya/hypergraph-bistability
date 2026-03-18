"""
Dwell Time Analysis v2 - 修复聚类问题
========================================
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

def simple_cluster_m(state):
    """用group-level mean作为状态标识"""
    L, k = 2, 3
    M = np.zeros((L, k))
    cnt = np.zeros((L, k))
    for v in range(len(state)):
        i = ga[v]
        l = la[v]
        M[l, i] += state[v]
        cnt[l, i] += 1
    M = M / (cnt + 1e-8)
    return tuple(M.flatten())

# 参数
L, k = 2, 3
Kc = [0.32, 0.40, 0.48]
N, E = 100, 150
ga = np.random.randint(0, k, N)
la = np.random.randint(0, L, N)
H = gen_hypergraph(N, E)
print(f"N={N}, E={E}")

# 临界区参数 - 使用更宽的λ范围
lam_vals = [0.0, 0.1, 0.2, 0.3, 0.4]
sigma_vals = [0.0, 0.005, 0.01, 0.02, 0.05]

burn = 500
T = 2000

print("\n" + "="*60)
print("Dwell Time Analysis v2 - 简单聚类")
print("="*60)

results = {}

for lam in lam_vals:
    print(f"\nλ = {lam}")
    results[lam] = {'attractors': [], 'transitions': []}
    
    for sigma in sigma_vals:
        attractor_keys = []
        
        for traj in range(15):
            m = np.random.uniform(0.1, 0.9, N)
            
            # burn-in
            for _ in range(burn):
                dm = micro_update(m, H, ga, la, Kc, lam, sigma)
                m = np.clip(m + 0.05*dm, 0.01, 0.99)
            
            # 采样 - 记录每个状态的group-level mean
            states = []
            for _ in range(T):
                dm = micro_update(m, H, ga, la, Kc, lam, sigma)
                m = np.clip(m + 0.05*dm, 0.01, 0.99)
                states.append(simple_cluster_m(m))
            
            # 统计独特吸引子
            uniq = list(set(states))
            attractor_keys.extend(uniq)
        
        # 计算这个(λ, σ)下的吸引子数量
        n_attractors = len(set(attractor_keys))
        results[lam]['attractors'].append(n_attractors)
        
        # 计算转换次数
        transitions = 0
        for traj in range(15):
            m = np.random.uniform(0.1, 0.9, N)
            for _ in range(burn):
                dm = micro_update(m, H, ga, la, Kc, lam, sigma)
                m = np.clip(m + 0.05*dm, 0.01, 0.99)
            
            prev_state = simple_cluster_m(m)
            for _ in range(T):
                dm = micro_update(m, H, ga, la, Kc, lam, sigma)
                m = np.clip(m + 0.05*dm, 0.01, 0.99)
                curr_state = simple_cluster_m(m)
                if curr_state != prev_state:
                    transitions += 1
                prev_state = curr_state
        
        results[lam]['transitions'].append(transitions / 15)
        
        print(f"  σ={sigma}: attractors={n_attractors}, avg_transitions={transitions/15:.1f}")

# 绘图
print("\n[绘图]")

# 图1: Attractors vs σ
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

colors = ['steelblue', 'coral', 'green', 'purple', 'orange']
for idx, lam in enumerate(lam_vals):
    axes[0].plot(sigma_vals, results[lam]['attractors'], 'o-', lw=2.5, ms=8,
                 color=colors[idx], label=f'λ={lam}')

axes[0].set_xlabel('σ (Noise)', fontsize=12)
axes[0].set_ylabel('# Attractors', fontsize=12)
axes[0].set_title('Attractor Count vs Noise', fontsize=14)
axes[0].legend()
axes[0].grid(alpha=0.3)

# 图2: Transitions vs σ
for idx, lam in enumerate(lam_vals):
    axes[1].plot(sigma_vals, results[lam]['transitions'], 's--', lw=2, ms=7,
                 color=colors[idx], label=f'λ={lam}')

axes[1].set_xlabel('σ (Noise)', fontsize=12)
axes[1].set_ylabel('Avg Transitions', fontsize=12)
axes[1].set_title('State Transitions vs Noise', fontsize=14)
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('F:/hypergraph_bistability/figures/noise/dwell_v2.png', dpi=150)
plt.close()
print("Saved: dwell_v2.png")

# 图3: 热力图
print("\n[热力图]")
heatmap = np.array([results[lam]['attractors'] for lam in lam_vals])

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(heatmap, cmap='viridis', aspect='auto', origin='lower')
ax.set_xticks(range(len(sigma_vals)))
ax.set_xticklabels([f'{s}' for s in sigma_vals])
ax.set_yticks(range(len(lam_vals)))
ax.set_yticklabels([f'{l}' for l in lam_vals])
ax.set_xlabel('σ (Noise)', fontsize=12)
ax.set_ylabel('λ (Coupling)', fontsize=12)
ax.set_title(f'Attractor Count Heatmap (N={N})', fontsize=14)
plt.colorbar(im, ax=ax, label='# Attractors')
plt.tight_layout()
plt.savefig('F:/hypergraph_bistability/figures/noise/heatmap_dwell.png', dpi=150)
plt.close()
print("Saved: heatmap_dwell.png")

print("\n" + "="*60)
print("完成!")
print("="*60)
