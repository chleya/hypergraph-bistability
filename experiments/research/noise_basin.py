"""
噪声驱动盆地跳跃实验
====================
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

def micro_update_noise(m, H, ga, la, Kc, lam, L, k, sigma):
    """Group-Mean Driven 微观更新 + 噪声"""
    N = len(m)
    
    # 群体均值
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
            self_term = a[i]*Mi**3 + b[i]*Mi**2 + c[i]*Mi
            cross = -lam * Mi * (np.sum(M[l, :]) - Mi)
            dM[l, i] = self_term + cross
    
    # 节点更新 + 噪声
    dm = np.zeros(N)
    for v in range(N):
        i, l = ga[v], la[v]
        dm[v] = dM[l, i]
        edges = np.where(H[v] > 0)[0]
        if len(edges) > 0:
            local = np.mean([np.mean(m[np.where(H[:,e]>0)[0]]) for e in edges])
            dm[v] += 0.02 * (local - m[v])
        # 加性噪声
        dm[v] += np.random.normal(0, sigma)
    
    return dm

def cluster(finals, th=0.1):
    uniq = []
    for s in finals:
        is_new = True
        for u in uniq:
            if np.sqrt(np.mean((s-u)**2)) < th:
                is_new = False
                break
        if is_new:
            uniq.append(s)
    return len(uniq), uniq

# 实验
print("=" * 50)
print("噪声驱动盆地跳跃实验")
print("=" * 50)

L, k = 2, 3
Kc = [0.32, 0.40, 0.48]
N, E = 50, 80
ga = np.random.randint(0, k, N)
la = np.random.randint(0, L, N)
H = gen_hypergraph(N, E)

# 测试不同sigma
sigmas = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05]
lam = 0.0  # 先测λ=0

print(f"\nλ={lam}, 不同噪声强度:")
results = []
for sigma in sigmas:
    all_states = []
    for _ in range(10):  # 10条轨迹
        m = np.random.uniform(0.1, 0.9, N)
        for _ in range(2000):  # 长轨迹
            dm = micro_update_noise(m, H, ga, la, Kc, lam, L, k, sigma)
            m = np.clip(m + 0.05 * dm, 0.01, 0.99)
        
        # 每100步采样
        for _ in range(20):
            for __ in range(100):
                dm = micro_update_noise(m, H, ga, la, Kc, lam, L, k, sigma)
                m = np.clip(m + 0.05 * dm, 0.01, 0.99)
            all_states.append(m.copy())
    
    n, _ = cluster(np.array(all_states))
    results.append((sigma, n))
    print(f"  σ={sigma}: {n}个有效吸引子")

# 绘图
fig, ax = plt.subplots(figsize=(9, 5))
sigmas_plot = [r[0] for r in results]
n_plot = [r[1] for r in results]
ax.plot(sigmas_plot, n_plot, 'o-', lw=2, ms=8, color='steelblue')
ax.set_xlabel('σ (Noise)', fontsize=12)
ax.set_ylabel('Effective # Attractors', fontsize=12)
ax.set_title(f'Noise vs Attractors (λ={lam})', fontsize=14)
ax.set_xscale('log')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('F:/hypergraph_bistability/figures/noise/noise_vs_attractors.png', dpi=150)
plt.close()
print("\nSaved: noise_vs_attractors.png")

# λ vs sigma 热力图
print("\n生成 λ-σ 热力图...")
lam_vals = [0.0, 0.2, 0.4, 0.6]
sigma_vals = [0.0, 0.005, 0.01, 0.02, 0.05]
heatmap = np.zeros((len(lam_vals), len(sigma_vals)))

for i, lam in enumerate(lam_vals):
    for j, sigma in enumerate(sigma_vals):
        all_states = []
        for _ in range(8):
            m = np.random.uniform(0.1, 0.9, N)
            for _ in range(1500):
                dm = micro_update_noise(m, H, ga, la, Kc, lam, L, k, sigma)
                m = np.clip(m + 0.05 * dm, 0.01, 0.99)
            for _ in range(15):
                for __ in range(100):
                    dm = micro_update_noise(m, H, ga, la, Kc, lam, L, k, sigma)
                    m = np.clip(m + 0.05 * dm, 0.01, 0.99)
                all_states.append(m.copy())
        
        n, _ = cluster(np.array(all_states))
        heatmap[i, j] = n

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(heatmap, cmap='viridis', aspect='auto', origin='lower')
ax.set_xticks(range(len(sigma_vals)))
ax.set_xticklabels([str(s) for s in sigma_vals])
ax.set_yticks(range(len(lam_vals)))
ax.set_yticklabels([str(l) for l in lam_vals])
ax.set_xlabel('σ (Noise)', fontsize=12)
ax.set_ylabel('λ (Coupling)', fontsize=12)
ax.set_title('Effective Attractors: λ vs σ', fontsize=14)
plt.colorbar(im, ax=ax, label='# Attractors')
plt.tight_layout()
plt.savefig('F:/hypergraph_bistability/figures/noise/heatmap_lambda_sigma.png', dpi=150)
plt.close()
print("Saved: heatmap_lambda_sigma.png")

print("\n完成!")
