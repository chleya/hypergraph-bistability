"""
噪声盆地跳跃 - 优化版 (更大N)
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

def cluster(finals, th=0.12):
    uniq = []
    for s in finals:
        is_new = True
        for u in uniq:
            if np.sqrt(np.mean((s-u)**2)) < th:
                is_new = False
                break
        if is_new:
            uniq.append(s)
    return len(uniq)

# 更大N
print("噪声实验 - 优化版 (N=60)")
L, k = 2, 3
Kc = [0.32, 0.40, 0.48]
N, E = 60, 100
ga = np.random.randint(0, k, N)
la = np.random.randint(0, L, N)
H = gen_hypergraph(N, E)

# 测试更高噪声范围
sigmas = [0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
lam = 0.0

print(f"\nλ={lam}:")
results = []
for sigma in sigmas:
    all_s = []
    for _ in range(10):
        m = np.random.uniform(0.1, 0.9, N)
        for _ in range(600):
            dm = micro_update(m, H, ga, la, Kc, lam, sigma)
            m = np.clip(m + 0.05*dm, 0.01, 0.99)
        for _ in range(40):
            for _ in range(40):
                dm = micro_update(m, H, ga, la, Kc, lam, sigma)
                m = np.clip(m + 0.05*dm, 0.01, 0.99)
            all_s.append(m.copy())
    n = cluster(np.array(all_s))
    results.append((sigma, n))
    print(f"  σ={sigma}: {n}")

# 绘图
fig, ax = plt.subplots(figsize=(9,5))
ax.plot([r[0] for r in results], [r[1] for r in results], 'o-', lw=2.5, ms=8, color='steelblue')
ax.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Single state')
ax.set_xlabel('σ (Noise)', fontsize=12)
ax.set_ylabel('# Attractors', fontsize=12)
ax.set_xscale('log')
ax.set_title(f'Noise vs Attractors (λ={lam}, N={N})', fontsize=14)
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('F:/hypergraph_bistability/figures/noise/noise_N60.png', dpi=150)
plt.close()
print("\nSaved: noise_N60.png")

# 热力图 - 更密
print("\n热力图...")
lam_vals = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
sigma_vals = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1]
hm = np.zeros((len(lam_vals), len(sigma_vals)))

for i, lam in enumerate(lam_vals):
    for j, sigma in enumerate(sigma_vals):
        all_s = []
        for _ in range(8):
            m = np.random.uniform(0.1, 0.9, N)
            for _ in range(500):
                dm = micro_update(m, H, ga, la, Kc, lam, sigma)
                m = np.clip(m + 0.05*dm, 0.01, 0.99)
            for _ in range(30):
                for _ in range(30):
                    dm = micro_update(m, H, ga, la, Kc, lam, sigma)
                    m = np.clip(m + 0.05*dm, 0.01, 0.99)
                all_s.append(m.copy())
        hm[i,j] = cluster(np.array(all_s))
        print(f"  λ={lam}, σ={sigma}: {int(hm[i,j])}")

fig, ax = plt.subplots(figsize=(9,6))
im = ax.imshow(hm, cmap='viridis', aspect='auto', origin='lower', vmin=0, vmax=15)
ax.set_xticks(range(len(sigma_vals)))
ax.set_xticklabels([f'{s}' for s in sigma_vals])
ax.set_yticks(range(len(lam_vals)))
ax.set_yticklabels([f'{l}' for l in lam_vals])
ax.set_xlabel('σ (Noise)', fontsize=12)
ax.set_ylabel('λ (Coupling)', fontsize=12)
ax.set_title(f'Attractors: λ vs σ (N={N})', fontsize=14)
plt.colorbar(im, ax=ax, label='# Attractors')
plt.tight_layout()
plt.savefig('F:/hypergraph_bistability/figures/noise/heatmap_N60.png', dpi=150)
plt.close()
print("\nSaved: heatmap_N60.png")
print("完成!")
