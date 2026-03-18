"""
噪声实验 - 放大N版本 (N=150)
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

def cluster_states(states, th=0.1):
    uniq = []
    for s in states:
        is_new = True
        for u in uniq:
            if np.sqrt(np.mean((s-u)**2)) < th:
                is_new = False
                break
        if is_new:
            uniq.append(s)
    return len(uniq)

# 放大N
print("噪声实验 - N=150")
L, k = 2, 3
Kc = [0.32, 0.40, 0.48]
N, E = 150, 200  # 放大N
ga = np.random.randint(0, k, N)
la = np.random.randint(0, L, N)
H = gen_hypergraph(N, E)
print(f"N={N}, E={E}")

# 测试不同λ和σ
lam_vals = [0.0, 0.1, 0.2, 0.3, 0.4]
sigma_vals = [0.0, 0.005, 0.01, 0.02, 0.05]
heatmap = np.zeros((len(lam_vals), len(sigma_vals)))

for i, lam in enumerate(lam_vals):
    for j, sigma in enumerate(sigma_vals):
        all_states = []
        for traj in range(8):
            m = np.random.uniform(0.1, 0.9, N)
            # burn-in
            for _ in range(600):
                dm = micro_update(m, H, ga, la, Kc, lam, sigma)
                m = np.clip(m + 0.05*dm, 0.01, 0.99)
            # 采样
            for _ in range(40):
                for _ in range(30):
                    dm = micro_update(m, H, ga, la, Kc, lam, sigma)
                    m = np.clip(m + 0.05*dm, 0.01, 0.99)
                all_states.append(m.copy())
        
        n = cluster_states(np.array(all_states))
        heatmap[i, j] = n
        print(f"  λ={lam}, σ={sigma}: {n}")

# 绘图
fig, ax = plt.subplots(figsize=(9, 6))
im = ax.imshow(heatmap, cmap='viridis', aspect='auto', origin='lower', vmin=0)
ax.set_xticks(range(len(sigma_vals)))
ax.set_xticklabels([f'{s}' for s in sigma_vals])
ax.set_yticks(range(len(lam_vals)))
ax.set_yticklabels([f'{l}' for l in lam_vals])
ax.set_xlabel('σ (Noise)', fontsize=12)
ax.set_ylabel('λ (Coupling)', fontsize=12)
ax.set_title(f'Attractors: λ vs σ (N={N})', fontsize=14)
plt.colorbar(im, ax=ax, label='# Attractors')
plt.tight_layout()
plt.savefig('F:/hypergraph_bistability/figures/noise/heatmap_N150.png', dpi=150)
plt.close()
print("\nSaved: heatmap_N150.png")

# 单条曲线
print("\nλ=0曲线:")
sigmas = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1]
curve = []
for sigma in sigmas:
    all_states = []
    for traj in range(8):
        m = np.random.uniform(0.1, 0.9, N)
        for _ in range(600):
            dm = micro_update(m, H, ga, la, Kc, 0.0, sigma)
            m = np.clip(m + 0.05*dm, 0.01, 0.99)
        for _ in range(40):
            for _ in range(30):
                dm = micro_update(m, H, ga, la, Kc, 0.0, sigma)
                m = np.clip(m + 0.05*dm, 0.01, 0.99)
            all_states.append(m.copy())
    n = cluster_states(np.array(all_states))
    curve.append(n)
    print(f"  σ={sigma}: {n}")

fig, ax = plt.subplots(figsize=(8,5))
ax.plot(sigmas, curve, 'o-', lw=2.5, ms=8, color='steelblue')
ax.set_xlabel('σ')
ax.set_ylabel('# Attractors')
ax.set_xscale('log')
ax.set_title(f'Noise vs Attractors (λ=0, N={N})')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('F:/hypergraph_bistability/figures/noise/curve_N150.png', dpi=150)
plt.close()
print("Saved: curve_N150.png")
print("\n完成!")
