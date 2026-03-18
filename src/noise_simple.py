"""
噪声驱动盆地跳跃 - 简化版
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

def micro_update(m, H, ga, la, Kc, lam, L, k, sigma):
    N = len(m)
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
        dm[v] = dM[l, i]
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
    return len(uniq)

# 实验
print("噪声实验")
L, k = 2, 3
Kc = [0.32, 0.40, 0.48]
N, E = 40, 60
ga = np.random.randint(0, k, N)
la = np.random.randint(0, L, N)
H = gen_hypergraph(N, E)

# 不同sigma @ λ=0
sigmas = [0.0, 0.005, 0.01, 0.02, 0.05]
print("\nλ=0:")
res = []
for sigma in sigmas:
    states = []
    for _ in range(5):
        m = np.random.uniform(0.1, 0.9, N)
        for _ in range(800):
            dm = micro_update(m, H, ga, la, Kc, 0.0, L, k, sigma)
            m = np.clip(m + 0.05*dm, 0.01, 0.99)
        states.append(m.copy())
    n = cluster(np.array(states))
    res.append((sigma, n))
    print(f"  σ={sigma}: {n}")

# 绘图
fig, ax = plt.subplots()
ax.plot([r[0] for r in res], [r[1] for r in res], 'o-', lw=2)
ax.set_xlabel('σ')
ax.set_ylabel('# Attractors')
ax.set_xscale('log')
plt.savefig('F:/hypergraph_bistability/figures/noise/n_vs_sigma.png', dpi=120)
plt.close()
print("\nSaved n_vs_sigma.png")

# 热力图
lam_vals = [0.0, 0.3, 0.6]
sigma_vals = [0.0, 0.01, 0.03]
hm = np.zeros((len(lam_vals), len(sigma_vals)))

for i, lam in enumerate(lam_vals):
    for j, sigma in enumerate(sigma_vals):
        states = []
        for _ in range(5):
            m = np.random.uniform(0.1, 0.9, N)
            for _ in range(600):
                dm = micro_update(m, H, ga, la, Kc, lam, L, k, sigma)
                m = np.clip(m + 0.05*dm, 0.01, 0.99)
            states.append(m.copy())
        hm[i,j] = cluster(np.array(states))

fig, ax = plt.subplots(figsize=(6,4))
im = ax.imshow(hm, cmap='viridis', aspect='auto')
ax.set_xticks(range(len(sigma_vals)))
ax.set_xticklabels([str(s) for s in sigma_vals])
ax.set_yticks(range(len(lam_vals)))
ax.set_yticklabels([str(l) for l in lam_vals])
ax.set_xlabel('σ')
ax.set_ylabel('λ')
plt.colorbar(im)
plt.savefig('F:/hypergraph_bistability/figures/noise/hm.png', dpi=120)
plt.close()
print("Saved hm.png")
print("完成!")
