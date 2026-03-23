"""拓扑变体实验 - 对比不同超图结构下的多稳态"""
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('F:/hypergraph_bistability/figures/topology/', exist_ok=True)
np.random.seed(42)

def gen_hypergraph(N, E):
    """均匀随机超图"""
    H = np.zeros((N, E))
    for e in range(E):
        s = np.random.randint(2, 5)
        nodes = np.random.choice(N, size=s, replace=False)
        H[nodes, e] = 1
    return H

def gen_powerlaw(N, E):
    """幂律超图"""
    H = np.zeros((N, E))
    for e in range(E):
        # 幂律选择：少数节点度数高
        if np.random.random() < 0.3:
            s = np.random.randint(4, 8)  # 大超边
        else:
            s = np.random.randint(2, 4)  # 小超边
        nodes = np.random.choice(N, size=s, replace=False)
        H[nodes, e] = 1
    return H

def gen_overlap(N, E, num_big=5, big_size=15):
    """高重叠超图"""
    H = np.zeros((N, E))
    # 普通超边
    for e in range(E - num_big):
        s = np.random.randint(2, 4)
        nodes = np.random.choice(N, size=s, replace=False)
        H[nodes, e] = 1
    # 大重叠超边
    for e in range(E - num_big, E):
        nodes = np.random.choice(N, size=big_size, replace=False)
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

def get_M(m, ga, la):
    L, k = 2, 3
    M = np.zeros((L, k))
    cnt = np.zeros((L, k))
    for v in range(len(m)):
        i, l = ga[v], la[v]
        M[l, i] += m[v]
        cnt[l, i] += 1
    return (M / (cnt + 1e-8)).flatten()

def cluster(states, th=0.2):
    if not states:
        return 0
    centers = []
    for s in states:
        found = any(np.linalg.norm(np.array(s)-np.array(c))<th for c in centers)
        if not found:
            centers.append(list(s))
    return len(centers)

# 参数
L, k = 2, 3
Kc = [0.32, 0.40, 0.48]
N, E = 80, 120
ga = np.random.randint(0, k, N)
la = np.random.randint(0, L, N)

topos = {
    'uniform': lambda: gen_hypergraph(N, E),
    'powerlaw': lambda: gen_powerlaw(N, E),
    'high_overlap': lambda: gen_overlap(N, E)
}

lam_vals = [0.0, 0.1, 0.2, 0.3, 0.4]
sigma_vals = [0.0, 0.01, 0.05]

print("拓扑变体实验")
results = {}

for tname, tfn in topos.items():
    print(f"\n[{tname}]")
    H = tfn()
    degrees = np.sum(H, axis=1)
    print(f"度数: min={degrees.min():.0f}, max={degrees.max():.0f}, mean={degrees.mean():.1f}")
    
    results[tname] = []
    for lam in lam_vals:
        row = []
        for sigma in sigma_vals:
            states = []
            for _ in range(8):
                m = np.random.uniform(0.1, 0.9, N)
                for _ in range(400):
                    dm = micro_update(m, H, ga, la, Kc, lam, sigma)
                    m = np.clip(m + 0.05*dm, 0.01, 0.99)
                for _ in range(30):
                    for _ in range(20):
                        dm = micro_update(m, H, ga, la, Kc, lam, sigma)
                        m = np.clip(m + 0.05*dm, 0.01, 0.99)
                    states.append(get_M(m, ga, la))
            n = cluster(states)
            row.append(n)
            print(f"  lam={lam}, sigma={sigma}: {n}")
        results[tname].append(row)

# 绘图
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for idx, (tname, data) in enumerate(results.items()):
    im = axes[idx].imshow(data, cmap='viridis', aspect='auto', origin='lower', vmin=2, vmax=12)
    axes[idx].set_xticks(range(len(sigma_vals)))
    axes[idx].set_xticklabels(sigma_vals)
    axes[idx].set_yticks(range(len(lam_vals)))
    axes[idx].set_yticklabels(lam_vals)
    axes[idx].set_xlabel('sigma')
    axes[idx].set_ylabel('lambda')
    axes[idx].set_title(tname)
    plt.colorbar(im, ax=axes[idx])
plt.tight_layout()
plt.savefig('F:/hypergraph_bistability/figures/topology/comparison.png', dpi=150)
plt.close()
print("\nSaved: comparison.png")

# 表格
print("\n结果:")
print("lambda", end="")
for s in sigma_vals:
    print(f"\t{s}", end="")
print()
for tname in topos.keys():
    print(f"\n{tname}")
    for li, lam in enumerate(lam_vals):
        print(f"{lam}", end="")
        for si in range(len(sigma_vals)):
            print(f"\t{results[tname][li][si]}", end="")
        print()
print("\n完成!")
