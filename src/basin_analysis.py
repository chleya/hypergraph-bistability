"""Better noise analysis - cluster states to find basins"""
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

def get_group_means(m, ga, la):
    """Get group-level means as state fingerprint"""
    L, k = 2, 3
    M = np.zeros((L, k))
    cnt = np.zeros((L, k))
    for v in range(len(m)):
        i, l = ga[v], la[v]
        M[l, i] += m[v]
        cnt[l, i] += 1
    M = M / (cnt + 1e-8)
    return M.flatten()

def cluster_states(states, threshold=0.15):
    """Simple clustering to count basins"""
    if len(states) == 0:
        return 0
    centers = []
    for s in states:
        found = False
        for c in centers:
            if np.linalg.norm(np.array(s) - np.array(c)) < threshold:
                found = True
                break
        if not found:
            centers.append(list(s))
    return len(centers)

# Setup
L, k = 2, 3
Kc = [0.32, 0.40, 0.48]
N, E = 80, 120
ga = np.random.randint(0, k, N)
la = np.random.randint(0, L, N)
H = gen_hypergraph(N, E)
print(f"N={N}, E={E}")

# Test with better clustering
lam_vals = [0.0, 0.1, 0.2, 0.3, 0.4]
sigma_vals = [0.0, 0.005, 0.01, 0.02, 0.05]

print("Testing basins (clustered):")
results = {}
for lam in lam_vals:
    results[lam] = []
    for sigma in sigma_vals:
        # Run multiple trajectories
        all_states = []
        for traj in range(10):
            m = np.random.uniform(0.1, 0.9, N)
            # Burn in
            for _ in range(400):
                dm = micro_update(m, H, ga, la, Kc, lam, sigma)
                m = np.clip(m + 0.05*dm, 0.01, 0.99)
            # Sample - collect group means
            for _ in range(30):
                for _ in range(20):
                    dm = micro_update(m, H, ga, la, Kc, lam, sigma)
                    m = np.clip(m + 0.05*dm, 0.01, 0.99)
                all_states.append(get_group_means(m, ga, la))
        
        n_basins = cluster_states(all_states, threshold=0.2)
        results[lam].append(n_basins)
        print(f"  lam={lam}, sigma={sigma}: {n_basins} basins")

# Plot heatmap
print("\nGenerating heatmap...")
heatmap = np.array([results[lam] for lam in lam_vals])

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(heatmap, cmap='viridis', aspect='auto', origin='lower')
ax.set_xticks(range(len(sigma_vals)))
ax.set_xticklabels([f'{s}' for s in sigma_vals])
ax.set_yticks(range(len(lam_vals)))
ax.set_yticklabels([f'{l}' for l in lam_vals])
ax.set_xlabel('sigma')
ax.set_ylabel('lambda')
ax.set_title('Basin Count: lambda vs sigma')
plt.colorbar(im, ax=ax, label='# basins')
plt.tight_layout()
plt.savefig('F:/hypergraph_bistability/figures/noise/basin_heatmap.png', dpi=150)
plt.close()
print("Saved: basin_heatmap.png")

# Print table
print("\nResults table:")
print("lambda\\sigma", end="")
for s in sigma_vals:
    print(f"\t{s}", end="")
print()
for lam in lam_vals:
    print(f"{lam}", end="")
    for n in results[lam]:
        print(f"\t{n}", end="")
    print()

print("\nDone!")
