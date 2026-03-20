"""
Q1: λ_c(L) 缩放定律研究
========================
目标：测量 k=3,4 对 L=1,2,3,4,5 时的 WTA (Winner-Take-All) 临界耦合 λ_c

WTA 定义: N_att(λ) = 2 (只剩 all-HIGH 和 all-LOW)
使用二分搜索找到精确的 λ_c

预计时间:
- k=3, L=1-5: ~10-30秒每个配置
- k=4, L=1-5: ~30-120秒每个配置
- 总计: ~20-60分钟
"""

import numpy as np
import json
import os
import sys
import io
import time
from math import comb
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from scipy.stats.qmc import Sobol
from sklearn.cluster import DBSCAN

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, 'F:/hypergraph_bistability/src')
os.makedirs('F:/hypergraph_bistability/results/Q1_lambda_c_scaling', exist_ok=True)

ALPHA = 1.0
A = 0.5
DBSCAN_EPS = 0.08
CONVERGE_TOL = 1e-4
T_SETTLE = 80.0
N_SOBOL = 512


def ode(t, M_flat, k, L, lam, mu):
    M = M_flat.reshape(k, L)
    dM = np.zeros_like(M)
    for i in range(k):
        for l in range(L):
            Mc = M[i, l]
            bistable = ALPHA * Mc * (1-Mc) * (Mc - A)
            gc = sum(lam * (np.mean(M[j,:]) - Mc) for j in range(k) if j != i)
            lc = sum(mu * (M[i,l2] - Mc) for l2 in range(L) if l2 != l)
            dM[i, l] = bistable + gc + lc
    return dM.flatten()


def is_converged(M_flat, k, L, lam, mu):
    dM = ode(0, M_flat, k, L, lam, mu)
    return np.linalg.norm(dM) < CONVERGE_TOL


def make_initials(k, L, n_sobol=N_SOBOL):
    d = k * L
    n_pow2 = 1
    while n_pow2 < n_sobol:
        n_pow2 *= 2
    sampler = Sobol(d=d, scramble=True)
    base = sampler.random(n_pow2) * 0.8 + 0.1
    extras = [np.full(d, 0.05), np.full(d, 0.95)]
    for g in range(k):
        v = np.full((k, L), 0.05)
        v[g, :] = 0.95
        extras.append(v.flatten())
    for l in range(L):
        v = np.full((k, L), 0.05)
        v[:, l] = 0.95
        extras.append(v.flatten())
    return np.vstack([base, extras])


def detect_attractors(finals, eps=DBSCAN_EPS):
    if len(finals) < 1:
        return []
    db = DBSCAN(eps=eps, min_samples=1).fit(finals)
    labels = db.labels_
    centers = [finals[labels == c].mean(axis=0) for c in np.unique(labels) if c >= 0]
    return centers


def count_attractors(k, L, lam, mu, n_sobol=N_SOBOL, t_settle=T_SETTLE):
    """Run experiment and return attractor count."""
    initials = make_initials(k, L, n_sobol)
    finals = []
    for M0 in initials:
        sol = solve_ivp(
            lambda t, M: ode(t, M, k, L, lam, mu),
            [0, t_settle], M0,
            method='RK45', rtol=1e-4, atol=1e-6,
            dense_output=False
        )
        Mf = sol.y[:, -1]
        if is_converged(Mf, k, L, lam, mu):
            finals.append(Mf)
    if not finals:
        return 0, []
    centers = detect_attractors(np.array(finals))
    return len(centers), centers


def count_by_nhigh(centers, k, L, threshold=0.5):
    """Count attractors by n_high (number of groups in HIGH state)."""
    n_high_counts = {n: 0 for n in range(k+1)}
    for c in centers:
        M = c.reshape(k, L)
        group_means = np.mean(M, axis=1)
        n_high = np.sum(group_means > threshold)
        n_high_counts[n_high] += 1
    return n_high_counts


def find_wta_threshold(k, L, mu=0.0, verbose=True):
    """
    Binary search for the smallest λ where N_att(λ) = 2 (WTA state).
    Returns λ_c and the N_att values at key points.
    """
    if verbose:
        print(f"  k={k}, L={L}, mu={mu}:", end=" ", flush=True)
    
    lam_lo, lam_hi = 0.0, 0.5
    n_att_lo = 0
    n_att_hi = 0
    
    for _ in range(60):
        mid = (lam_lo + lam_hi) / 2
        n_att, _ = count_attractors(k, L, mid, mu)
        
        if n_att > 2:
            lam_lo = mid
            n_att_lo = n_att
        else:
            lam_hi = mid
            n_att_hi = n_att
    
    lam_c = (lam_lo + lam_hi) / 2
    n_att_c = n_att_lo
    
    if verbose:
        print(f"λ_c ≈ {lam_c:.6f} (N_att(λ<λ_c)={n_att_lo}, N_att(λ>λ_c)={n_att_hi})")
    
    return lam_c, n_att_lo


def find_wta_threshold_fast(k, L, mu=0.0, verbose=True):
    """
    Faster version: scan λ values first, then binary search.
    """
    if verbose:
        print(f"  k={k}, L={L}, mu={mu}:", end=" ", flush=True)
    
    lam_grid = [0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.13, 0.15, 0.20, 0.30]
    
    n_att_grid = {}
    for lam in lam_grid:
        n_att, _ = count_attractors(k, L, lam, mu)
        n_att_grid[lam] = n_att
        if verbose:
            print(f"λ={lam:.3f}:{n_att}", end=" ", flush=True)
    
    wta_lams = [lam for lam in lam_grid if n_att_grid[lam] <= 2]
    
    if not wta_lams:
        if verbose:
            print(f"  → WTA not found in [0.001, 0.3]")
        return None, n_att_grid
    
    lam_wta_min = min(wta_lams)
    
    if lam_wta_min == min(lam_grid):
        if verbose:
            print(f"  → WTA at smallest λ tested")
        return lam_wta_min, n_att_grid
    
    idx = lam_grid.index(lam_wta_min)
    lam_below = lam_grid[idx - 1]
    
    lam_lo, lam_hi = lam_below, lam_wta_min
    for _ in range(50):
        mid = (lam_lo + lam_hi) / 2
        n_att, _ = count_attractors(k, L, mid, mu)
        if n_att > 2:
            lam_lo = mid
        else:
            lam_hi = mid
    
    lam_c = (lam_lo + lam_hi) / 2
    
    if verbose:
        print(f"  → λ_c ≈ {lam_c:.6f}")
    
    return lam_c, n_att_grid


def main():
    print("=" * 60)
    print("Q1: λ_c(L) 缩放定律研究")
    print("=" * 60)
    
    results = {}
    
    configs = [
        (3, 1), (3, 2), (3, 3),
        (4, 1), (4, 2),
    ]
    
    for k, L in configs:
        print(f"\n[Config] k={k}, L={L}")
        t0 = time.time()
        lam_c, n_att_grid = find_wta_threshold_fast(k, L, mu=0.0, verbose=True)
        elapsed = time.time() - t0
        print(f"  → 耗时: {elapsed:.1f}秒")
        
        results[f"k{k}_L{L}"] = {
            "k": k,
            "L": L,
            "lambda_c": lam_c,
            "n_att_grid": {str(k): v for k, v in n_att_grid.items()},
            "elapsed_seconds": elapsed
        }
    
    print("\n" + "=" * 60)
    print("汇总: λ_c(L)")
    print("=" * 60)
    print(f"{'Config':12s} {'L':4s} {'λ_c':>10s}")
    print("-" * 30)
    for key, val in results.items():
        lc = val['lambda_c']
        lc_str = f"{lc:.6f}" if lc is not None else "N/A"
        print(f"{key:12s} {val['L']:4d} {lc_str:>10s}")
    
    with open('F:/hypergraph_bistability/results/Q1_lambda_c_scaling/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\n[Saved] results/Q1_lambda_c_scaling/results.json")
    
    print("\n理论分析:")
    print("λ_c 是否随 L 变化？如果变化，变化趋势是什么？")
    
    k3_results = [(r['L'], r['lambda_c']) for r in [results[f"k3_L{L}"] for L in [1,2,3]] if r['lambda_c'] is not None]
    k4_results = [(r['L'], r['lambda_c']) for r in [results[f"k4_L{L}"] for L in [1,2]] if r['lambda_c'] is not None]
    
    if len(k3_results) >= 2:
        print(f"\nk=3: L={k3_results[0][0]} → λ_c={k3_results[0][1]:.4f}")
        for i in range(1, len(k3_results)):
            ratio = k3_results[i][1] / k3_results[0][1] if k3_results[0][1] > 0 else float('nan')
            print(f"     L={k3_results[i][0]} → λ_c={k3_results[i][1]:.4f} (比值: {ratio:.2f})")
    
    if len(k4_results) >= 2:
        print(f"\nk=4: L={k4_results[0][0]} → λ_c={k4_results[0][1]:.4f}")
        for i in range(1, len(k4_results)):
            ratio = k4_results[i][1] / k4_results[0][1] if k4_results[0][1] > 0 else float('nan')
            print(f"     L={k4_results[i][0]} → λ_c={k4_results[i][1]:.4f} (比值: {ratio:.2f})")


if __name__ == '__main__':
    main()