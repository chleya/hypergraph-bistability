"""
Q2: (λ, μ) 相图研究 v2
=======================
系统性扫描 (λ, μ) 空间，找到 μ_c 相变点

关键：只有 L≥2 时 μ 才有意义！
- L=1: 无层间耦合，μ 无效应
- L≥2: μ 控制层间同步/反同步

目标:
1. k=3, L=2 完整相图 (主要)
2. k=4, L=2 验证
3. 找到 μ_c (相变临界点)
"""

import numpy as np
import json
import os
import sys
import io
import time
from scipy.integrate import solve_ivp
from scipy.stats.qmc import Sobol
from sklearn.cluster import DBSCAN

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, 'F:/hypergraph_bistability/src')
os.makedirs('F:/hypergraph_bistability/results/Q2_lambda_mu_phase', exist_ok=True)

ALPHA = 1.0
A = 0.5
DBSCAN_EPS = 0.08
CONVERGE_TOL = 1e-4
T_SETTLE = 40.0
N_SOBOL = 128


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


def scan_phase_diagram(k, L, mu_values, lambda_values):
    """Scan (λ, μ) space and return N_att matrix."""
    results = {}
    for mu in mu_values:
        results[f"{mu:.2f}"] = {}
        for lam in lambda_values:
            n_att, _ = count_attractors(k, L, lam, mu)
            results[f"{mu:.2f}"][f"{lam:.3f}"] = n_att
            print(f"  mu={mu:.2f}, lam={lam:.3f}: N_att={n_att}")
    return results


def main():
    print("=" * 60)
    print("Q2: (λ, μ) 相图研究")
    print("=" * 60)

    mu_values = [-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
    lambda_values = [0.001, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15]

    configs = [
        (3, 2, "k3_L2"),
        (3, 3, "k3_L3"),
    ]

    all_results = {}

    for k, L, name in configs:
        print(f"\n[{name}] {'-'*40}")
        t0 = time.time()

        results = scan_phase_diagram(k, L, mu_values, lambda_values)

        elapsed = time.time() - t0
        print(f"  耗时: {elapsed:.1f}秒")

        all_results[name] = {
            "k": k,
            "L": L,
            "N_att_matrix": results,
            "elapsed_seconds": elapsed
        }

    print("\n" + "=" * 60)
    print("汇总: N_att(λ, μ)")
    print("=" * 60)

    for k, L, name in configs:
        print(f"\n{name} (k={k}, L={L}):")
        print(f"{'mu\lambda':>8s}", end="")
        for lam in lambda_values:
            print(f"{lam:>7.3f}", end="")
        print()
        print("-" * (8 + 7 * len(lambda_values)))
        for mu in mu_values:
            print(f"{mu:>8.2f}", end="")
            for lam in lambda_values:
                n = all_results[name]['N_att_matrix'][f"{mu:.2f}"][f"{lam:.3f}"]
                print(f"{n:>7d}", end="")
            print()

    print("\n" + "=" * 60)
    print("相图解读")
    print("=" * 60)
    print("""
μ < 0:  反同步，层间耦合对抗 group coupling
       → 需要更大 λ 才能达到 WTA
       → 可能完全抑制 WTA (N_att > 2 即使 λ很大)

μ = 0:  无层间耦合，原始相变

μ > 0:  同步，层间耦合增强 group coupling 效果
       → 需要更小 λ 达到 WTA
       → 加速相变

μ_c:    临界点，μ 改变相变性质的分界
""")

    with open('F:/hypergraph_bistability/results/Q2_lambda_mu_phase/results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print("[Saved] results/Q2_lambda_mu_phase/results.json")


if __name__ == '__main__':
    main()