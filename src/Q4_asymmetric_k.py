"""
Q4: 非对称 k 研究
=================
目标：验证非对称 group 容量是否影响 λ_c 或 N_att

方法：
- 在 ODE 中给不同 group 不同 A_i 值（不对称）
- 测量 N_att(λ) 看是否与对称情况不同
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats.qmc import Sobol
from sklearn.cluster import DBSCAN
import json
import os

os.makedirs('F:/hypergraph_bistability/results/Q4_asymmetric_k', exist_ok=True)

ALPHA = 1.0
DBSCAN_EPS = 0.08
CONVERGE_TOL = 1e-4
T_SETTLE = 40.0
N_SOBOL = 128


def ode_asym(t, M_flat, k, L, lam, A_list):
    """非对称 bistable: 不同 group 有不同 A_i"""
    M = M_flat.reshape(k, L)
    dM = np.zeros_like(M)
    for i in range(k):
        A_i = A_list[i]
        for l in range(L):
            Mc = M[i, l]
            bistable = ALPHA * Mc * (1-Mc) * (Mc - A_i)
            gc = sum(lam * (np.mean(M[j,:]) - Mc) for j in range(k) if j != i)
            dM[i, l] = bistable + gc
    return dM.flatten()


def is_converged(M_flat, k, L, lam, A_list):
    dM = ode_asym(0, M_flat, k, L, lam, A_list)
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
    return np.vstack([base, extras])


def detect_attractors(finals, eps=DBSCAN_EPS):
    if len(finals) < 1:
        return []
    db = DBSCAN(eps=eps, min_samples=1).fit(finals)
    labels = db.labels_
    centers = [finals[labels == c].mean(axis=0) for c in np.unique(labels) if c >= 0]
    return centers


def count_attractors(k, L, lam, A_list, n_sobol=N_SOBOL):
    initials = make_initials(k, L, n_sobol)
    finals = []
    for M0 in initials:
        sol = solve_ivp(
            lambda t, M: ode_asym(t, M, k, L, lam, A_list),
            [0, T_SETTLE], M0,
            method='RK45', rtol=1e-4, atol=1e-6
        )
        Mf = sol.y[:, -1]
        if is_converged(Mf, k, L, lam, A_list):
            finals.append(Mf)
    if not finals:
        return 0, []
    return len(detect_attractors(np.array(finals))), finals


def main():
    print("Q4: Asymmetric k Study")
    print("=" * 50)
    
    k = 3
    L = 1
    lambda_values = [0.001, 0.01, 0.03, 0.05, 0.08, 0.1, 0.15]
    
    print("Symmetric case: A = [0.5, 0.5, 0.5]")
    A_sym = [0.5, 0.5, 0.5]
    
    print("\nAsymmetric case 1: A = [0.4, 0.5, 0.6]")
    A_asym1 = [0.4, 0.5, 0.6]
    
    print("\nAsymmetric case 2: A = [0.3, 0.5, 0.7]")
    A_asym2 = [0.3, 0.5, 0.7]
    
    print("\n" + "-" * 50)
    print(f"{'lambda':>8s} | {'sym':>6s} | {'asym1':>6s} | {'asym2':>6s}")
    print("-" * 50)
    
    results = {'sym': {}, 'asym1': {}, 'asym2': {}}
    
    for lam in lambda_values:
        n_sym, _ = count_attractors(k, L, lam, A_sym)
        n_asym1, _ = count_attractors(k, L, lam, A_asym1)
        n_asym2, _ = count_attractors(k, L, lam, A_asym2)
        
        results['sym'][str(lam)] = n_sym
        results['asym1'][str(lam)] = n_asym1
        results['asym2'][str(lam)] = n_asym2
        
        print(f"{lam:>8.3f} | {n_sym:>6d} | {n_asym1:>6d} | {n_asym2:>6d}")
    
    print("""
Interpretation:
- If N_att is similar across symmetric/asymmetric cases:
  Asymmetry does NOT affect the qualitative bifurcation structure
- If N_att differs significantly:
  Asymmetry changes the effective landscape

Note: Different A_i values shift the bistable potential depth
for each group, but the collective WTA dynamics may be robust
    """)
    
    with open('F:/hypergraph_bistability/results/Q4_asymmetric_k/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("[Saved] results/Q4_asymmetric_k/results.json")


if __name__ == '__main__':
    main()