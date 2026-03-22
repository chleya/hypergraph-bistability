"""
Q3: Noise Escape Rate - Fast Version
=====================================
"""

import numpy as np
import os
import json

os.makedirs('F:/hypergraph_bistability/results/Q3_noise_escape', exist_ok=True)

ALPHA = 1.0
A = 0.5
K = 3
L = 1
DT = 0.2
T_MAX = 150.0


def euler_noise(M_flat, k, L, lam, sigma):
    M = M_flat.reshape(k, L)
    dM = np.zeros_like(M)
    for i in range(k):
        for l in range(L):
            Mc = M[i, l]
            bistable = ALPHA * Mc * (1-Mc) * (Mc - A)
            gc = sum(lam * (np.mean(M[j,:]) - Mc) for j in range(k) if j != i)
            dM[i, l] = bistable + gc
    noise = sigma * np.random.randn(k, L)
    return np.clip(M + (dM + noise) * DT, 0, 1).flatten()


def run_traj(M0, k, L, lam, sigma):
    M = M0.copy()
    for _ in range(int(T_MAX / DT)):
        M = euler_noise(M, k, L, lam, sigma)
    return M


def measure_p_escape(k, L, lam, sigma, n=15):
    M0 = np.full((k, L), 0.95)
    count = 0
    for i in range(n):
        np.random.seed(i * 777 + 42)
        M_final = run_traj(M0.flatten(), k, L, lam, sigma)
        if np.mean(M_final) < 0.3:
            count += 1
    return count / n


def main():
    print("Q3: Noise Escape Rate")
    print("=" * 40)
    
    lam_list = [0.05, 0.1, 0.2]
    sigma_list = [0.0, 0.1, 0.2, 0.3, 0.5]
    
    results = {}
    
    for lam in lam_list:
        results[str(lam)] = {}
        for sig in sigma_list:
            p = measure_p_escape(K, L, lam, sig, n=15)
            results[str(lam)][str(sig)] = p
            print(f"lam={lam:.2f}, sig={sig:.2f}: p_escape={p:.0%}")
    
    print("\nTable:")
    print(f"{'lam':>6s}", end="")
    for sig in sigma_list:
        print(f"{sig:>7.2f}", end="")
    print()
    
    for lam in lam_list:
        print(f"{lam:>6.2f}", end="")
        for sig in sigma_list:
            p = results[str(lam)][str(sig)]
            print(f"{p:>7.0%}", end="")
        print()
    
    with open('F:/hypergraph_bistability/results/Q3_noise_escape/results.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()