"""
Jacobian Analysis for Dimension Ascension Theory
================================================

Core dynamics (bistable form with capacity constraint):
dM_{i,l}/dt = α · M_{i,l} · (Kc_i - M_{i,l}) · (M_{i,l} - a)
            + λ · Σ_{j≠i} (⟨M_j⟩ - M_{i,l})           [group coupling]
            + μ · Σ_{l'≠l} (M_{i,l'} - M_{i,l})        [layer coupling]

Where ⟨M_j⟩ = (1/L) · Σ_{l'} M_{j,l'} is group j's layer average.

Jacobian definition:
J[(i,l), (j,l')] = ∂dM_{i,l}/∂M_{j,l'}

At λ=0, μ=0: Block diagonal, D_eff = k × L
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import json
import os

os.makedirs('F:/hypergraph_bistability/figures/jacobian_analysis', exist_ok=True)
os.makedirs('F:/hypergraph_bistability/results/jacobian_analysis', exist_ok=True)


def bistable_dynamics(M_flat, t, k, L, alpha, a, Kc_list, lambda_coupling, mu_coupling):
    """
    Bistable dynamics with capacity constraint.
    
    dM_{i,l}/dt = α·M_{i,l}·(Kc_i-M_{i,l})·(M_{i,l}-a)
                 + λ·Σ_{j≠i}(⟨M_j⟩ - M_{i,l})
                 + μ·Σ_{l'≠l}(M_{i,l'} - M_{i,l})
    """
    M = M_flat.reshape(k, L)
    dMdt = np.zeros_like(M)
    
    for i in range(k):
        for l in range(L):
            Mc = M[i, l]
            Kc_i = Kc_list[i]
            
            # Bistable term with capacity constraint
            # Fixed points at M=0, M=a, M=Kc_i
            # Stable at M=0 and M=Kc_i if a ∈ (0, Kc_i)
            bistable = alpha * Mc * (Kc_i - Mc) * (Mc - a)
            
            # Group coupling (winner-take-all competition)
            group_coupling = 0
            for j in range(k):
                if j != i:
                    Mj_mean = np.mean(M[j, :])  # layer average of group j
                    group_coupling += lambda_coupling * (Mj_mean - Mc)
            
            # Layer coupling (synchronization/competition)
            layer_coupling = 0
            for l2 in range(L):
                if l2 != l:
                    layer_coupling += mu_coupling * (M[i, l2] - Mc)
            
            dMdt[i, l] = bistable + group_coupling + layer_coupling
    
    return dMdt.flatten()


def compute_jacobian_numerical(M_flat, k, L, alpha, a, Kc_list, lambda_coupling, mu_coupling, eps=1e-7):
    """
    Compute Jacobian numerically using finite differences.
    J[i,j] = (dMdt_i(M + eps·e_j) - dMdt_i(M)) / eps
    """
    n = k * L
    J = np.zeros((n, n))
    f0 = bistable_dynamics(M_flat, 0, k, L, alpha, a, Kc_list, lambda_coupling, mu_coupling)
    
    for j in range(n):
        M_perturbed = M_flat.copy()
        M_perturbed[j] += eps
        f1 = bistable_dynamics(M_perturbed, 0, k, L, alpha, a, Kc_list, lambda_coupling, mu_coupling)
        J[:, j] = (f1 - f0) / eps
    
    return J


def analyze_fixed_point(M_fixed, k, L, alpha, a, Kc_list, lambda_coupling, mu_coupling):
    """
    Analyze stability of a fixed point by computing Jacobian eigenvalues.
    Returns D_eff = number of eigenvalues with negative real part.
    """
    J = compute_jacobian_numerical(M_fixed, k, L, alpha, a, Kc_list, lambda_coupling, mu_coupling)
    eigenvalues = np.linalg.eigvals(J)
    
    # Count eigenvalues with negative real part
    n_negative = np.sum(np.real(eigenvalues) < 0)
    
    return {
        'D_eff': n_negative,
        'eigenvalues': eigenvalues,
        'max_real_part': np.max(np.real(eigenvalues)),
        'min_real_part': np.min(np.real(eigenvalues)),
        'trace': np.trace(J),
        'det': np.linalg.det(J)
    }


def find_fixed_points_gridsearch(k, L, alpha, a, Kc_list, lambda_coupling, mu_coupling, n_grid=20):
    """
    Find fixed points by grid search.
    A state M is a fixed point if dMdt ≈ 0.
    """
    M_flat = np.zeros(k * L)
    fixed_points = []
    
    # Grid search over [0, Kc_i] for each element
    M_ranges = [np.linspace(0, Kc_list[i % k], n_grid) for i in range(k * L)]
    
    # Simple approach: sample random initial conditions and converge
    for _ in range(n_grid ** 2):
        M0 = np.array([np.random.choice(Kc_list[i % k] * np.linspace(0, 1, n_grid)) 
                       for i in range(k * L)])
        
        # Simulate to convergence
        try:
            sol = solve_ivp(
                lambda t, M: bistable_dynamics(M, t, k, L, alpha, a, Kc_list, lambda_coupling, mu_coupling),
                [0, 100], M0, method='RK45', rtol=1e-10, atol=1e-12
            )
            M_final = sol.y[:, -1]
            
            # Check if it's a new fixed point
            dM = bistable_dynamics(M_final, 0, k, L, alpha, a, Kc_list, lambda_coupling, mu_coupling)
            if np.max(np.abs(dM)) < 1e-6:
                # Round to binary (0 or Kc_i) for uniqueness check
                rounded = tuple(round(m / Kc_list[i % k]) * Kc_list[i % k] for i, m in enumerate(M_final))
                if rounded not in [tuple(fp) for fp in fixed_points]:
                    fixed_points.append(rounded)
        except:
            pass
    
    return [np.array(fp) for fp in fixed_points]


def run_phase1_k2_L1():
    """
    Phase 1: k=2, L=1, scan lambda
    =================================
    Minimum nontrivial system to study coupling-induced collapse.
    """
    print("=" * 60)
    print("Phase 1: k=2, L=1 System - Lambda Scan")
    print("=" * 60)
    
    k, L = 2, 1
    alpha = 1.0
    a = 0.5
    Kc_list = [1.0, 1.0]  # Normalized capacity
    
    lambda_range = np.linspace(0, 1.0, 51)
    
    results = []
    
    for lambda_coupling in lambda_range:
        mu_coupling = 0.0  # Start with no layer coupling
        
        # Find all fixed points
        fixed_points = find_fixed_points_gridsearch(k, L, alpha, a, Kc_list, lambda_coupling, mu_coupling, n_grid=10)
        
        # For each fixed point, compute D_eff
        D_eff_values = []
        n_stable = 0
        
        for fp in fixed_points:
            analysis = analyze_fixed_point(fp, k, L, alpha, a, Kc_list, lambda_coupling, mu_coupling)
            D_eff_values.append(analysis['D_eff'])
            if analysis['max_real_part'] < 0:
                n_stable += 1
        
        # Theoretical prediction: at λ=0, should have 4 attractors (2^2)
        # At λ_c, should collapse to fewer
        results.append({
            'lambda': lambda_coupling,
            'n_fixed_points': len(fixed_points),
            'D_eff_values': D_eff_values,
            'n_stable': n_stable
        })
        
        if lambda_coupling in [0, 0.3, 0.5, 1.0]:
            print(f"λ={lambda_coupling:.1f}: n_fp={len(fixed_points)}, n_stable={n_stable}, D_eff={D_eff_values}")
    
    # Save results
    with open('F:/hypergraph_bistability/results/jacobian_analysis/phase1_k2_L1.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def verify_block_diagonal():
    """
    Verify that at λ=0, μ=0, the Jacobian is block diagonal.
    """
    print("=" * 60)
    print("Verification: Block Diagonal Structure at λ=0, μ=0")
    print("=" * 60)
    
    k, L = 2, 2
    alpha = 1.0
    a = 0.5
    Kc_list = [1.0, 1.0]
    
    # Fixed point at (0, 0, 0, 0) - all elements in LOW state
    M_fixed = np.array([0.0, 0.0, 0.0, 0.0])
    
    J = compute_jacobian_numerical(M_fixed, k, L, alpha, a, Kc_list, 
                                   lambda_coupling=0.0, mu_coupling=0.0)
    
    print("Jacobian at M=(0,0,0,0), λ=0, μ=0:")
    print(J)
    print()
    
    # Check if block diagonal
    eigenvalues = np.linalg.eigvals(J)
    print(f"Eigenvalues: {eigenvalues}")
    print(f"Real parts: {np.real(eigenvalues)}")
    print()
    
    # Verify block structure manually
    print("Block structure check (should be 4x4 with 1x1 blocks on diagonal):")
    for i in range(k * L):
        for j in range(k * L):
            if i != j and abs(J[i, j]) > 1e-10:
                print(f"  WARNING: J[{i},{j}] = {J[i,j]} (should be 0)")
    
    return J, eigenvalues


if __name__ == '__main__':
    # Verify block diagonal structure
    J, evals = verify_block_diagonal()
    
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Max |off-diagonal| = {np.max(np.abs(J - np.diag(np.diag(J))))}")
    print(f"All eigenvalues real parts negative: {all(np.real(evals) < 0)}")
    print()
    
    # Run Phase 1
    print("Starting Phase 1...")
    results = run_phase1_k2_L1()
    print(f"Phase 1 complete. Saved {len(results)} data points.")
