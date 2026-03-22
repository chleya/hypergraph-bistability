"""
k=1, L=1 Baseline Experiment
============================

验证当 k=1, L=1, λ=0, μ=0 时，系统是否只有2个吸引子。

这是验证 D_eff = kL 规律的基线测试。
"""

import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json

os.makedirs('F:/hypergraph_bistability/results/dimension_study', exist_ok=True)

from multi_stability.multi_stability_core import (
    MultiLayerMultiGroupHypergraph,
    count_stable_states
)


def run_k1_L1_baseline(n_runs=50, T=80, seed_start=42):
    """运行 k=1, L=1 基线实验"""
    
    print("=" * 60)
    print("k=1, L=1 Baseline Experiment")
    print("验证: 当 k=1, L=1, λ=0, μ=0 时，只有 2 个吸引子")
    print("=" * 60)
    print(f"n_runs={n_runs}, T={T}")
    
    final_M_values = []
    M_histories = []
    
    for run in range(n_runs):
        seed = seed_start + run * 100
        
        hg = MultiLayerMultiGroupHypergraph(
            N=50, 
            n_groups=1,   # k=1
            n_layers=1,   # L=1
            gamma=0.35,
            state_dim=16,
            inter_group_competition=0.0,  # λ=0
            inter_layer_coupling=0.0,       # μ=0
            seed=seed
        )
        
        history = hg.run_dynamics(steps=T)
        
        final_states = hg.get_all_stable_states()
        final_M = list(final_states.values())[0]
        final_M_values.append(final_M)
        
        M_histories.append(history[0][0])
        
        if (run + 1) % 10 == 0:
            print(f"  Run {run + 1}/{n_runs}")
    
    final_M_values = np.array(final_M_values)
    M_histories = np.array(M_histories)
    
    n_attractors, unique_M = count_stable_states(
        {i: m for i, m in enumerate(final_M_values)}, 
        tolerance=0.1
    )
    
    print(f"\n[Results]")
    print(f"  Total runs: {n_runs}")
    print(f"  Distinct attractors (tolerance=0.1): {n_attractors}")
    print(f"  Unique M values: {sorted(unique_M)}")
    print(f"  Mean M: {final_M_values.mean():.3f} ± {final_M_values.std():.3f}")
    
    bins = np.linspace(0, 1, 21)
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(final_M_values, bins=bins, alpha=0.7, edgecolor='black')
    for uM in unique_M:
        plt.axvline(uM, color='red', linestyle='--', linewidth=2)
    plt.xlabel('Final M')
    plt.ylabel('Count')
    plt.title(f'k=1 L=1 Baseline: Attractor Distribution\n(n={n_attractors} attractors)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    t = np.arange(T + 1)
    for i in range(min(20, n_runs)):
        if len(M_histories[i]) == len(t):
            plt.plot(t, M_histories[i], alpha=0.3, linewidth=0.5)
    plt.xlabel('Time')
    plt.ylabel('M')
    plt.title('M(t) Trajectories (first 20 runs)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = 'F:/hypergraph_bistability/figures/dimension_study/k1_L1_baseline.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n[Saved] {save_path}")
    plt.close()
    
    results = {
        'k': 1,
        'L': 1,
        'lambda': 0.0,
        'mu': 0.0,
        'n_runs': n_runs,
        'n_attractors': n_attractors,
        'unique_M': unique_M,
        'final_M_mean': float(final_M_values.mean()),
        'final_M_std': float(final_M_values.std()),
        'all_final_M': final_M_values.tolist()
    }
    
    json_path = 'F:/hypergraph_bistability/results/dimension_study/k1_L1_baseline.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"[Saved] {json_path}")
    
    return results


def run_dimension_scaling():
    """扫描不同 k, L 组合，验证 N_att ≈ 2^{kL}"""
    
    print("\n" + "=" * 60)
    print("Dimension Scaling Experiment")
    print("验证: N_att ≈ 2^{kL}")
    print("=" * 60)
    
    configs = [
        (1, 1),
        (2, 1),
        (3, 1),
        (1, 2),
        (2, 2),
        (3, 2),
    ]
    
    results = {}
    
    for k, L in configs:
        print(f"\n[k={k}, L={L}] ", end='')
        
        final_M_list = []
        n_runs = 30
        T = 80
        
        for run in range(n_runs):
            seed = 42 + run * 100
            
            hg = MultiLayerMultiGroupHypergraph(
                N=50,
                n_groups=k,
                n_layers=L,
                gamma=0.35,
                state_dim=16,
                inter_group_competition=0.0,  # λ=0
                inter_layer_coupling=0.0,       # μ=0
                seed=seed
            )
            
            history = hg.run_dynamics(steps=T)
            final_states = hg.get_all_stable_states()
            
            key = list(final_states.keys())[0]
            final_M_list.append(final_states[key])
        
        n_att, unique = count_stable_states(
            {i: m for i, m in enumerate(final_M_list)},
            tolerance=0.1
        )
        
        predicted = 2 ** (k * L)
        
        results[(k, L)] = {
            'n_attractors': n_att,
            'predicted_2_kL': predicted,
            'ratio': n_att / predicted if predicted > 0 else 0
        }
        
        print(f"N_att={n_att}, predicted=2^{k*L}={predicted}, ratio={n_att/predicted:.2f}")
    
    print("\n[Summary Table]")
    print(f"{'k':<5} {'L':<5} {'k*L':<5} {'N_att':<10} {'2^{kL}':<10} {'ratio':<10}")
    print("-" * 45)
    for (k, L), r in sorted(results.items()):
        print(f"{k:<5} {L:<5} {k*L:<5} {r['n_attractors']:<10} {r['predicted_2_kL']:<10} {r['ratio']:<10.2f}")
    
    return results


if __name__ == '__main__':
    results = run_k1_L1_baseline(n_runs=50, T=80)
    
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    if results['n_attractors'] == 2:
        print("PASS: k=1, L=1 has exactly 2 attractors (confirms 2^1 = 2)")
    else:
        print(f"NOTE: k=1, L=1 has {results['n_attractors']} attractors")
        print("Expected 2 for baseline verification")
    
    dim_results = run_dimension_scaling()
