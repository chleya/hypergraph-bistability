"""
L→∞ 极限行为研究
=================
问题：当层数 L → ∞ 时，相变的性质是什么？

理论分析：
- L→∞ 时，group mean 的波动按 CLT 消失
- 但 μ=0 时层间无耦合
- 需要考虑两种极限顺序

数值方法：
- 测量不同 L 下，N_att(λ) 的"相变宽度"
- 定义：Δλ = λ(N_att=50%) - λ(N_att=90%)
- 观察 Δλ 随 L 的变化趋势
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats.qmc import Sobol
from sklearn.cluster import DBSCAN
import json
import os

os.makedirs('F:/hypergraph_bistability/results/L_infinity', exist_ok=True)

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
            method='RK45', rtol=1e-4, atol=1e-6
        )
        Mf = sol.y[:, -1]
        if is_converged(Mf, k, L, lam, mu):
            finals.append(Mf)
    if not finals:
        return 0, []
    return len(detect_attractors(np.array(finals))), finals


def measure_transition_width(k, L, mu, lambda_grid):
    """
    测量相变宽度：Δλ = λ(N_att=50%) - λ(N_att=90%)
    """
    n_att_values = {}
    for lam in lambda_grid:
        n_att, _ = count_attractors(k, L, lam, mu)
        n_att_values[lam] = n_att
    
    n_att_max = max(n_att_values.values())
    n_att_90 = n_att_max * 0.9
    n_att_50 = n_att_max * 0.5
    
    # 找到 N_att 刚低于 90% 和 50% 的 λ
    lambda_90 = None
    lambda_50 = None
    
    sorted_lams = sorted(lambda_grid)
    for i, lam in enumerate(sorted_lams):
        if n_att_values[lam] <= n_att_90:
            if i > 0:
                lambda_90 = (lam + sorted_lams[i-1]) / 2
            else:
                lambda_90 = lam
            break
    
    for i, lam in enumerate(sorted_lams):
        if n_att_values[lam] <= n_att_50:
            if i > 0:
                lambda_50 = (lam + sorted_lams[i-1]) / 2
            else:
                lambda_50 = lam
            break
    
    if lambda_90 and lambda_50:
        delta_lambda = lambda_50 - lambda_90
    else:
        delta_lambda = None
    
    return {
        'n_att_values': n_att_values,
        'lambda_90': lambda_90,
        'lambda_50': lambda_50,
        'delta_lambda': delta_lambda,
        'n_att_max': n_att_max
    }


def main():
    print("L→∞ 极限行为研究")
    print("=" * 50)
    
    k = 3
    L_values = [1, 2, 3, 4, 5]
    mu = 0.0
    lambda_grid = [0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.15]
    
    results = {}
    
    print(f"\n系统: k={k}, mu={mu}")
    print(f"L values: {L_values}")
    print(f"Lambda grid: {lambda_grid}")
    print()
    
    for L in L_values:
        print(f"[L={L}]", end=" ", flush=True)
        r = measure_transition_width(k, L, mu, lambda_grid)
        results[L] = r
        
        if r['delta_lambda']:
            print(f"N_max={r['n_att_max']}, Δλ={r['delta_lambda']:.4f}")
        else:
            print(f"N_max={r['n_att_max']}, Δλ=N/A")
    
    print("\n" + "=" * 50)
    print("汇总: 相变宽度 Δλ 随 L 的变化")
    print("=" * 50)
    
    print(f"{'L':>4s} | {'N_max':>6s} | {'λ_90%':>8s} | {'λ_50%':>8s} | {'Δλ':>8s}")
    print("-" * 50)
    
    for L in L_values:
        r = results[L]
        if r['lambda_90']:
            lambda_90_str = f"{r['lambda_90']:.4f}"
        else:
            lambda_90_str = "N/A"
        
        if r['lambda_50']:
            lambda_50_str = f"{r['lambda_50']:.4f}"
        else:
            lambda_50_str = "N/A"
        
        if r['delta_lambda']:
            delta_str = f"{r['delta_lambda']:.4f}"
        else:
            delta_str = "N/A"
        
        print(f"{L:>4d} | {r['n_att_max']:>6d} | {lambda_90_str:>8s} | {lambda_50_str:>8s} | {delta_str:>8s}")
    
    # 理论分析
    print("\n" + "=" * 50)
    print("理论分析")
    print("=" * 50)
    print("""
关键问题：当 L→∞ 时，相变是连续的还是离散的？

理论预测：
1. 当 L 增加，group coupling 对每层的作用被稀释
2. 相变宽度 Δλ 应该随 L 增加而增加（相变变缓）
3. 当 L→∞，Δλ → ∞（完全连续相变，无临界点）

物理解释：
- L=1: 每组只有一层，耦合直接作用 → 突变
- L=2: 每组两层，耦合被平均 → 略微平缓
- L→∞: 层数无限，耦合无限稀释 → 连续

数值验证：
- 如果 Δλ 随 L 线性或超线性增长 → 支持连续相变假说
- 如果 Δλ 收敛到有限值 → 存在有限临界 L
    """)
    
    # 检查 Δλ 是否发散
    delta_lambdas = [results[L]['delta_lambda'] for L in L_values if results[L]['delta_lambda']]
    
    if len(delta_lambdas) >= 2:
        ratio = delta_lambdas[-1] / delta_lambdas[0] if delta_lambdas[0] > 0 else float('inf')
        print(f"\nΔλ(L=5) / Δλ(L=1) = {ratio:.2f}")
        
        if ratio > 2:
            print("→ Δλ 明显增长，支持 L→∞ 时相变连续化的假说")
        else:
            print("→ Δλ 增长有限，可能存在饱和效应")
    
    with open('F:/hypergraph_bistability/results/L_infinity/results.json', 'w') as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)
    print("\n[Saved] results/L_infinity/results.json")


if __name__ == '__main__':
    main()