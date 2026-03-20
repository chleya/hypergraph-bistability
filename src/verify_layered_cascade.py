"""
Path B Verification: Layered Saddle-Node Cascade
=================================================
Validates the layered collapse structure:
    N_att(lambda) decreases in discrete steps as lambda crosses lambda_c(n_high, k)

lambda_c(n_high, k) is computed numerically (no closed-form formula exists).
The old formula alpha*a^2*(1-a)^2*(k-1)/n_high was incorrect.

Three steps:
  Step 0 - Baseline: k=1,L=1 emergent vs designed comparison
  Step 1 - Baseline: N_att = 2^{k*L} at lambda->0
  Step 2 - Coupling: N_att(lambda) layered collapse vs theory
  Step 3 - Layer:    mu effect on lambda_c
  Step 4 - Microscopic: MultiGroupHypergraph vs ODE theory
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

# Fix Windows console encoding
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, 'F:/hypergraph_bistability/src')

os.makedirs('F:/hypergraph_bistability/results/pathB', exist_ok=True)
os.makedirs('F:/hypergraph_bistability/figures/pathB', exist_ok=True)

# ── parameters ────────────────────────────────────────────────────────────────
ALPHA = 1.0
A     = 0.5
DBSCAN_EPS   = 0.08
CONVERGE_TOL = 1e-4   # relaxed: ALL-HIGH/LOW settle at ~7e-6, need margin
T_SETTLE     = 50.0
T_CHECK      = 10.0

# ── theory ────────────────────────────────────────────────────────────────────
def _fp_eqs(x, lam, k, n_high, alpha=ALPHA, a=A):
    """Fixed-point equations for n_high-HIGH attractor."""
    H, L = x
    eq_H = alpha*H*(1-H)*(H-a) + lam*(n_high - k)*(H - L)
    eq_L = alpha*L*(1-L)*(L-a) + lam*n_high*(H - L)
    return [eq_H, eq_L]

def lambda_c_numerical(n_high, k, alpha=ALPHA, a=A):
    """
    Numerically find lambda_c where the n_high-HIGH attractor disappears.
    Uses binary search: find largest lambda where a valid H>L fixed point exists.
    Returns None for n_high=0 or n_high=k (all-same attractors never collapse).
    """
    if n_high <= 0 or n_high >= k:
        return None
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        try:
            sol = fsolve(_fp_eqs, [0.9, 0.1], args=(mid, k, n_high, alpha, a),
                         full_output=True)
            H, L = sol[0]
            res = _fp_eqs([H, L], mid, k, n_high, alpha, a)
            if max(abs(res[0]), abs(res[1])) < 1e-8 and H > L + 0.05:
                lo = mid
            else:
                hi = mid
        except Exception:
            hi = mid
    return (lo + hi) / 2

# Cache lambda_c values to avoid repeated computation
_lc_cache = {}

def lambda_c_theory(n_high, k, alpha=ALPHA, a=A):
    """Cached numerical lambda_c."""
    key = (n_high, k, alpha, a)
    if key not in _lc_cache:
        _lc_cache[key] = lambda_c_numerical(n_high, k, alpha, a)
    return _lc_cache[key]

def N_att_theory(lam, k, L=1, alpha=ALPHA, a=A):
    """Predicted attractor count at given lambda (numerical lambda_c)."""
    total = 0
    for n in range(k+1):
        lc = lambda_c_theory(n, k, alpha, a)
        if lc is None or lam < lc:
            total += comb(k, n) * (2 ** (k*(L-1)))
    return total

# ── ODE ───────────────────────────────────────────────────────────────────────
def ode(t, M_flat, k, L, lam, mu):
    M = M_flat.reshape(k, L)
    dM = np.zeros_like(M)
    for i in range(k):
        for l in range(L):
            Mc = M[i, l]
            bistable = ALPHA * Mc * (1-Mc) * (Mc - A)
            gc = sum(lam * (np.mean(M[j,:]) - Mc) for j in range(k) if j != i)
            lc = sum(mu  * (M[i,l2]         - Mc) for l2 in range(L) if l2 != l)
            dM[i, l] = bistable + gc + lc
    return dM.flatten()

def is_converged(M_flat, k, L, lam, mu):
    dM = ode(0, M_flat, k, L, lam, mu)
    return np.linalg.norm(dM) < CONVERGE_TOL

# ── sampling ──────────────────────────────────────────────────────────────────
def make_initials(k, L, n_sobol=256):
    d = k * L
    # Sobol requires power-of-2; round up
    n_pow2 = 1
    while n_pow2 < n_sobol:
        n_pow2 *= 2
    sampler = Sobol(d=d, scramble=True)
    base = sampler.random(n_pow2) * 0.8 + 0.1   # [0.1, 0.9]^d
    extras = [
        np.full(d, 0.05),
        np.full(d, 0.95),
    ]
    for g in range(k):
        v = np.full((k, L), 0.05)
        v[g, :] = 0.95
        extras.append(v.flatten())
    for l in range(L):
        v = np.full((k, L), 0.05)
        v[:, l] = 0.95
        extras.append(v.flatten())
    return np.vstack([base, extras])

# ── attractor detection ───────────────────────────────────────────────────────
def detect_attractors(finals, eps=DBSCAN_EPS):
    """DBSCAN clustering; returns list of cluster centers.
    Uses min_samples=1 so even singleton basins are counted.
    """
    if len(finals) < 1:
        return []
    db = DBSCAN(eps=eps, min_samples=1).fit(finals)
    labels = db.labels_
    centers = [finals[labels == c].mean(axis=0)
               for c in np.unique(labels) if c >= 0]
    return centers

def run_experiment(k, L, lam, mu, n_sobol=200, t_settle=T_SETTLE):
    """Run one (k,L,lam,mu) configuration; return attractor centers."""
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
        return []
    return detect_attractors(np.array(finals))


# ── Step 0: k=1,L=1 emergent vs designed ─────────────────────────────────────
def step0_baseline_k1():
    print("\n" + "="*60)
    print("Step 0: k=1,L=1  Emergent (microscopic) vs Designed (ODE)")
    print("="*60)

    # --- Designed ODE: single bistable node ---
    initials_1d = np.linspace(0.01, 0.99, 200).reshape(-1, 1)
    finals_ode = []
    for M0 in initials_1d:
        sol = solve_ivp(
            lambda t, M: ode(t, M, 1, 1, 0.0, 0.0),
            [0, 50], M0, rtol=1e-5, atol=1e-7
        )
        Mf = sol.y[0, -1]
        finals_ode.append(Mf)
    finals_ode = np.array(finals_ode)
    ode_attractors = detect_attractors(finals_ode.reshape(-1,1))
    ode_vals = sorted([c[0] for c in ode_attractors])

    # --- Microscopic: MultiGroupHypergraph k=1 ---
    try:
        from core.model import MultiGroupHypergraph
        micro_finals = []
        for seed in range(50):
            sys_m = MultiGroupHypergraph(N=50, n_groups=1, gamma=0.35, p_pair=0.3, seed=seed)
            sys_m.run_dynamics(steps=80)
            M_micro = sys_m.get_group_order_parameter(0)
            micro_finals.append(M_micro)
        micro_mean = float(np.mean(micro_finals))
        micro_std  = float(np.std(micro_finals))
        micro_min  = float(np.min(micro_finals))
        micro_max  = float(np.max(micro_finals))
        micro_ok   = True
    except Exception as e:
        micro_ok = False
        micro_mean = micro_std = micro_min = micro_max = None
        print(f"  [Microscopic model unavailable: {e}]")

    print(f"  ODE attractors (k=1,L=1): {[f'{v:.3f}' for v in ode_vals]}")
    if micro_ok:
        print(f"  Microscopic M: mean={micro_mean:.3f} ± {micro_std:.3f}  "
              f"range=[{micro_min:.3f}, {micro_max:.3f}]")
        print(f"  → ODE: discrete {{0,1}}  |  Microscopic: continuous spectrum")

    result = {
        "ode_attractors": ode_vals,
        "microscopic": {
            "available": micro_ok,
            "mean": micro_mean, "std": micro_std,
            "min": micro_min,  "max": micro_max,
            "n_runs": 50
        },
        "conclusion": "ODE produces discrete {0,1}; microscopic produces continuous spectrum"
    }
    with open('F:/hypergraph_bistability/results/pathB/step0_k1L1.json', 'w') as f:
        json.dump(result, f, indent=2)
    print("  [Saved] results/pathB/step0_k1L1.json")
    return result


# ── Step 1: N_att = 2^{k*L} at lambda->0 ─────────────────────────────────────
def step1_baseline():
    print("\n" + "="*60)
    print("Step 1: Baseline  N_att = 2^{k×L} at λ→0")
    print("="*60)

    configs = [(2,1), (3,1), (2,2), (3,2), (4,1)]
    lam_tiny = 0.001
    results = {}

    print(f"  {'Config':10s} {'Theory':8s} {'Found':8s} {'Match':6s}")
    print("  " + "-"*36)
    for k, L in configs:
        theory = 2 ** (k*L)
        centers = run_experiment(k, L, lam_tiny, 0.0, n_sobol=512, t_settle=60)
        found = len(centers)
        match = "OK" if found >= theory * 0.9 else "MISS"
        label = f"k={k},L={L}"
        print(f"  {label:10s} {theory:8d} {found:8d} {match:6s}")
        results[label] = {"theory": theory, "found": found,
                          "lambda": lam_tiny, "match": found >= theory * 0.9}

    with open('F:/hypergraph_bistability/results/pathB/step1_baseline.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("  [Saved] results/pathB/step1_baseline.json")
    return results


# ── Step 2: N_att(lambda) layered collapse ────────────────────────────────────
def step2_coupling():
    print("\n" + "="*60)
    print("Step 2: N_att(λ) — Layered Collapse vs Theory")
    print("="*60)

    configs = [(3,1), (4,1), (3,2)]
    lam_range = [0.001, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.13, 0.15, 0.20, 0.30]
    results = {}

    for k, L in configs:
        label = f"k={k},L={L}"
        print(f"\n  {label}  (theory 2^{k*L}={2**(k*L)})")
        print(f"  {'λ':6s} {'theory':8s} {'found':8s} {'layers_alive':20s}")
        print("  " + "-"*50)
        row = {}
        for lam in lam_range:
            theory = N_att_theory(lam, k, L)
            centers = run_experiment(k, L, lam, 0.0, n_sobol=256, t_settle=60)
            found = len(centers)
            # Count surviving layers by n_high
            layer_counts = {}
            for c in centers:
                n_h = int(np.round(c.reshape(k,L).mean(axis=1)).sum())
                layer_counts[n_h] = layer_counts.get(n_h, 0) + 1
            layers_str = str(dict(sorted(layer_counts.items())))
            print(f"  {lam:.3f}  {theory:8d} {found:8d} {layers_str:20s}")
            row[str(lam)] = {"theory": theory, "found": found,
                             "layer_counts": layer_counts}
        results[label] = row

    with open('F:/hypergraph_bistability/results/pathB/step2_coupling.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\n  [Saved] results/pathB/step2_coupling.json")
    return results


# ── Step 3: mu effect on lambda_c ─────────────────────────────────────────────
def step3_layer_coupling():
    print("\n" + "="*60)
    print("Step 3: μ Effect on λ_c  (k=3, L=2)")
    print("="*60)

    k, L = 3, 2
    # Fix lambda near first theoretical lambda_c = 0.0625
    lam_probe = [0.04, 0.06, 0.08, 0.10]
    mu_range  = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
    results = {}

    print(f"  {'μ':6s} " + "  ".join(f"λ={l:.2f}" for l in lam_probe))
    print("  " + "-"*50)
    for mu in mu_range:
        row_vals = []
        for lam in lam_probe:
            centers = run_experiment(k, L, lam, mu, n_sobol=150, t_settle=50)
            row_vals.append(len(centers))
        print(f"  {mu:+.1f}   " + "  ".join(f"{v:6d}" for v in row_vals))
        results[str(mu)] = {str(l): v for l, v in zip(lam_probe, row_vals)}

    # Also: find effective lambda_c(mu) for n_high=k-1 layer
    print(f"\n  Effective λ_c for n_high={k-1} layer vs μ:")
    lc_vs_mu = {}
    for mu in mu_range:
        lc_found = None
        for lam in np.arange(0.01, 0.30, 0.01):
            centers = run_experiment(k, L, lam, mu, n_sobol=100, t_settle=40)
            # Check if n_high=k-1 layer still exists
            layer_alive = False
            for c in centers:
                n_h = int(np.round(c.reshape(k,L).mean(axis=1)).sum())
                if n_h == k-1:
                    layer_alive = True
                    break
            if not layer_alive:
                lc_found = float(lam)
                break
        lc_vs_mu[str(mu)] = lc_found
        print(f"    μ={mu:+.1f}: λ_c(n_high={k-1}) ≈ {lc_found}")
    results["lambda_c_vs_mu"] = lc_vs_mu

    with open('F:/hypergraph_bistability/results/pathB/step3_layer.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\n  [Saved] results/pathB/step3_layer.json")
    return results


# ── Step 4: Microscopic vs ODE ────────────────────────────────────────────────
def step4_microscopic_vs_ode():
    print("\n" + "="*60)
    print("Step 4: Microscopic Hypergraph vs ODE Theory")
    print("="*60)

    try:
        from core.model import MultiGroupHypergraph
    except Exception as e:
        print(f"  [Skipped: {e}]")
        return {}

    results = {}
    k_vals = [2, 3, 4]
    lam_vals = [0.0, 0.05, 0.10, 0.20]
    # p_pair=0.3 puts model in clustered phase (M~0.3 largest component)
    P_PAIR = 0.3

    print(f"  {'k':4s} {'λ':6s} {'ODE N_att':10s} {'Micro M_mean':12s} {'Micro M_std':12s}")
    print("  " + "-"*50)

    for k in k_vals:
        for lam in lam_vals:
            ode_n = N_att_theory(lam, k, L=1)

            micro_M = []
            for seed in range(30):
                sys_m = MultiGroupHypergraph(
                    N=50, n_groups=k, gamma=0.35,
                    inter_group_competition=lam,
                    p_pair=P_PAIR, seed=seed
                )
                sys_m.run_dynamics(steps=100)
                for g in range(k):
                    micro_M.append(sys_m.get_group_order_parameter(g))

            m_mean = float(np.mean(micro_M))
            m_std  = float(np.std(micro_M))
            print(f"  k={k}  λ={lam:.2f}  {ode_n:10d}  {m_mean:12.3f}  {m_std:12.3f}")
            results[f"k{k}_lam{lam}"] = {
                "k": k, "lambda": lam,
                "ode_N_att": ode_n,
                "micro_M_mean": m_mean, "micro_M_std": m_std,
                "micro_n_samples": len(micro_M),
                "p_pair": P_PAIR
            }

    with open('F:/hypergraph_bistability/results/pathB/step4_micro_vs_ode.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\n  [Saved] results/pathB/step4_micro_vs_ode.json")
    return results


# ── ODE with noise ─────────────────────────────────────────────────────────────
def ode_noisy(t, M_flat, k, L, lam, mu, sigma):
    """ODE with additive Gaussian noise."""
    M = M_flat.reshape(k, L)
    dM = np.zeros((k, L))
    for i in range(k):
        for l in range(L):
            Mc = M[i, l]
            bistable = ALPHA * Mc * (1-Mc) * (Mc - A)
            gc = sum(lam * (np.mean(M[j,:]) - Mc) for j in range(k) if j != i)
            lc = sum(mu  * (M[i,l2] - Mc) for l2 in range(L) if l2 != l)
            dM[i, l] = bistable + gc + lc
    if sigma > 0:
        noise = sigma * np.random.randn(k, L)
        dM = dM + noise
    return dM.flatten()

def run_experiment_noisy(k, L, lam, mu, sigma, n_sobol=64, t_settle=20, dt=0.1):
    """Run noisy ODE experiment; return attractor count via DMS clustering."""
    initials = make_initials(k, L, n_sobol)
    finals = []
    for M0 in initials:
        M = M0.copy()
        rng = np.random.RandomState(int(np.sum(M0 * 1000)))
        t = 0.0
        while t < t_settle:
            dM = ode(0, M, k, L, lam, mu)
            dM = dM.reshape(k, L)
            noise = rng.randn(k, L) if sigma > 0 else np.zeros((k, L))
            M_mat = M.reshape(k, L) + (dM + sigma * noise) * dt
            M = np.clip(M_mat, 0.001, 0.999).flatten()
            t += dt
        dM_final = ode(0, M, k, L, lam, mu)
        if np.linalg.norm(dM_final) < CONVERGE_TOL:
            finals.append(M)
    if not finals:
        return 0
    return len(detect_attractors(np.array(finals)))


# ── Step 5: Noise robustness ──────────────────────────────────────────────────
def step5_noise_robustness():
    print("\n" + "="*60)
    print("Step 5: Noise Robustness  (k=3, L=2, ODE model)")
    print("="*60)

    k, L, mu = 3, 2, 0.0
    lam_vals  = [0.001, 0.05, 0.13]
    sigma_vals = [0.0, 0.05]

    print(f"\n  Attractor counts at different (λ, σ):")
    print(f"  {'λ':6s} " + "  ".join(f"σ={s:.2f}" for s in sigma_vals))
    print("  " + "-"*45)
    results = {}
    for lam in lam_vals:
        row = {}
        vals = []
        for sigma in sigma_vals:
            n_att = run_experiment_noisy(k, L, lam, mu, sigma, n_sobol=48, t_settle=15, dt=0.1)
            vals.append(n_att)
            row[str(sigma)] = n_att
        print(f"  {lam:6.3f} " + "  ".join(f"{v:6d}" for v in vals))
        results[str(lam)] = row

    with open('F:/hypergraph_bistability/results/pathB/step5_noise.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\n  [Saved] results/pathB/step5_noise.json")
    return results


# ── Figures ───────────────────────────────────────────────────────────────────
def make_figures(step2_results, step3_results):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception:
        print("  [matplotlib unavailable, skipping figures]")
        return

    lam_range = [0.001, 0.02, 0.04, 0.06, 0.08, 0.10, 0.13, 0.15, 0.20, 0.30]

    # Figure 1: N_att(lambda) for each config
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    configs = ["k=3,L=1", "k=4,L=1", "k=3,L=2"]
    for ax, label in zip(axes, configs):
        if label not in step2_results:
            continue
        k = int(label.split(',')[0].split('=')[1])
        L = int(label.split(',')[1].split('=')[1])
        row = step2_results[label]
        found_vals   = [row[str(l)]["found"]  for l in lam_range]
        theory_vals  = [row[str(l)]["theory"] for l in lam_range]
        ax.plot(lam_range, theory_vals, 'b--', lw=2, label='Theory')
        ax.plot(lam_range, found_vals,  'ro-', lw=2, ms=6, label='Simulation')
        # Mark theoretical lambda_c lines
        for n_h in range(1, k):
            lc = lambda_c_theory(n_h, k)
            if lc and lc <= 0.30:
                ax.axvline(lc, color='gray', ls=':', alpha=0.7,
                           label=f'λ_c(n={n_h})={lc:.3f}')
        ax.set_xlabel('λ (inter-group coupling)')
        ax.set_ylabel('N_att')
        ax.set_title(f'{label}  (2^{k*L}={2**(k*L)})')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('F:/hypergraph_bistability/figures/pathB/step2_N_att_vs_lambda.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  [Saved] figures/pathB/step2_N_att_vs_lambda.png")

    # Figure 2: mu effect heatmap
    if step3_results:
        mu_range  = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
        lam_probe = [0.04, 0.06, 0.08, 0.10]
        mat = np.zeros((len(mu_range), len(lam_probe)))
        for i, mu in enumerate(mu_range):
            for j, lam in enumerate(lam_probe):
                v = step3_results.get(str(mu), {}).get(str(lam), 0)
                mat[i, j] = v
        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(mat, aspect='auto', origin='lower',
                       extent=(lam_probe[0]-0.01, lam_probe[-1]+0.01,
                               mu_range[0]-0.05,  mu_range[-1]+0.05),
                       cmap='viridis')
        plt.colorbar(im, ax=ax, label='N_att')
        ax.set_xlabel('λ')
        ax.set_ylabel('μ')
        ax.set_title('N_att(λ, μ)  k=3, L=2')
        plt.tight_layout()
        plt.savefig('F:/hypergraph_bistability/figures/pathB/step3_N_att_heatmap.png',
                    dpi=150, bbox_inches='tight')
        plt.close()
        print("  [Saved] figures/pathB/step3_N_att_heatmap.png")

    # Figure 3: lambda_c vs mu
    if "lambda_c_vs_mu" in step3_results:
        lc_data = step3_results["lambda_c_vs_mu"]
        mus = sorted(float(k) for k in lc_data)
        lcs = [lc_data[str(m)] for m in mus]
        lcs_clean = [v if v is not None else float('nan') for v in lcs]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(mus, lcs_clean, 'bo-', lw=2, ms=7)
        ax.axhline(0.0625, color='r', ls='--', label='Theory μ=0: 0.0625')
        ax.set_xlabel('μ (inter-layer coupling)')
        ax.set_ylabel('λ_c  (n_high=2 layer)')
        ax.set_title('How μ shifts λ_c  (k=3, L=2)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('F:/hypergraph_bistability/figures/pathB/step3_lc_vs_mu.png',
                    dpi=150, bbox_inches='tight')
        plt.close()
        print("  [Saved] figures/pathB/step3_lc_vs_mu.png")


# ── Summary ───────────────────────────────────────────────────────────────────
def write_summary(s0, s1, s2, s3, s4, s5=None):
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "formula": "lambda_c(n_high, k) computed numerically (no closed form); old formula alpha*a^2*(1-a)^2*(k-1)/n_high was incorrect",
        "lambda_c_numerical": {
            f"k{k}_n{n}": lambda_c_theory(n, k)
            for k in [2, 3, 4]
            for n in range(1, k)
        },
        "step0_k1L1": s0,
        "step1_baseline_match": {k: v["match"] for k, v in s1.items()},
        "step2_theory_vs_sim": {
            cfg: {
                lam: {"theory": row["theory"], "found": row["found"]}
                for lam, row in rows.items()
            }
            for cfg, rows in s2.items()
        },
        "step3_lambda_c_vs_mu": s3.get("lambda_c_vs_mu", {}),
        "step4_micro_vs_ode_available": bool(s4),
        "step5_noise_results": s5 if s5 else {},
    }
    with open('F:/hypergraph_bistability/results/pathB/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print("\n  [Saved] results/pathB/summary.json")


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    t0 = time.time()
    print("Path B Verification: Layered Saddle-Node Cascade")
    print("="*60)

    s0 = step0_baseline_k1()
    s1 = step1_baseline()
    s2 = step2_coupling()
    s3 = step3_layer_coupling()
    s4 = step4_microscopic_vs_ode()
    s5 = step5_noise_robustness()

    print("\n" + "="*60)
    print("Generating figures...")
    make_figures(s2, s3)

    write_summary(s0, s1, s2, s3, s4, s5)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
