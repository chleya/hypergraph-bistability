"""
Optimal Control - Fine Parameter Search
"""
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import os

os.makedirs('F:/hypergraph_bistability/figures/control', exist_ok=True)

def hypergraph_dynamics(M_flat, t, a_list, b_list, c_list, lam, mu, L, k):
    M = M_flat.reshape((L, k))
    dM = np.zeros((L, k))
    for l in range(L):
        for i in range(k):
            st = a_list[i]*M[l,i]**3 + b_list[i]*M[l,i]**2 + c_list[i]*M[l,i]
            cr = -lam * M[l,i] * np.sum(M[l, np.arange(k) != i])
            cl = mu * np.sum(M[np.arange(L) != l, i])
            dM[l,i] = st + cr + cl
    return dM.flatten()

def run_with_boost(lam, Kc_list, bs, bd, bf, tl, tg, N_init=50):
    L, k = 2, 3
    mu = 0.0
    a_list = [-3/kc for kc in Kc_list]
    b_list = [4.5/kc for kc in Kc_list]
    c_list = [-1.5/kc for kc in Kc_list]
    
    success = 0
    for _ in range(N_init):
        init = np.random.uniform(0.1, 0.9, L*k)
        
        t1 = np.linspace(0, bs, 200)
        sol1 = odeint(hypergraph_dynamics, init, t1, args=(a_list, b_list, c_list, lam, mu, L, k))
        
        a_b = a_list.copy()
        b_b = b_list.copy()
        c_b = c_list.copy()
        a_b[tg] /= bf
        b_b[tg] /= bf
        c_b[tg] /= bf
        
        t2 = np.linspace(bs, bs+bd, 150)
        sol2 = odeint(hypergraph_dynamics, sol1[-1], t2, args=(a_b, b_b, c_b, lam, mu, L, k))
        
        t3 = np.linspace(bs+bd, 250, 300)
        sol3 = odeint(hypergraph_dynamics, sol2[-1], t3, args=(a_list, b_list, c_list, lam, mu, L, k))
        
        M = sol3[-1].reshape((L, k))
        target_M = M[tl, tg]
        others = [M[l,g] for l in range(L) for g in range(k) if (l,g) != (tl, tg)]
        
        if target_M > np.mean(others) + 0.15:
            success += 1
    
    return success / N_init

print("=" * 60)
print("Optimal Control - Fine Parameter Search")
print("=" * 60)

Kc_list = [0.32, 0.40, 0.48]
lam = 0.1

boost_starts = [20, 30, 40, 50, 60]
boost_durations = [15, 25, 35, 45]
boost_factors = [1.2, 1.4, 1.6, 1.8, 2.0]

best_success = 0
best_params = None
results = []

total = len(boost_starts) * len(boost_durations) * len(boost_factors) * 6
count = 0

for bs in boost_starts:
    for bd in boost_durations:
        for bf in boost_factors:
            for tl in range(2):
                for tg in range(3):
                    count += 1
                    if count % 100 == 0:
                        print(f"Progress: {count}/{total}")
                    
                    s = run_with_boost(lam, Kc_list, bs, bd, bf, tl, tg)
                    results.append({'bs': bs, 'bd': bd, 'bf': bf, 'tl': tl, 'tg': tg, 's': s})
                    
                    if s > best_success:
                        best_success = s
                        best_params = (bs, bd, bf, tl, tg)

print("\n" + "=" * 60)
print("Best Result")
print("=" * 60)
print(f"start = {best_params[0]}")
print(f"duration = {best_params[1]}")
print(f"factor = {best_params[2]}")
print(f"target = ({best_params[3]}, {best_params[4]})")
print(f"success = {best_success:.2f}")

# Analysis
print("\nParameter Effects:")

# Start
starts = {}
for r in results:
    bs = r['bs']
    if bs not in starts: starts[bs] = []
    starts[bs].append(r['s'])

print("\nBoost Start:")
for bs in sorted(starts.keys()):
    print(f"  {bs}: {np.mean(starts[bs]):.2f}")

# Duration
durs = {}
for r in results:
    bd = r['bd']
    if bd not in durs: durs[bd] = []
    durs[bd].append(r['s'])

print("\nBoost Duration:")
for bd in sorted(durs.keys()):
    print(f"  {bd}: {np.mean(durs[bd]):.2f}")

# Factor
facts = {}
for r in results:
    bf = r['bf']
    if bf not in facts: facts[bf] = []
    facts[bf].append(r['s'])

print("\nBoost Factor:")
for bf in sorted(facts.keys()):
    print(f"  {bf}: {np.mean(facts[bf]):.2f}")

print("\nDone!")
