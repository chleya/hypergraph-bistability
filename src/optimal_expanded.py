"""
Expanded Optimal Control Search
"""
import numpy as np
from scipy.integrate import odeint
import json

def F(M, t, a, b, c, lam, mu, L, k):
    M = M.reshape((L, k))
    dM = np.zeros((L, k))
    for l in range(L):
        for i in range(k):
            st = a[i]*M[l,i]**3 + b[i]*M[l,i]**2 + c[i]*M[l,i]
            cr = -lam * M[l,i] * np.sum(M[l, np.arange(k) != i])
            cl = mu * np.sum(M[np.arange(L) != l, i])
            dM[l,i] = st + cr + cl
    return dM.flatten()

def run_boost(lam, bs, bd, bf, tl, tg, N=40):
    Kc = [0.32, 0.40, 0.48]
    a = [-3/kc for kc in Kc]
    b = [4.5/kc for kc in Kc]
    c = [-1.5/kc for kc in Kc]
    L, k = 2, 3
    mu = 0.0
    
    success = 0
    for _ in range(N):
        init = np.random.uniform(0.1, 0.9, L*k)
        
        t1 = np.linspace(0, bs, 100)
        sol1 = odeint(F, init, t1, args=(a, b, c, lam, mu, L, k))
        
        a_b = a.copy()
        b_b = b.copy()
        c_b = c.copy()
        a_b[tg] /= bf
        b_b[tg] /= bf
        c_b[tg] /= bf
        
        t2 = np.linspace(bs, bs+bd, 100)
        sol2 = odeint(F, sol1[-1], t2, args=(a_b, b_b, c_b, lam, mu, L, k))
        
        t3 = np.linspace(bs+bd, 200, 100)
        sol3 = odeint(F, sol2[-1], t3, args=(a, b, c, lam, mu, L, k))
        
        M = sol3[-1].reshape((L, k))
        target_M = M[tl, tg]
        others = [M[l,g] for l in range(L) for g in range(k) if (l,g) != (tl, tg)]
        
        if target_M > np.mean(others) + 0.15:
            success += 1
    
    return success / N

# Expanded search
lam = 0.1
print("Expanded parameter scan at lambda = 0.1")

# More combinations based on what worked
tests = []

# Based on previous: bs=40, bd=30, bf=1.5 gave 63%
# Let's explore around that
for bs in [35, 40, 45]:
    for bd in [25, 30, 35]:
        for bf in [1.3, 1.5, 1.7]:
            for tl in [0, 1]:
                for tg in [0, 1, 2]:
                    tests.append((bs, bd, bf, tl, tg))

results = []
best_s = 0
best_p = None

for i, (bs, bd, bf, tl, tg) in enumerate(tests):
    s = run_boost(lam, bs, bd, bf, tl, tg, N=40)
    results.append({'bs': bs, 'bd': bd, 'bf': bf, 'tl': tl, 'tg': tg, 's': s})
    if s > best_s:
        best_s = s
        best_p = (bs, bd, bf, tl, tg)
    
    if (i+1) % 20 == 0:
        print(f"Progress: {i+1}/{len(tests)}, best so far: {best_s:.2f}")

print(f"\nBest: {best_p} -> {best_s:.2f}")

# Analyze by parameter
print("\nAnalysis by parameter:")

# Start
starts = {}
for r in results:
    bs = r['bs']
    if bs not in starts: starts[bs] = []
    starts[bs].append(r['s'])
print("\nBoost Start:")
for bs in sorted(starts.keys()):
    print(f"  {bs}: {np.mean(starts[bs]):.3f}")

# Duration
durs = {}
for r in results:
    bd = r['bd']
    if bd not in durs: durs[bd] = []
    durs[bd].append(r['s'])
print("\nBoost Duration:")
for bd in sorted(durs.keys()):
    print(f"  {bd}: {np.mean(durs[bd]):.3f}")

# Factor
facts = {}
for r in results:
    bf = r['bf']
    if bf not in facts: facts[bf] = []
    facts[bf].append(r['s'])
print("\nBoost Factor:")
for bf in sorted(facts.keys()):
    print(f"  {bf}: {np.mean(facts[bf]):.3f}")

# Save results
with open('F:/hypergraph_bistability/results/optimal_control.json', 'w') as f:
    json.dump({'best': best_p, 'best_s': best_s, 'results': results}, f)

print("\nDone!")
