"""
Simple Optimal Control - Reduced Parameter Scan
"""
import numpy as np
from scipy.integrate import odeint

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

def run_boost(lam, bs, bd, bf, tl, tg, N=30):
    Kc = [0.32, 0.40, 0.48]
    a = [-3/kc for kc in Kc]
    b = [4.5/kc for kc in Kc]
    c = [-1.5/kc for kc in Kc]
    L, k = 2, 3
    mu = 0.0
    
    success = 0
    for _ in range(N):
        init = np.random.uniform(0.1, 0.9, L*k)
        
        # Phase 1: free
        t1 = np.linspace(0, bs, 100)
        sol1 = odeint(F, init, t1, args=(a, b, c, lam, mu, L, k))
        
        # Phase 2: boost
        a_b = a.copy()
        b_b = b.copy()
        c_b = c.copy()
        a_b[tg] /= bf
        b_b[tg] /= bf
        c_b[tg] /= bf
        
        t2 = np.linspace(bs, bs+bd, 100)
        sol2 = odeint(F, sol1[-1], t2, args=(a_b, b_b, c_b, lam, mu, L, k))
        
        # Phase 3: free
        t3 = np.linspace(bs+bd, 200, 100)
        sol3 = odeint(F, sol2[-1], t3, args=(a, b, c, lam, mu, L, k))
        
        M = sol3[-1].reshape((L, k))
        target_M = M[tl, tg]
        others = [M[l,g] for l in range(L) for g in range(k) if (l,g) != (tl, tg)]
        
        if target_M > np.mean(others) + 0.15:
            success += 1
    
    return success / N

# Quick scan
lam = 0.1
print("Quick parameter scan at lambda = 0.1")

# Try key combinations
tests = [
    (30, 20, 1.5, 0, 0),
    (30, 20, 2.0, 0, 0),
    (40, 30, 1.5, 0, 0),
    (40, 30, 2.0, 0, 0),
    (50, 40, 1.5, 0, 0),
    (50, 40, 2.0, 0, 0),
    (30, 20, 1.5, 1, 2),
    (40, 30, 2.0, 1, 2),
]

for bs, bd, bf, tl, tg in tests:
    s = run_boost(lam, bs, bd, bf, tl, tg, N=30)
    print(f"bs={bs}, bd={bd}, bf={bf}, target=({tl},{tg}): success={s:.2f}")

print("\nDone!")
