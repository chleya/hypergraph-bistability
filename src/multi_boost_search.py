"""
Multiple Boost Strategy - Optimized
"""
import numpy as np
from scipy.integrate import odeint

def run_multi(lam, boosts, N=50):
    Kc = [0.32, 0.40, 0.48]
    a = [-3/kc for kc in Kc]
    b = [4.5/kc for kc in Kc]
    c = [-1.5/kc for kc in Kc]
    L, k = 2, 3
    mu = 0.0
    
    def dynamics(M, t):
        M = M.reshape((L, k))
        dM = np.zeros((L, k))
        
        a_eff = a[:]
        b_eff = b[:]
        c_eff = c[:]
        
        for bs, bd, bf, tg in boosts:
            if bs <= t < bs + bd:
                a_eff[tg] /= bf
                b_eff[tg] /= bf
                c_eff[tg] /= bf
        
        for l in range(L):
            for i in range(k):
                st = a_eff[i]*M[l,i]**3 + b_eff[i]*M[l,i]**2 + c_eff[i]*M[l,i]
                cr = -lam * M[l,i] * np.sum(M[l, np.arange(k) != i])
                cl = mu * np.sum(M[np.arange(L) != l, i])
                dM[l,i] = st + cr + cl
        
        return dM.flatten()
    
    success = 0
    for _ in range(N):
        init = np.random.uniform(0.1, 0.9, L*k)
        t = np.linspace(0, 300, 300)
        sol = odeint(dynamics, init, t)
        
        M = sol[-1].reshape((L, k))
        
        best_margin = -1
        for tl in range(L):
            for tg in range(k):
                target_M = M[tl, tg]
                others = [M[l,g] for l in range(L) for g in range(k) if (l,g) != (tl, tg)]
                margin = target_M - np.mean(others)
                if margin > best_margin:
                    best_margin = margin
        
        if best_margin > 0.15:
            success += 1
    
    return success / N

print("Multiple Boost Strategy Search")
print("=" * 50)

# Baseline
print("\n0. Baseline (no boost):")
s = run_multi(0.1, [], N=50)
print(f"   No boost: {s:.2f}")

# Single strong boost
print("\n1. Single Strong Boost:")
tests = [
    ((30, 50, 2.0, 0),),
    ((40, 50, 2.0, 0),),
    ((30, 60, 2.5, 0),),
    ((40, 60, 2.5, 0),),
    ((50, 50, 2.0, 0),),
]
for boosts in tests:
    s = run_multi(0.1, boosts, N=50)
    print(f"   {boosts}: {s:.2f}")

# Double boost same target
print("\n2. Double Boost (same target):")
tests = [
    [(20, 25, 1.5, 0), (60, 25, 1.5, 0)],
    [(30, 25, 1.5, 0), (70, 25, 1.5, 0)],
    [(25, 30, 1.8, 0), (65, 30, 1.8, 0)],
]
for boosts in tests:
    s = run_multi(0.1, boosts, N=50)
    print(f"   {boosts}: {s:.2f}")

# Triple boost
print("\n3. Triple Boost:")
tests = [
    [(15, 20, 1.4, 0), (40, 20, 1.4, 1), (65, 20, 1.4, 2)],
    [(20, 20, 1.5, 0), (50, 20, 1.5, 1), (80, 20, 1.5, 2)],
    [(25, 25, 1.3, 0), (55, 25, 1.3, 1), (85, 25, 1.3, 2)],
]
for boosts in tests:
    s = run_multi(0.1, boosts, N=50)
    print(f"   {boosts}: {s:.2f}")

# Different lambda values
print("\n4. Different lambda:")
for lam in [0.05, 0.1, 0.15, 0.2]:
    boosts = [(30, 40, 2.0, 0)]
    s = run_multi(lam, boosts, N=50)
    print(f"   lambda={lam}: {s:.2f}")

print("\n" + "=" * 50)
print("Done!")
