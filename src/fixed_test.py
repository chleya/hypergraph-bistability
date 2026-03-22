"""
Fixed: Check SPECIFIC target success
"""
import numpy as np
from scipy.integrate import odeint

def run_fixed(lam, boosts, target=(0,0), N=50):
    """
    Check if SPECIFIC target achieves dominance.
    """
    Kc = [0.32, 0.40, 0.48]
    a = [-3/kc for kc in Kc]
    b = [4.5/kc for kc in Kc]
    c = [-1.5/kc for kc in Kc]
    L, k = 2, 3
    mu = 0.0
    tl, tg = target
    
    def dynamics(M, t):
        M = M.reshape((L, k))
        dM = np.zeros((L, k))
        
        a_eff = a[:]
        b_eff = b[:]
        c_eff = c[:]
        
        for bs, bd, bf, tg_idx in boosts:
            if bs <= t < bs + bd:
                a_eff[tg_idx] /= bf
                b_eff[tg_idx] /= bf
                c_eff[tg_idx] /= bf
        
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
        
        # Check SPECIFIC target
        target_M = M[tl, tg]
        others = [M[l,g] for l in range(L) for g in range(k) if (l,g) != (tl, tg)]
        
        if target_M > np.mean(others) + 0.15:
            success += 1
    
    return success / N

print("Fixed: Check Specific Target")
print("=" * 50)

# Baseline (no boost)
print("\n0. Baseline (no boost):")
for target in [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)]:
    s = run_fixed(0.1, [], target=target, N=50)
    print(f"   target {target}: {s:.2f}")

# Single strong boost for different targets
print("\n1. Single Boost (target 0,0):")
tests = [
    ((30, 60, 2.5, 0),),
    ((40, 60, 2.5, 0),),
    ((50, 60, 2.5, 0),),
]
for boosts in tests:
    s = run_fixed(0.1, boosts, target=(0,0), N=50)
    print(f"   {boosts}: {s:.2f}")

# Different targets
print("\n2. Single Boost for different targets:")
boosts = ((40, 60, 2.5, 0),)
for target in [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)]:
    s = run_fixed(0.1, boosts, target=target, N=50)
    print(f"   target {target}: {s:.2f}")

# Try different boost targets
print("\n3. Boost different targets:")
for boost_target in [0, 1, 2]:
    boosts = ((40, 60, 2.5, boost_target),)
    s = run_fixed(0.1, boosts, target=(0,0), N=50)
    print(f"   boost target {boost_target}: {s:.2f}")

print("\n" + "=" * 50)
print("Done!")
