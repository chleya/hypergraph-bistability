"""
Extended Search
"""
import numpy as np
from scipy.integrate import odeint

def run(lam, boosts, target=(0,0), N=50):
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
        target_M = M[tl, tg]
        others = [M[l,g] for l in range(L) for g in range(k) if (l,g) != (tl, tg)]
        
        if target_M > np.mean(others) + 0.15:
            success += 1
    
    return success / N

print("Extended Parameter Search")
print("=" * 50)

# Best from before: target (1,1) got 66%
# Try to improve that

# More duration/factor combinations
print("\n1. Longer duration, stronger factor:")
for bd in [40, 50, 60, 70, 80]:
    for bf in [2.0, 2.5, 3.0]:
        boosts = [(30, bd, bf, 0)]
        s = run(0.1, boosts, target=(1,1), N=50)
        if s >= 0.60:
            print(f"   bd={bd}, bf={bf}: {s:.2f} *")

# Try different start times
print("\n2. Different start times:")
for bs in [20, 25, 30, 35, 40, 45, 50]:
    boosts = [(bs, 60, 2.5, 0)]
    s = run(0.1, boosts, target=(1,1), N=50)
    if s >= 0.60:
        print(f"   bs={bs}: {s:.2f} *")

# Try different lambda
print("\n3. Different lambda:")
for lam in [0.05, 0.08, 0.1, 0.12, 0.15]:
    boosts = [(30, 60, 2.5, 0)]
    s = run(lam, boosts, target=(1,1), N=50)
    if s >= 0.60:
        print(f"   lam={lam}: {s:.2f} *")

# Try double boost
print("\n4. Double boost:")
tests = [
    [(20, 30, 2.0, 0), (70, 30, 2.0, 0)],
    [(25, 30, 2.0, 0), (75, 30, 2.0, 0)],
    [(30, 30, 2.0, 0), (80, 30, 2.0, 0)],
    [(20, 40, 2.0, 0), (70, 40, 2.0, 0)],
]
for boosts in tests:
    s = run(0.1, boosts, target=(1,1), N=50)
    if s >= 0.60:
        print(f"   {boosts}: {s:.2f} *")

print("\n" + "=" * 50)
print("Done!")
