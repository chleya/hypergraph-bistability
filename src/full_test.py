"""
Full test with the working approach
"""
import numpy as np
from scipy.integrate import odeint

def run(lam, bs, bd, bf, tl, tg, N=50):
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
        
        # Boost is active between bs and bs+bd
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
        t = np.linspace(0, 200, 200)
        sol = odeint(dynamics, init, t)
        
        M = sol[-1].reshape((L, k))
        target_M = M[tl, tg]
        others = [M[l,g] for l in range(L) for g in range(k) if (l,g) != (tl, tg)]
        
        if target_M > np.mean(others) + 0.15:
            success += 1
    
    return success / N

print("Testing key parameter combinations...")
print("-" * 50)

# Test around the successful parameters
tests = [
    (0.1, 40, 30, 1.5, 0, 0),
    (0.1, 35, 30, 1.5, 0, 0),
    (0.1, 45, 30, 1.5, 0, 0),
    (0.1, 40, 25, 1.5, 0, 0),
    (0.1, 40, 35, 1.5, 0, 0),
    (0.1, 40, 30, 1.3, 0, 0),
    (0.1, 40, 30, 1.7, 0, 0),
    (0.1, 30, 30, 1.5, 0, 0),
    (0.1, 50, 30, 1.5, 0, 0),
    (0.1, 40, 20, 1.5, 0, 0),
]

for lam, bs, bd, bf, tl, tg in tests:
    s = run(lam, bs, bd, bf, tl, tg, N=50)
    print(f"bs={bs}, bd={bd}, bf={bf}: success={s:.2f}")

# Test different targets
print("\nTesting different targets...")
for tl in [0, 1]:
    for tg in [0, 1, 2]:
        s = run(0.1, 40, 30, 1.5, tl, tg, N=50)
        print(f"target=({tl},{tg}): success={s:.2f}")

print("\nDone!")
