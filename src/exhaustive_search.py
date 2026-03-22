"""
Exhaustive Search
"""
import numpy as np
from scipy.integrate import odeint

def run(lam, boosts, target=(0,0), N=60):
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

print("Exhaustive Search - All targets")
print("=" * 50)

best_overall = 0
best_overall_params = None

# Try all targets with key boost params
for target in [(l, g) for l in range(2) for g in range(3)]:
    print(f"\nTarget: {target}")
    
    # Baseline
    s = run(0.1, [], target=target, N=60)
    print(f"  No boost: {s:.2f}")
    
    # Single boost variations
    best_for_target = s
    best_params = None
    
    for bs in [20, 30, 40, 50]:
        for bd in [30, 40, 50, 60]:
            for bf in [1.5, 2.0, 2.5]:
                for boost_tg in [0, 1, 2]:
                    boosts = [(bs, bd, bf, boost_tg)]
                    s = run(0.1, boosts, target=target, N=60)
                    
                    if s > best_for_target:
                        best_for_target = s
                        best_params = boosts
                        
                        if s > best_overall:
                            best_overall = s
                            best_overall_params = (target, boosts)
    
    if best_params:
        print(f"  Best: {best_params} -> {best_overall:.2f}")
    else:
        print(f"  Best: {best_for_target:.2f}")

print("\n" + "=" * 50)
print(f"Best Overall: {best_overall_params}")
print(f"Success Rate: {best_overall:.2f}")
print("Done!")
