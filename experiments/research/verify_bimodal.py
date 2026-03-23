"""
Verify Bimodal Discovery
"""
import numpy as np
from scipy.integrate import odeint

def run_bimodal(lam, boosts, target=(0,0), N=100):
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
        # Bimodal: half low, half high
        init = np.concatenate([np.random.uniform(0.1, 0.3, L*k//2), np.random.uniform(0.7, 0.9, L*k - L*k//2)])
        np.random.shuffle(init)
        
        t = np.linspace(0, 300, 300)
        sol = odeint(dynamics, init, t)
        
        M = sol[-1].reshape((L, k))
        target_M = M[tl, tg]
        others = [M[l,g] for l in range(L) for g in range(k) if (l,g) != (tl, tg)]
        
        if target_M > np.mean(others) + 0.15:
            success += 1
    
    return success / N

print("=" * 60)
print("VERIFICATION: Bimodal Initial Condition")
print("=" * 60)

# Test all targets
print("\nTest all targets with bimodal IC (no boost):")
for target in [(l, g) for l in range(2) for g in range(3)]:
    s = run_bimodal(0.1, [], target=target, N=100)
    print(f"  target {target}: {s:.2f}")

# Test with boost
print("\nTest with boost:")
boosts = [(30, 60, 2.5, 0)]
for target in [(l, g) for l in range(2) for g in range(3)]:
    s = run_bimodal(0.1, boosts, target=target, N=100)
    print(f"  target {target}: {s:.2f}")

# Different bimodal splits
print("\nDifferent bimodal splits (target=(0,0)):")
for split in [0.3, 0.4, 0.5, 0.6, 0.7]:
    def run_split(lam, boosts, target, N=100):
        success = 0
        for _ in range(N):
            n_low = int(L*k * split)
            n_high = L*k - n_low
            init = np.concatenate([np.random.uniform(0.1, 0.3, n_low), np.random.uniform(0.7, 0.9, n_high)])
            np.random.shuffle(init)
            
            t = np.linspace(0, 300, 300)
            sol = odeint(lambda M, t: dynamics(M, t, lam, boosts), init, t)
            
            M = sol[-1].reshape((L, k))
            target_M = M[target[0], target[1]]
            others = [M[l,g] for l in range(L) for g in range(k) if (l,g) != target]
            
            if target_M > np.mean(others) + 0.15:
                success += 1
        return success / N
    
    def dynamics(M, t, lam, boosts):
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
    
    s = run_split(0.1, [], (0,0), N=100)
    print(f"  split={split}: {s:.2f}")

print("\n" + "=" * 60)
print("Done!")
