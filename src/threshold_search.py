"""
Try different thresholds and simulation times
"""
import numpy as np
from scipy.integrate import odeint

def run(lam, boosts, target=(0,0), N=60, threshold=0.15, sim_time=300):
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
        t = np.linspace(0, sim_time, sim_time)
        sol = odeint(dynamics, init, t)
        
        M = sol[-1].reshape((L, k))
        target_M = M[tl, tg]
        others = [M[l,g] for l in range(L) for g in range(k) if (l,g) != (tl, tg)]
        
        if target_M > np.mean(others) + threshold:
            success += 1
    
    return success / N

print("Different Thresholds and Simulation Times")
print("=" * 50)

# Best from before: target (1,1), boost (30, 60, 2.5, 0)
target = (1, 1)
boosts = [(30, 60, 2.5, 0)]

# Different thresholds
print("\n1. Different thresholds:")
for threshold in [0.10, 0.12, 0.15, 0.18, 0.20, 0.25]:
    s = run(0.1, boosts, target=target, N=60, threshold=threshold, sim_time=300)
    print(f"   threshold={threshold}: {s:.2f}")

# Different simulation times
print("\n2. Different sim times:")
for sim_time in [200, 300, 400, 500]:
    s = run(0.1, boosts, target=target, N=60, threshold=0.15, sim_time=sim_time)
    print(f"   sim_time={sim_time}: {s:.2f}")

# Different initial conditions
print("\n3. Different initial conditions:")

def run_ic(lam, boosts, target, ic_type, N=60):
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
        if ic_type == "uniform":
            init = np.random.uniform(0.1, 0.9, L*k)
        elif ic_type == "low":
            init = np.random.uniform(0.0, 0.3, L*k)
        elif ic_type == "high":
            init = np.random.uniform(0.7, 1.0, L*k)
        elif ic_type == "bimodal":
            init = np.concatenate([np.random.uniform(0.1, 0.3, L*k//2), np.random.uniform(0.7, 0.9, L*k - L*k//2)])
        
        t = np.linspace(0, 300, 300)
        sol = odeint(dynamics, init, t)
        
        M = sol[-1].reshape((L, k))
        target_M = M[tl, tg]
        others = [M[l,g] for l in range(L) for g in range(k) if (l,g) != (tl, tg)]
        
        if target_M > np.mean(others) + 0.15:
            success += 1
    
    return success / N

for ic_type in ["uniform", "low", "high", "bimodal"]:
    s = run_ic(0.1, boosts, target, ic_type, N=60)
    print(f"   {ic_type}: {s:.2f}")

print("\n" + "=" * 50)
print("Done!")
