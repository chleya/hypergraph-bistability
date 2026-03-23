"""
Random Sampling with More Trials
"""
import numpy as np
from scipy.integrate import odeint
import random

def dynamics(M, t, a, b, c, lam, mu, L, k, tg, bf, boost):
    M = M.reshape((L, k))
    dM = np.zeros((L, k))
    
    a_eff = a[:]
    b_eff = b[:]
    c_eff = c[:]
    
    if boost:
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

def run(lam, bs, bd, bf, tl, tg, N=50):
    Kc = [0.32, 0.40, 0.48]
    a = [-3/kc for kc in Kc]
    b = [4.5/kc for kc in Kc]
    c = [-1.5/kc for kc in Kc]
    L, k = 2, 3
    mu = 0.0
    
    success = 0
    for _ in range(N):
        init = np.random.uniform(0.1, 0.9, L*k)
        
        t1 = np.linspace(0, bs, 80)
        sol1 = odeint(dynamics, init, t1, args=(a,b,c,lam,mu,L,k,tg,1.0,False))
        
        t2 = np.linspace(bs, bs+bd, 80)
        sol2 = odeint(dynamics, sol1[-1], t2, args=(a,b,c,lam,mu,L,k,tg,bf,True))
        
        t3 = np.linspace(bs+bd, 200, 80)
        sol3 = odeint(dynamics, sol2[-1], t3, args=(a,b,c,lam,mu,L,k,tg,1.0,False))
        
        M = sol3[-1].reshape((L, k))
        target_M = M[tl, tg]
        others = [M[l,g] for l in range(L) for g in range(k) if (l,g) != (tl, tg)]
        
        if target_M > np.mean(others) + 0.15:
            success += 1
    
    return success / N

# Random sampling
lam = 0.1
n_trials = 200

best_s = 0
best_p = None
results = []

random.seed(42)
np.random.seed(42)

print("Random Sampling for Optimal Control")
print(f"Lambda = {lam}, Trials = {n_trials}")
print("-" * 40)

for i in range(n_trials):
    bs = random.choice([15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70])
    bd = random.choice([10, 15, 20, 25, 30, 35, 40, 45, 50])
    bf = random.choice([1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.2, 2.5])
    tl = random.randint(0, 1)
    tg = random.randint(0, 2)
    
    s = run(lam, bs, bd, bf, tl, tg, N=40)
    results.append({'bs': bs, 'bd': bd, 'bf': bf, 'tl': tl, 'tg': tg, 's': s})
    
    if s > best_s:
        best_s = s
        best_p = (bs, bd, bf, tl, tg)
        print(f"Trial {i+1}: NEW BEST! {best_s:.2f} with {best_p}")
    
    if (i+1) % 50 == 0:
        print(f"Progress: {i+1}/{n_trials}, best: {best_s:.2f}")

print("-" * 40)
print(f"Best: {best_p} -> {best_s:.2f}")

# Fine tune around best
print("\nFine-tuning...")
for _ in range(50):
    bs = best_p[0] + random.choice([-5, 0, 5, 10])
    bd = best_p[1] + random.choice([-5, 0, 5, 10])
    bf = best_p[2] + random.choice([-0.2, 0, 0.2])
    tl, tg = best_p[3], best_p[4]
    
    bs = max(10, min(80, bs))
    bd = max(5, min(60, bd))
    bf = max(1.1, min(2.5, bf))
    
    s = run(lam, bs, bd, bf, tl, tg, N=60)
    
    if s > best_s:
        best_s = s
        best_p = (bs, bd, bf, tl, tg)
        print(f"Fine: NEW BEST! {best_s:.2f} with {best_p}")

print(f"\nFinal Best: {best_p} -> {best_s:.2f}")
print("Done!")
