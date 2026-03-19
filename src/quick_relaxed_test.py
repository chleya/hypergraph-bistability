"""
Quick test of relaxed problem formulation
"""
import numpy as np
from scipy.integrate import odeint

L = 2
k = 3
lambda_ = 0.1
mu = 0.0
Kc = [0.32, 0.40, 0.48]

a = [-3/kc for kc in Kc]
b = [4.5/kc for kc in Kc]
c = [-1.5/kc for kc in Kc]

def F(M, t, lambda_, mu):
    M = M.reshape((L, k))
    dM = np.zeros((L, k))
    for l in range(L):
        for i in range(k):
            st = a[i]*M[l,i]**3 + b[i]*M[l,i]**2 + c[i]*M[l,i]
            cr = -lambda_ * M[l,i] * np.sum(M[l, np.arange(k) != i])
            cl = mu * np.sum(M[np.arange(L) != l, i])
            dM[l,i] = st + cr + cl
    return dM.flatten()

# Relaxed formulation: any dominant state = success
def test_relaxed(N=200, T=500, margin=0.08):
    success = 0
    for _ in range(N):
        # Initialize in unstable region (Chen recommendation)
        M = np.random.uniform(0.3, 0.7, L*k)
        
        # Random target (any dominant pattern)
        target_pattern = np.random.choice([0, 1], size=L*k, p=[0.5, 0.5])
        
        # Simple boost: apply at t=30
        boost_group = np.random.randint(0, k)
        boost_factor = 2.0
        
        t = np.linspace(0, T, T)
        
        # Apply boost at t=30
        def dynamics_boost(M, t):
            M = M.reshape((L, k))
            dM = np.zeros((L, k))
            
            # Apply boost at t=30
            if 25 <= t < 55:
                boost_mask = np.zeros((L, k))
                boost_mask[:, boost_group] = 1
                M_boost = M + boost_factor * boost_mask * 0.05
                M_boost = np.clip(M_boost, 0.01, 0.99)
            else:
                M_boost = M
            
            for l in range(L):
                for i in range(k):
                    st = a[i]*M_boost[l,i]**3 + b[i]*M_boost[l,i]**2 + c[i]*M_boost[l,i]
                    cr = -lambda_ * M_boost[l,i] * np.sum(M_boost[l, np.arange(k) != i])
                    cl = mu * np.sum(M_boost[np.arange(L) != l, i])
                    dM[l,i] = st + cr + cl
            return dM.flatten()
        
        sol = odeint(dynamics_boost, M, t)
        M_final = sol[-1].reshape((L, k))
        
        # Success = any group dominates (not specific target!)
        max_val = np.max(M_final)
        dominant_groups = np.sum(M_final > 0.5)
        
        if dominant_groups >= 1:
            success += 1
    
    return success / N

print("Relaxed Problem Formulation Test")
print("=" * 50)
print(f"Success rate: {test_relaxed(N=200):.1%}")
print("=" * 50)
print("\nThis is the 'any dominant state' formulation")
print("that Chen recommended - much easier than exact target!")
