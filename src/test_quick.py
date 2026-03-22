"""
Quick Test - Optimal Control
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

Kc = [0.32, 0.40, 0.48]
a = [-3/kc for kc in Kc]
b = [4.5/kc for kc in Kc]
c = [-1.5/kc for kc in Kc]
L, k = 2, 3
lam = 0.1
mu = 0.0

# Test one run
init = np.random.uniform(0.1, 0.9, L*k)
t = np.linspace(0, 100, 200)
sol = odeint(F, init, t, args=(a, b, c, lam, mu, L, k))

M = sol[-1].reshape((L, k))
print("Final state:")
print(M)
print("Test OK!")
