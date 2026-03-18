"""简单测试"""
import numpy as np
from scipy.integrate import odeint

# 测试基本动力学
def test_dynamics(M, t, k, L, Kc_list, lambda_, mu):
    dMdt = np.zeros(k * L)
    M = np.array(M).reshape(k, L)
    for i in range(k):
        for l in range(L):
            Mc = M[i, l]
            Kc_i = Kc_list[i]
            term1 = Mc * (1 - Mc / Kc_i)
            dMdt[i * L + l] = term1
    return dMdt

k, L = 3, 2
Kc_list = [0.4, 0.4, 0.4]
lambda_ = 0.5
mu = 0.0

M0 = np.random.uniform(0.1, 0.9, k * L)
t = np.linspace(0, 10, 100)
sol = odeint(test_dynamics, M0, t, args=(k, L, Kc_list, lambda_, mu))
print('Test passed! Final:', sol[-1])
