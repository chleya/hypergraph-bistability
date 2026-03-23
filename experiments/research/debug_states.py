"""调试版"""
import numpy as np
from scipy.integrate import odeint

def dynamics(M, t, k, L, Kc_list, lam, mu):
    dMdt = np.zeros(k * L)
    M_arr = np.array(M).reshape(k, L)
    
    for i in range(k):
        for l in range(L):
            Mc = M_arr[i, l]
            Kc_i = Kc_list[i]
            term1 = Mc * (1 - Mc / Kc_i)
            
            # 竞争
            competition = 0
            for j in range(k):
                if j != i:
                    Mj_mean = np.mean(M_arr[j, :])
                    competition += lam * (Mj_mean - Mc)
            
            # 层间
            layer_coup = 0
            for l2 in range(L):
                if l2 != l:
                    layer_coup += mu * (M_arr[i, l2] - Mc)
            
            dMdt[i * L + l] = term1 + competition + layer_coup
    
    return dMdt


k, L = 3, 2
Kc_list = [0.32, 0.40, 0.48]
lam, mu = 0.5, 0.0
n_init = 10
t = np.linspace(0, 30, 300)

print("Testing different initial conditions:")
for i in range(n_init):
    M0 = np.random.uniform(0.1, 0.9, k * L)
    sol = odeint(dynamics, M0, t, args=(k, L, Kc_list, lam, mu))
    final = sol[-1]
    print(f"  Init {i}: M = {final}")
