"""
Group-Mean Driven 微观 - 简化快速版
===================================
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import os

os.makedirs('F:/hypergraph_bistability/figures/microscopic/', exist_ok=True)
np.random.seed(42)

def gen_hypergraph(N, E):
    H = np.zeros((N, E))
    for e in range(E):
        s = np.random.randint(2, 5)
        nodes = np.random.choice(N, size=s, replace=False)
        H[nodes, e] = 1
    return H

def micro_dynamics(t, m, H, ga, la, Kc, lam, L, k):
    N = len(m)
    M = np.zeros((L, k))
    cnt = np.zeros((L, k))
    for v in range(N):
        i, l = ga[v], la[v]
        M[l, i] += m[v]
        cnt[l, i] += 1
    M = M / (cnt + 1e-8)
    
    a = [-3.0/x for x in Kc]
    b = [4.5/x for x in Kc]
    c = [-1.5/x for x in Kc]
    
    dM = np.zeros((L, k))
    for l in range(L):
        for i in range(k):
            self_t = a[i]*M[l,i]**3 + b[i]*M[l,i]**2 + c[i]*M[l,i]
            cross = -lam * M[l,i] * np.sum(M[l, :])
            dM[l,i] = self_t + cross
    
    dm = np.zeros(N)
    for v in range(N):
        i, l = ga[v], la[v]
        dm[v] = dM[l, i]
        edges = np.where(H[v] > 0)[0]
        if len(edges) > 0:
            lm = np.mean([np.mean(m[np.where(H[:,e]>0)[0]]) for e in edges])
            dm[v] += 0.02 * (lm - m[v])
    return dm

def mf_dynamics(t, Mf, Kc, lam, L, k):
    M = Mf.reshape(L, k)
    dM = np.zeros((L, k))
    a = [-3.0/x for x in Kc]
    b = [4.5/x for x in Kc]
    c = [-1.5/x for x in Kc]
    for l in range(L):
        for i in range(k):
            st = a[i]*M[l,i]**3 + b[i]*M[l,i]**2 + c[i]*M[l,i]
            cr = -lam * M[l,i] * np.sum(M[l, :])
            dM[l,i] = st + cr
    return dM.flatten()

def cluster(finals, th=0.08):
    uniq = []
    for s in finals:
        is_new = True
        for u in uniq:
            if np.sqrt(np.mean((s-u)**2)) < th:
                is_new = False
                break
        if is_new:
            uniq.append(s)
    return len(uniq)

# 实验
print("=" * 50)
print("Group-Mean Driven vs Mean-Field")
print("=" * 50)

L, k = 2, 3
Kc = [0.32, 0.40, 0.48]
N, E = 50, 80
ga = np.random.randint(0, k, N)
la = np.random.randint(0, L, N)
H = gen_hypergraph(N, E)
print(f"超图: N={N}, E={E}")

# 微观
print("\n微观:")
micro = []
for lam in [0.0, 0.3, 0.6, 0.9]:
    finals = []
    for _ in range(30):
        init = np.random.uniform(0.1, 0.9, N)
        sol = solve_ivp(micro_dynamics, [0, 50], init,
                       args=(H, ga, la, Kc, lam, L, k), method='RK45',
                       atol=1e-5, rtol=1e-5)
        finals.append(sol.y[:, -1])
    n = cluster(np.array(finals))
    micro.append((lam, n))
    print(f"  λ={lam}: {n}")

# MF
print("\nMF:")
mf = []
for lam in [0.0, 0.3, 0.6, 0.9]:
    finals = []
    for _ in range(30):
        init = np.random.uniform(0.1, 0.9, L*k)
        sol = solve_ivp(mf_dynamics, [0, 50], init,
                       args=(Kc, lam, L, k), method='RK45',
                       atol=1e-5, rtol=1e-5)
        finals.append(sol.y[:, -1])
    n = cluster(np.array(finals))
    mf.append((lam, n))
    print(f"  λ={lam}: {n}")

# 绘图
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot([x[0] for x in micro], [x[1] for x in micro], 'o-', 
        label='Microscopic', lw=2, ms=8, color='steelblue')
ax.plot([x[0] for x in mf], [x[1] for x in mf], 's--', 
        label='Mean-Field', lw=2, ms=8, color='coral')
ax.set_xlabel('λ')
ax.set_ylabel('Attractors')
ax.set_title('Group-Mean Driven Microscopic vs MF')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('F:/hypergraph_bistability/figures/microscopic/v3.png', dpi=150)
plt.close()
print("\nSaved v3.png")
print("完成!")
