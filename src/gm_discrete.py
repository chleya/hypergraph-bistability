"""
Group-Mean Driven 微观 - 极简离散版
===================================
使用离散迭代而非ODE，更快
"""
import numpy as np
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

def micro_update(m, H, ga, la, Kc, lam, L, k):
    """Group-Mean Driven 微观更新"""
    N = len(m)
    
    # Step 1: 计算群体均值 M
    M = np.zeros((L, k))
    cnt = np.zeros((L, k))
    for v in range(N):
        i, l = ga[v], la[v]
        M[l, i] += m[v]
        cnt[l, i] += 1
    M = M / (cnt + 1e-8)
    
    # Step 2: 计算 dM
    a = [-3.0/x for x in Kc]
    b = [4.5/x for x in Kc]
    c = [-1.5/x for x in Kc]
    
    dM = np.zeros((L, k))
    for l in range(L):
        for i in range(k):
            Mi = M[l, i]
            self_term = a[i]*Mi**3 + b[i]*Mi**2 + c[i]*Mi
            cross = -lam * Mi * (np.sum(M[l, :]) - Mi)
            dM[l, i] = self_term + cross
    
    # Step 3: 每个节点跟随群体dM + 超边微调
    dm = np.zeros(N)
    for v in range(N):
        i, l = ga[v], la[v]
        dm[v] = dM[l, i]  # 核心：跟随群体
        
        # 超边局部微调
        edges = np.where(H[v] > 0)[0]
        if len(edges) > 0:
            local = np.mean([np.mean(m[np.where(H[:,e]>0)[0]]) for e in edges])
            dm[v] += 0.02 * (local - m[v])
    
    return dm

def mf_update(M, Kc, lam, L, k):
    """Mean-Field 更新"""
    M_new = np.zeros((L, k))
    a = [-3.0/x for x in Kc]
    b = [4.5/x for x in Kc]
    c = [-1.5/x for x in Kc]
    
    for l in range(L):
        for i in range(k):
            Mi = M[l, i]
            st = a[i]*Mi**3 + b[i]*Mi**2 + c[i]*Mi
            cr = -lam * Mi * (np.sum(M[l, :]) - Mi)
            M_new[l, i] = np.clip(Mi + 0.05 * (st + cr), 0, 1)
    return M_new

def cluster(finals, th=0.1):
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
print("Group-Mean Driven 微观 vs Mean-Field")
print("=" * 50)

L, k = 2, 3
Kc = [0.32, 0.40, 0.48]
N, E = 50, 80
ga = np.random.randint(0, k, N)
la = np.random.randint(0, L, N)
H = gen_hypergraph(N, E)
print(f"N={N}, E={E}")

# 微观
print("\n微观:")
micro = []
for lam in [0.0, 0.2, 0.4, 0.6, 0.8]:
    finals = []
    for _ in range(50):
        m = np.random.uniform(0.1, 0.9, N)
        for _ in range(100):
            dm = micro_update(m, H, ga, la, Kc, lam, L, k)
            m = np.clip(m + 0.05 * dm, 0.01, 0.99)
        finals.append(m.copy())
    n = cluster(np.array(finals))
    micro.append((lam, n))
    print(f"  λ={lam}: {n}个吸引子")

# MF
print("\nMean-Field:")
mf = []
for lam in [0.0, 0.2, 0.4, 0.6, 0.8]:
    finals = []
    for _ in range(50):
        M = np.random.uniform(0.1, 0.9, (L, k))
        for _ in range(100):
            M = mf_update(M, Kc, lam, L, k)
        finals.append(M.flatten())
    n = cluster(np.array(finals))
    mf.append((lam, n))
    print(f"  λ={lam}: {n}个吸引子")

# 绘图
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot([x[0] for x in micro], [x[1] for x in micro], 'o-', 
        label='Microscopic (Group-Mean)', lw=2, ms=8, color='steelblue')
ax.plot([x[0] for x in mf], [x[1] for x in mf], 's--', 
        label='Mean-Field', lw=2, ms=8, color='coral')
ax.set_xlabel('λ (Coupling)', fontsize=12)
ax.set_ylabel('Number of Attractors', fontsize=12)
ax.set_title('Group-Mean Driven Microscopic vs MF', fontsize=14)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
ax.set_ylim(0, 55)
plt.tight_layout()
plt.savefig('F:/hypergraph_bistability/figures/microscopic/gm_discrete.png', dpi=150)
plt.close()
print("\nSaved: gm_discrete.png")

# 验证共享basin
print("\n验证共享basin (λ=0):")
finals = []
for _ in range(80):
    m = np.random.uniform(0.1, 0.9, N)
    for _ in range(100):
        dm = micro_update(m, H, ga, la, Kc, 0.0, L, k)
        m = np.clip(m + 0.05 * dm, 0.01, 0.99)
    finals.append(m.copy())

# 群体均值
gmeans = np.zeros((80, L*k))
for idx, mm in enumerate(finals):
    for v in range(N):
        i, l = ga[v], la[v]
        gmeans[idx, l*k + i] = mm[v]

n_basin = cluster(gmeans, th=0.15)
print(f"群体均值basin: {n_basin}个 (如果独立会有~80个)")
print("\n完成!")
