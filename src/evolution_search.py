"""
Random Search with Evolution
============================
Simple evolutionary approach to find better control parameters.
"""
import numpy as np
from scipy.integrate import odeint
import random

def dynamics(M_flat, t, a, b, c, lam, mu, L, k, boost_target, boost_factor, boost_active):
    M = M_flat.reshape((L, k))
    dM = np.zeros((L, k))
    
    a_eff = a.copy()
    b_eff = b.copy()
    c_eff = c.copy()
    
    if boost_active:
        a_eff[boost_target] /= boost_factor
        b_eff[boost_target] /= boost_factor
        c_eff[boost_target] /= boost_factor
    
    for l in range(L):
        for i in range(k):
            st = a_eff[i]*M[l,i]**3 + b_eff[i]*M[l,i]**2 + c_eff[i]*M[l,i]
            cr = -lam * M[l,i] * np.sum(M[l, np.arange(k) != i])
            cl = mu * np.sum(M[np.arange(L) != l, i])
            dM[l,i] = st + cr + cl
    
    return dM.flatten()

def evaluate(lam, bs, bd, bf, tl, tg, N=50):
    Kc = [0.32, 0.40, 0.48]
    a = [-3/kc for kc in Kc]
    b = [4.5/kc for kc in Kc]
    c = [-1.5/kc for kc in Kc]
    L, k = 2, 3
    mu = 0.0
    
    success = 0
    for _ in range(N):
        init = np.random.uniform(0.1, 0.9, L*k)
        
        # Phase 1
        t1 = np.linspace(0, bs, 100)
        sol1 = odeint(dynamics, init, t1, args=(a,b,c,lam,mu,L,k,tg,1.0,False))
        
        # Phase 2 (boost)
        t2 = np.linspace(bs, bs+bd, 100)
        sol2 = odeint(dynamics, sol1[-1], t2, args=(a,b,c,lam,mu,L,k,tg,bf,True))
        
        # Phase 3
        t3 = np.linspace(bs+bd, 250, 100)
        sol3 = odeint(dynamics, sol2[-1], t3, args=(a,b,c,lam,mu,L,k,tg,1.0,False))
        
        M = sol3[-1].reshape((L, k))
        target_M = M[tl, tg]
        others = [M[l,g] for l in range(L) for g in range(k) if (l,g) != (tl, tg)]
        
        if target_M > np.mean(others) + 0.15:
            success += 1
    
    return success / N

# Evolution
print("=" * 60)
print("Evolutionary Search for Optimal Control")
print("=" * 60)

lam = 0.1
pop_size = 30
n_generations = 10
best_ever = 0
best_params_ever = None

# Initial population
population = []
for _ in range(pop_size):
    individual = {
        'bs': random.choice([20, 30, 40, 50, 60]),
        'bd': random.choice([15, 25, 35, 45]),
        'bf': random.choice([1.2, 1.4, 1.6, 1.8, 2.0]),
        'tl': random.randint(0, 1),
        'tg': random.randint(0, 2)
    }
    population.append(individual)

for gen in range(n_generations):
    # Evaluate
    scores = []
    for ind in population:
        s = evaluate(lam, ind['bs'], ind['bd'], ind['bf'], ind['tl'], ind['tg'])
        scores.append(s)
        
        if s > best_ever:
            best_ever = s
            best_params_ever = ind.copy()
    
    # Print best of generation
    gen_best = max(scores)
    print(f"Gen {gen+1}: best={gen_best:.2f}, ever={best_ever:.2f}")
    
    # Selection (keep top 50%)
    indexed_scores = [(i, s) for i, s in enumerate(scores)]
    sorted_idx = sorted(indexed_scores, key=lambda x: x[1], reverse=True)
    survivors = [population[i] for i, _ in sorted_idx[:pop_size//2]]
    
    # Reproduction
    new_pop = survivors.copy()
    while len(new_pop) < pop_size:
        parent = random.choice(survivors)
        child = {
            'bs': parent['bs'] + random.choice([-10, 0, 10]) if random.random() < 0.3 else parent['bs'],
            'bd': parent['bd'] + random.choice([-10, 0, 10]) if random.random() < 0.3 else parent['bd'],
            'bf': parent['bf'] + random.choice([-0.2, 0, 0.2]) if random.random() < 0.3 else parent['bf'],
            'tl': parent['tl'],
            'tg': parent['tg']
        }
        # Clamp
        child['bs'] = max(10, min(80, child['bs']))
        child['bd'] = max(5, min(60, child['bd']))
        child['bf'] = max(1.1, min(2.5, child['bf']))
        new_pop.append(child)
    
    population = new_pop

print("\n" + "=" * 60)
print("Best Ever Found:")
print(best_params_ever)
print(f"Success rate: {best_ever:.2f}")
print("=" * 60)

# Fine-tune around best
print("\nFine-tuning around best...")
fine_best = best_ever
fine_params = best_params_ever

for _ in range(20):
    for bs_delta in [-5, 0, 5]:
        for bd_delta in [-5, 0, 5]:
            for bf_delta in [-0.1, 0, 0.1]:
                ind = {
                    'bs': max(10, min(80, fine_params['bs'] + bs_delta)),
                    'bd': max(5, min(60, fine_params['bd'] + bd_delta)),
                    'bf': max(1.1, min(2.5, fine_params['bf'] + bf_delta)),
                    'tl': fine_params['tl'],
                    'tg': fine_params['tg']
                }
                s = evaluate(lam, ind['bs'], ind['bd'], ind['bf'], ind['tl'], ind['tg'], N=60)
                if s > fine_best:
                    fine_best = s
                    fine_params = ind.copy()

print(f"\nAfter fine-tuning: {fine_best:.2f}")
print(f"Params: {fine_params}")

print("\nDone!")
