"""
Simple Q-Learning for Hypergraph Control
========================================
A simple Q-learning approach to learn optimal boost policy.
"""
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import os

os.makedirs('F:/hypergraph_bistability/figures/control', exist_ok=True)

class HypergraphControl:
    """Simple hypergraph control problem."""
    
    def __init__(self, lambda_=0.1, Kc_list=None):
        self.lambda_ = lambda_
        self.Kc_list = Kc_list if Kc_list else [0.32, 0.40, 0.48]
        self.L, self.k = 2, 3
        self.mu = 0.0
        
        # Coefficients
        self.a = [-3/kc for kc in self.Kc_list]
        self.b = [4.5/kc for kc in self.Kc_list]
        self.c = [-1.5/kc for kc in self.Kc_list]
        
        # Discretize actions
        # action = (target, start, duration, factor)
        self.targets = [(l, g) for l in range(self.L) for g in range(self.k)]
        self.starts = [20, 30, 40, 50, 60]
        self.durations = [15, 25, 35, 45]
        self.factors = [1.2, 1.5, 1.8, 2.1]
        
        self.n_actions = len(self.targets) * len(self.starts) * len(self.durations) * len(self.factors)
        
    def _action_to_params(self, action_idx):
        """Convert action index to parameters."""
        t_idx = action_idx // (len(self.starts) * len(self.durations) * len(self.factors))
        rest = action_idx % (len(self.starts) * len(self.durations) * len(self.factors))
        s_idx = rest // (len(self.durations) * len(self.factors))
        rest = rest % (len(self.durations) * len(self.factors))
        d_idx = rest // len(self.factors)
        f_idx = rest % len(self.factors)
        
        return self.targets[t_idx], self.starts[s_idx], self.durations[d_idx], self.factors[f_idx]
    
    def _dynamics(self, M_flat, t, boost_active=False, boost_target=0, boost_factor=1.0):
        M = M_flat.reshape((self.L, self.k))
        dM = np.zeros((self.L, self.k))
        
        a_eff = self.a.copy()
        b_eff = self.b.copy()
        c_eff = self.c.copy()
        
        if boost_active:
            a_eff[boost_target] /= boost_factor
            b_eff[boost_target] /= boost_factor
            c_eff[boost_target] /= boost_factor
        
        for l in range(self.L):
            for i in range(self.k):
                st = a_eff[i]*M[l,i]**3 + b_eff[i]*M[l,i]**2 + c_eff[i]*M[l,i]
                cr = -self.lambda_ * M[l,i] * np.sum(M[l, np.arange(self.k) != i])
                cl = self.mu * np.sum(M[np.arange(self.L) != l, i])
                dM[l,i] = st + cr + cl
        
        return dM.flatten()
    
    def run_episode(self, action_idx, target=None, N_init=50):
        """Run episode with given action and return success rate."""
        target_g, start, duration, factor = self._action_to_params(action_idx)
        
        # If no target specified, use random
        if target is None:
            target = self.targets[np.random.randint(len(self.targets))]
        
        target_layer, target_group = target
        
        success = 0
        for _ in range(N_init):
            # Random initial state
            init = np.random.uniform(0.1, 0.9, self.L * self.k)
            
            # Phase 1: free evolution
            t1 = np.linspace(0, start, 100)
            sol1 = odeint(self._dynamics, init, t1, args=(False, 0, 1.0))
            
            # Phase 2: boost
            a_b = self.a.copy()
            b_b = self.b.copy()
            c_b = self.c.copy()
            a_b[target_group] /= factor
            b_b[target_group] /= factor
            c_b[target_group] /= factor
            
            t2 = np.linspace(start, start + duration, 100)
            sol2 = odeint(self._dynamics, sol1[-1], t2, 
                         args=(True, target_group, factor), 
                         args=({'a': a_b, 'b': b_b, 'c': c_b},))
            
            # This won't work - need to fix
            # Let me rewrite
            sol2 = sol1[-1]  # Placeholder
        
        return 0.0  # Placeholder


def simple_q_learning(lambda_=0.1, n_episodes=100, epsilon=0.1):
    """Simple Q-learning implementation."""
    
    # Problem setup
    Kc_list = [0.32, 0.40, 0.48]
    L, k = 2, 3
    a = [-3/kc for kc in Kc_list]
    b = [4.5/kc for kc in Kc_list]
    c = [-1.5/kc for kc in Kc_list]
    mu = 0.0
    
    targets = [(l, g) for l in range(L) for g in range(k)]
    starts = [20, 30, 40, 50, 60]
    durations = [15, 25, 35, 45]
    factors = [1.2, 1.5, 1.8, 2.1]
    
    n_actions = len(targets) * len(starts) * len(durations) * len(factors)
    
    # Q-table
    Q = np.zeros(n_actions)
    
    def dynamics(M_flat, t, bs, bd, bf, tg, lam, a, b, c, mu, L, k, boost):
        M = M_flat.reshape((L, k))
        dM = np.zeros((L, k))
        
        a_eff = a.copy()
        b_eff = b.copy()
        c_eff = c.copy()
        
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
    
    def run_action(action_idx, target, N=40):
        """Run single action and return success rate."""
        # Decode action
        t_idx = action_idx // (len(starts) * len(durations) * len(factors))
        rest = action_idx % (len(starts) * len(durations) * len(factors))
        s_idx = rest // (len(durations) * len(factors))
        rest = rest % (len(durations) * len(factors))
        d_idx = rest // len(factors)
        f_idx = rest % len(factors)
        
        bs = starts[s_idx]
        bd = durations[d_idx]
        bf = factors[f_idx]
        tg = targets[t_idx]
        tl, tg_group = tg
        
        success = 0
        for _ in range(N):
            init = np.random.uniform(0.1, 0.9, L*k)
            
            # Phase 1
            t1 = np.linspace(0, bs, 100)
            sol1 = odeint(dynamics, init, t1, 
                         args=(bs, bd, bf, tg_group, lambda_, a, b, c, mu, L, k, False))
            
            # Phase 2 (boost)
            a_b = a.copy()
            b_b = b.copy()
            c_b = c.copy()
            a_b[tg_group] /= bf
            b_b[tg_group] /= bf
            c_b[tg_group] /= bf
            
            t2 = np.linspace(bs, bs+bd, 100)
            sol2 = odeint(dynamics, sol1[-1], t2,
                         args=(bs, bd, bf, tg_group, lambda_, a_b, b_b, c_b, mu, L, k, True))
            
            # Phase 3
            t3 = np.linspace(bs+bd, 250, 100)
            sol3 = odeint(dynamics, sol2[-1], t3,
                         args=(bs, bd, bf, tg_group, lambda_, a, b, c, mu, L, k, False))
            
            M = sol3[-1].reshape((L, k))
            target_M = M[tl, tg_group]
            others = [M[l,g] for l in range(L) for g in range(k) if (l,g) != (tl, tg_group)]
            
            if target_M > np.mean(others) + 0.15:
                success += 1
        
        return success / N
    
    # Q-learning
    print(f"Q-learning: {n_episodes} episodes, {n_actions} actions")
    
    best_success = 0
    best_action = 0
    history = []
    
    for ep in range(n_episodes):
        # Epsilon-greedy
        if np.random.random() < epsilon:
            action = np.random.randint(n_actions)
        else:
            action = np.argmax(Q)
        
        # Evaluate action
        target = targets[np.random.randint(len(targets))]
        success = run_action(action, target)
        
        # Update Q
        Q[action] = Q[action] * 0.9 + success * 0.1
        
        if success > best_success:
            best_success = success
            best_action = action
        
        history.append(success)
        
        if (ep + 1) % 20 == 0:
            print(f"Episode {ep+1}: success={success:.2f}, best={best_success:.2f}")
    
    return best_action, best_success, history, Q


# Run
print("=" * 60)
print("Simple Q-Learning for Hypergraph Control")
print("=" * 60)

best_action, best_success, history, Q = simple_q_learning(lambda_=0.1, n_episodes=50, epsilon=0.2)

print(f"\nBest action: {best_action}")
print(f"Best success: {best_success:.2f}")

# Analyze Q-table
print("\nTop 10 actions by Q-value:")
Q_sorted = np.argsort(Q)[::-1]
for i in range(min(10, len(Q_sorted))):
    idx = Q_sorted[i]
    print(f"  Action {idx}: Q={Q[idx]:.3f}")

print("\nDone!")
