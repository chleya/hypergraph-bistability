"""
Hypergraph Control Gym Environment
===================================
A simple gym-style environment for training RL agents to control multistability.
"""
import numpy as np
from scipy.integrate import odeint
import gym
from gym import spaces

class HypergraphControlEnv(gym.Env):
    """
    Environment for controlling hypergraph multistability.
    
    State: Current M values (6 dims for L=2, k=3)
    Actions: 
        - boost_target: which group to boost (0, 1, 2)
        - boost_start: when to apply boost (discretized)
        - boost_duration: how long to boost (discretized)
        - boost_factor: how strong (discretized)
    """
    
    def __init__(self, lambda_=0.1, Kc_list=None, target=None):
        super().__init__()
        
        self.lambda_ = lambda_
        self.Kc_list = Kc_list if Kc_list else [0.32, 0.40, 0.48]
        self.L, self.k = 2, 3
        self.mu = 0.0
        
        # Target attractor (if None, random)
        self.target = target
        
        # Compute coefficients
        self.a = [-3/kc for kc in self.Kc_list]
        self.b = [4.5/kc for kc in self.Kc_list]
        self.c = [-1.5/kc for kc in self.Kc_list]
        
        # Action space: discrete
        # action[0]: boost_target (0, 1, 2)
        # action[1]: boost_start (0-4 -> 20, 30, 40, 50, 60)
        # action[2]: boost_duration (0-3 -> 15, 25, 35, 45)
        # action[3]: boost_factor (0-3 -> 1.2, 1.5, 1.8, 2.1)
        self.action_space = spaces.MultiDiscrete([3, 5, 4, 4])
        
        # Observation space
        self.observation_space = spaces.Box(
            low=-0.1, high=1.1, 
            shape=(self.L * self.k,), 
            dtype=np.float32
        )
        
        self.state = None
        self.max_steps = 250
        self.current_step = 0
        
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
                self_term = a_eff[i]*M[l,i]**3 + b_eff[i]*M[l,i]**2 + c_eff[i]*M[l,i]
                cross_group = -self.lambda_ * M[l,i] * np.sum(M[l, np.arange(self.k) != i])
                cross_layer = self.mu * np.sum(M[np.arange(self.L) != l, i])
                dM[l,i] = self_term + cross_group + cross_layer
        
        return dM.flatten()
    
    def reset(self):
        """Reset environment to random initial state."""
        self.state = np.random.uniform(0.1, 0.9, self.L * self.k).astype(np.float32)
        
        # Random target if not specified
        if self.target is None:
            self.target = (np.random.randint(0, self.L), np.random.randint(0, self.k))
        
        self.current_step = 0
        self.episode_history = []
        
        return self.state
    
    def step(self, action):
        """Execute action and return (state, reward, done, info)."""
        boost_target, boost_start_idx, boost_duration_idx, boost_factor_idx = action
        
        # Map action to values
        boost_start = 20 + boost_start_idx * 10  # 20, 30, 40, 50, 60
        boost_duration = 15 + boost_duration_idx * 10  # 15, 25, 35, 45
        boost_factor = 1.2 + boost_factor_idx * 0.3  # 1.2, 1.5, 1.8, 2.1
        
        self.current_step += 1
        
        # Run simulation for this step
        # Simplified: just simulate the full trajectory and check result at end
        # In real gym, we'd do step-by-step, but here we just compute end state
        
        # Check if boost is active at this time
        if self.current_step >= boost_start and self.current_step < boost_start + boost_duration:
            boost_active = True
        else:
            boost_active = False
        
        # One step of dynamics (simplified)
        dt = 1.0
        M_flat = self.state
        t_span = [0, dt]
        
        a_eff = self.a.copy()
        b_eff = self.b.copy()
        c_eff = self.c.copy()
        
        if boost_active:
            a_eff[boost_target] /= boost_factor
            b_eff[boost_target] /= boost_factor
            c_eff[boost_target] /= boost_factor
        
        # Simple Euler step
        M = M_flat.reshape((self.L, self.k))
        dM = np.zeros((self.L, self.k))
        for l in range(self.L):
            for i in range(self.k):
                st = a_eff[i]*M[l,i]**3 + b_eff[i]*M[l,i]**2 + c_eff[i]*M[l,i]
                cr = -self.lambda_ * M[l,i] * np.sum(M[l, np.arange(self.k) != i])
                cl = self.mu * np.sum(M[np.arange(self.L) != l, i])
                dM[l,i] = st + cr + cl
        
        new_state = M_flat + dM * dt
        new_state = np.clip(new_state, -0.1, 1.1).astype(np.float32)
        
        self.state = new_state
        
        # Check if done
        done = self.current_step >= self.max_steps
        
        # Compute reward
        M = self.state.reshape((self.L, self.k))
        target_layer, target_group = self.target
        target_M = M[target_layer, target_group]
        others = [M[l,g] for l in range(self.L) for g in range(self.k) 
                 if (l,g) != (target_layer, target_group)]
        avg_other = np.mean(others) if others else 0
        
        # Reward: +1 if target > others + threshold, else small penalty
        if target_M > avg_other + 0.15:
            reward = 10.0  # Success!
        else:
            reward = (target_M - avg_other) * 0.1  # Progress reward
        
        info = {
            'target_M': target_M,
            'avg_other': avg_other,
            'success': target_M > avg_other + 0.15
        }
        
        return self.state, reward, done, info
    
    def get_success_rate(self, policy, n_episodes=100):
        """Evaluate policy success rate."""
        successes = 0
        for _ in range(n_episodes):
            state = self.reset()
            done = False
            while not done:
                action = policy(state)
                state, reward, done, info = self.step(action)
            if info['success']:
                successes += 1
        return successes / n_episodes


# Simple random policy for testing
class RandomPolicy:
    def __init__(self, action_space):
        self.action_space = action_space
    
    def __call__(self, state):
        return self.action_space.sample()


# Test the environment
if __name__ == "__main__":
    env = HypergraphControlEnv(lambda_=0.1)
    
    print("Testing HypergraphControlEnv...")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Test random policy
    policy = RandomPolicy(env.action_space)
    success_rate = env.get_success_rate(policy, n_episodes=50)
    print(f"\nRandom policy success rate: {success_rate:.2f}")
    
    # Test one episode
    state = env.reset()
    print(f"\nInitial state: {state}")
    
    total_reward = 0
    for i in range(10):
        action = policy(state)
        state, reward, done, info = env.step(action)
        total_reward += reward
        if i % 5 == 0:
            print(f"Step {i}: reward={reward:.2f}, target_M={info['target_M']:.2f}")
    
    print(f"\nTotal reward: {total_reward:.2f}")
    print("Environment test OK!")
