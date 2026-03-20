"""
Tests for core dynamics functions
"""
import numpy as np
import sys
sys.path.insert(0, 'F:/hypergraph_bistability/src')


def test_gen_hypergraph():
    from noise_simple import gen_hypergraph
    N, E = 20, 30
    H = gen_hypergraph(N, E)
    assert H.shape == (N, E)


def test_micro_update():
    from noise_simple import gen_hypergraph, micro_update
    N, E = 20, 30
    L, k = 2, 3
    Kc = [0.32, 0.40, 0.48]
    lam = 0.1
    sigma = 0.01

    H = gen_hypergraph(N, E)
    ga = np.random.randint(0, k, N)
    la = np.random.randint(0, L, N)
    m = np.random.uniform(0.1, 0.9, N)

    dm = micro_update(m, H, ga, la, Kc, lam, L, k, sigma)
    assert len(dm) == N


def test_cluster():
    from noise_simple import cluster
    finals = np.random.uniform(0.1, 0.9, (5, 10))
    n = cluster(finals, th=0.1)
    assert n >= 1


def test_dynamics_F():
    from test_quick import F
    L, k = 2, 3
    Kc = [0.32, 0.40, 0.48]
    a = [-3/kc for kc in Kc]
    b = [4.5/kc for kc in Kc]
    c = [-1.5/kc for kc in Kc]
    lam, mu = 0.1, 0.0

    M_flat = np.random.uniform(0.1, 0.9, L*k)
    t = 0
    dM = F(M_flat, t, a, b, c, lam, mu, L, k)
    assert len(dM) == L * k


def test_odeint_integration():
    from scipy.integrate import odeint
    from test_quick import F

    Kc = [0.32, 0.40, 0.48]
    a = [-3/kc for kc in Kc]
    b = [4.5/kc for kc in Kc]
    c = [-1.5/kc for kc in Kc]
    L, k = 2, 3
    lam, mu = 0.1, 0.0

    init = np.random.uniform(0.1, 0.9, L*k)
    t = np.linspace(0, 10, 20)
    sol = odeint(F, init, t, args=(a, b, c, lam, mu, L, k))
    assert sol.shape == (len(t), L*k)
