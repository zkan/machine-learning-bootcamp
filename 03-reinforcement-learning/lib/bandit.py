"""Epsilon-greedy bandit. Four arms, Bernoulli rewards, one seed."""

import numpy as np

# Classroom odds for the notebook. The paper game hides these.
TRUE_RATES = np.array([0.15, 0.35, 0.55, 0.75])
N_ARMS = 4


def run_epsilon_greedy(epsilon: float, n_pulls: int, seed: int, n_arms: int = N_ARMS):
    if not 0 <= epsilon <= 1:
        raise ValueError("epsilon must be between 0 and 1")
    if n_pulls < 1:
        raise ValueError("n_pulls must be at least 1")
    rng = np.random.default_rng(seed)
    counts = np.zeros(n_arms)
    values = np.zeros(n_arms)
    choices = np.empty(n_pulls, dtype=int)
    rewards = np.empty(n_pulls, dtype=float)
    for step in range(n_pulls):
        if rng.random() < epsilon:
            arm = int(rng.integers(0, n_arms))
        else:
            # Ties take the lowest arm index. That keeps the seed replay stable.
            arm = int(np.argmax(values))
        reward = float(rng.random() < TRUE_RATES[arm])
        counts[arm] += 1
        values[arm] += (reward - values[arm]) / counts[arm]
        choices[step] = arm
        rewards[step] = reward
    return choices, rewards
