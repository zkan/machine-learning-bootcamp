import numpy as np

from bandit import run_epsilon_greedy


def test_bandit_repeats_with_the_same_seed():
    first, rewards = run_epsilon_greedy(0.1, 20, seed=42)
    second, again = run_epsilon_greedy(0.1, 20, seed=42)
    other, _ = run_epsilon_greedy(0.1, 20, seed=7)
    assert np.array_equal(first, second)
    assert np.array_equal(rewards, again)
    assert len(first) == 20
    assert round(float(rewards.mean()), 2) == 0.05
    assert rewards.min() >= 0 and rewards.max() <= 1
    assert not np.array_equal(first, other)
