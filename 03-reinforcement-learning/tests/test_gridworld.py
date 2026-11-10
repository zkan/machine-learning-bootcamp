import numpy as np

from gridworld import q_learning, greedy_policy, reaches_goal


def test_q_learning_repeats_with_the_same_seed():
    first, rewards = q_learning(seed=42)
    second, again = q_learning(seed=42)
    other, _ = q_learning(seed=7)
    assert np.allclose(first, second)
    assert np.allclose(rewards, again)
    assert not np.allclose(first, other)


def test_greedy_policy_reaches_the_goal_and_avoids_the_pit():
    q, rewards = q_learning(gamma=0.9, seed=42)
    assert reaches_goal(greedy_policy(q))
    assert rewards[-50:].mean() > rewards[:50].mean()
    short_sighted, _ = q_learning(gamma=0.0, seed=42)
    assert not reaches_goal(greedy_policy(short_sighted))
