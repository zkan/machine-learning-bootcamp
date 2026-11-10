"""A 4 by 4 grid and tabular Q-learning. Deterministic moves, one seed."""

import numpy as np

N = 4
START = (0, 0)
GOAL = (3, 3)
PIT = (3, 1)
STEP_COST = -0.04
GOAL_REWARD = 1.0
PIT_REWARD = -1.0
# up, right, down, left
ACTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
ARROWS = "↑→↓←"


def state_index(row: int, col: int) -> int:
    return row * N + col


def index_rc(state: int) -> tuple[int, int]:
    return divmod(int(state), N)


def step(state: int, action: int) -> tuple[int, float, bool]:
    """One move. Stepping into the goal or the pit ends the episode."""
    row, col = index_rc(state)
    d_row, d_col = ACTIONS[action]
    row = min(max(row + d_row, 0), N - 1)
    col = min(max(col + d_col, 0), N - 1)
    nxt = state_index(row, col)
    if (row, col) == GOAL:
        return nxt, GOAL_REWARD, True
    if (row, col) == PIT:
        return nxt, PIT_REWARD, True
    return nxt, STEP_COST, False


def q_learning(
    episodes: int = 400,
    alpha: float = 0.5,
    gamma: float = 0.9,
    epsilon: float = 0.2,
    seed: int = 42,
    max_steps: int = 40,
):
    """Return the Q table and the total reward of each episode."""
    rng = np.random.default_rng(seed)
    q = np.zeros((N * N, len(ACTIONS)))
    totals = np.empty(episodes)
    start = state_index(*START)
    for episode in range(episodes):
        state = start
        total = 0.0
        for _ in range(max_steps):
            if rng.random() < epsilon:
                action = int(rng.integers(0, len(ACTIONS)))
            else:
                action = int(np.argmax(q[state]))
            nxt, reward, done = step(state, action)
            future = 0.0 if done else gamma * float(np.max(q[nxt]))
            q[state, action] += alpha * (reward + future - q[state, action])
            total += reward
            state = nxt
            if done:
                break
        totals[episode] = total
    return q, totals


def greedy_policy(q: np.ndarray) -> np.ndarray:
    return np.argmax(q, axis=1)


def policy_arrows(policy: np.ndarray) -> str:
    lines = []
    for row in range(N):
        cells = []
        for col in range(N):
            if (row, col) == GOAL:
                cells.append("G")
            elif (row, col) == PIT:
                cells.append("X")
            else:
                cells.append(ARROWS[int(policy[state_index(row, col)])])
        lines.append(" ".join(cells))
    return "\n".join(lines)


def reaches_goal(policy: np.ndarray, max_steps: int = 20) -> bool:
    """True when the greedy moves reach the goal and never step on the pit."""
    state = state_index(*START)
    for _ in range(max_steps):
        nxt, _reward, done = step(state, int(policy[state]))
        row, col = index_rc(nxt)
        if (row, col) == PIT:
            return False
        if done and (row, col) == GOAL:
            return True
        if nxt == state:
            return False
        state = nxt
    return False
