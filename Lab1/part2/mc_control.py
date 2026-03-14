import numpy as np
from collections import defaultdict
from tqdm import tqdm
from mdp import GridWorld5x5Env
import matplotlib.pyplot as plt

# -----------------------
# Utilities
# -----------------------
ARROW = {0: "↑", 1: "←", 2: "↓", 3: "→"}

def eps_greedy_action_from_Q(Q, s, n_actions, eps=0.1):
    """ε-greedy action selection w.r.t. Q(s,a) stored as Q[(s,a)]."""
    if np.random.rand() < eps:
        return np.random.randint(n_actions)
    qs = np.array([Q[(s, a)] for a in range(n_actions)], dtype=float)
    max_q = np.max(qs)
    best_actions = np.flatnonzero(qs == max_q)
    return int(np.random.choice(best_actions))

def greedy_policy_from_Q(Q, states, n_actions):
    """Return greedy deterministic policy: state_id -> best_action."""
    policy = {}
    for s in states:
        qs = np.array([Q[(s, a)] for a in range(n_actions)], dtype=float)
        best_actions = np.flatnonzero(qs == np.max(qs))
        policy[s] = int(np.random.choice(best_actions))
    return policy

def print_policy_from_stateid(env, policy_stateid):
    """policy_stateid maps state_id -> action"""
    for y in reversed(range(env.H)):
        row = []
        for x in range(env.W):
            s = (x, y)
            if s in env.blocks:
                row.append(" # ")
            elif s == env.goal:
                row.append(" G ")
            else:
                sid = env._state_to_id(s)
                a = policy_stateid.get(sid, None)
                row.append(f" {ARROW.get(a,'?')} ")
        print("".join(row))

# -----------------------
# Monte Carlo Control (First-Visit, ε-soft behavior)
# -----------------------
def mc_control(env: GridWorld5x5Env, episodes=20000, eps=0.1, gamma=0.9, max_steps=500):
    """
    First-visit Monte Carlo control (ε-greedy behavior w.r.t. Q, improves implicitly as Q improves).

    Returns:
    Q: defaultdict(float) mapping (s_id, a) -> Q-value
    pi_greedy: dict mapping s_id -> greedy action from Q
    returns: list of episode returns (sum of rewards per episode)
    lengths: list of episode lengths
    """
    n_actions = env.action_space.n

    Q = defaultdict(float) # Q[(s,a)] -> value
    N = defaultdict(int) # N[(s,a)] -> visit count (for incremental mean)

    returns = []
    lengths = []

    for ep in tqdm(range(episodes)):
        obs, info = env.reset()
        episode = []

        # Generate one episode
        for t in range(max_steps):
            s = int(obs)
            a = eps_greedy_action_from_Q(Q, s, n_actions, eps=eps)
            obs2, r, terminated, truncated, info = env.step(a)

            episode.append((s, a, r))
            obs = obs2

            if terminated or truncated:
                break

        # Log episode stats (useful for learning curve)
        returns.append(sum(r for (_, _, r) in episode))
        lengths.append(len(episode))

        # First-visit MC update: process returns backward
        visited = set()
        G = 0.0
        for (s, a, r) in reversed(episode):
            G = r + gamma * G
            if (s, a) in visited:
                continue
            visited.add((s, a))

            N[(s, a)] += 1

            # Q_old * (N - 1) + G = Q_new * N
            # Q_new = (Q_old * (N - 1) + G) / N
            # Q_new = Q_old += (Q_new - Q_old)
            # = (Q_old * (N - 1) + G) / N - Q_old
            # = G / N - Q_old / N
            # = (G - Q_old) / N
            Q[(s, a)] += (G - Q[(s, a)]) / N[(s, a)]

    # Greedy policy extracted from Q for all non-terminal valid states
    states_ids = [env._state_to_id(s) for s in env.states if s != env.goal]
    pi_greedy = greedy_policy_from_Q(Q, states_ids, n_actions)

    return Q, pi_greedy, returns, lengths
 
 

def q_learning(env, episodes=20000, eps=0.1, alpha=0.1, gamma=0.9, max_steps=500):

    """
    Tabular Q-learning with ε-greedy exploration (fixed eps=0.1) and fixed learning rate alpha=0.1.

    Assumes env.obs_mode="discrete" so that obs is an integer state id.
    Uses env.step(a) to sample transitions (agent does not know transition model).

    Returns:
    Q: defaultdict(float) mapping (s_id, a) -> Q-value
    pi_greedy: dict mapping s_id -> greedy action from Q
    returns: list of episode returns
    lengths: list of episode lengths
    """
    n_actions = env.action_space.n

    Q = defaultdict(float)
    returns = []
    lengths = []

    def eps_greedy_action(Q, s):
        if np.random.rand() < eps:
            return int(np.random.randint(n_actions))
        qs = np.array([Q[(s, a)] for a in range(n_actions)], dtype=float)
        max_q = np.max(qs)
        best_actions = np.flatnonzero(qs == max_q)
        return int(np.random.choice(best_actions))

    for ep in tqdm(range(episodes)):
        obs, info = env.reset()
        total_r = 0.0

        for t in range(max_steps):
            s = int(obs)
            a = eps_greedy_action(Q, s)

            obs2, r, terminated, truncated, info = env.step(a)
            s2 = int(obs2)
            done = terminated or truncated

            # TD target: r + gamma * max_a' Q(s',a') (0 if terminal)
            if done:
                target = r
            else:
                next_qs = np.array([Q[(s2, ap)] for ap in range(n_actions)], dtype=float)
                target = r + gamma * np.max(next_qs)

            # Q-learning update
            Q[(s, a)] += alpha * (target - Q[(s, a)])

            total_r += r
            obs = obs2

            if done:
                break

        returns.append(total_r)
        lengths.append(t + 1)

    # Extract greedy policy over valid env.states (exclude goal)
    states_ids = [env._state_to_id(s) for s in env.states if s != env.goal]
    pi_greedy = {}
    for s in states_ids:
        qs = np.array([Q[(s, a)] for a in range(n_actions)], dtype=float)
        best_actions = np.flatnonzero(qs == np.max(qs))
        pi_greedy[s] = int(np.random.choice(best_actions))

# -----------------------
# Plotting
# -----------------------
def plot_learning_curve(returns, window=50, title="MC Control Learning Curve (Episode Return)"):

    """
    Plots raw episode return + moving average.
    (This is the right 'convergence' visualization for MC, not log(delta).)
    """
    plt.figure()
    plt.plot(returns, linewidth=1, alpha=0.35, label="Episode return")

    if len(returns) >= window:
        ma = np.convolve(returns, np.ones(window) / window, mode="valid")
        plt.plot(range(window - 1, len(returns)), ma, linewidth=2, label=f"Moving avg ({window})")

    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title(title)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("mc_learning_curve.png", dpi=200)
    plt.show()
    """
 Shows 4 heatmaps (one per action) with values printed in each cell.
 """
    """
    Shows 4 heatmaps (one per action) with values printed in each cell.
    """
    n_actions = env.action_space.n
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for action in range(n_actions):
        grid = np.full((env.H, env.W), np.nan, dtype=float)

        for y in range(env.H):
            for x in range(env.W):
                s = (x, y)
                if s in env.blocks:
                    continue
                sid = env._state_to_id(s)
                grid[y, x] = Q[(sid, action)]

        ax = axes[action]
        im = ax.imshow(grid)
        ax.set_title(f"Q(s,a) for action {ARROW[action]}")
        ax.set_xticks(range(env.W))
        ax.set_yticks(range(env.H))
        ax.invert_yaxis()

        # Print values
        for y in range(env.H):
            for x in range(env.W):
                if not np.isnan(grid[y, x]):
                    ax.text(x, y, f"{grid[y, x]:.2f}", ha="center", va="center", fontsize=8)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.show()
    
# Run
# -----------------------
if __name__ == "__main__":
    env = GridWorld5x5Env(gamma=0.9, stochastic=True) # Task 2 setting (unknown stochastic dynamics)
    Q_mc, pi_mc, returns, lengths = mc_control(env, episodes=20000, eps=0.1, gamma=0.9, max_steps=500)

    env = GridWorld5x5Env(gamma=0.9, stochastic=True) # Task 2 setting (unknown stochastic dynamics)
    Q_mc, pi_mc, returns, lengths = mc_control(env, episodes=20000, eps=0.1, gamma=0.9, max_steps=500)

    print("=== MC Learned Greedy Policy (trained with ε=0.1) ===")
    print_policy_from_stateid(env, pi_mc)

    plot_learning_curve(returns, window=100, title="MC Control: Episode Return vs Episodes")