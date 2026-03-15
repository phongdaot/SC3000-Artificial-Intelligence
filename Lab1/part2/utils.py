import numpy as np
import gymnasium as gym
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

def plot_learning_curve(returns, chunk_size=50, title="Learning Curve", save_path=None):
    plt.figure(figsize=(8, 5))

    variance = np.var(returns)

    # Compute chunk maxima
    if len(returns) >= chunk_size:
        chunk_min = []
        chunk_x = []

        for i in range(0, len(returns), chunk_size):
            chunk = returns[i:i + chunk_size]
            if len(chunk) == 0:
                continue

            chunk_min.append(np.min(chunk))
            chunk_x.append(i + len(chunk) - 1)  # last episode index in chunk

        plt.plot(chunk_x, chunk_min, linewidth=1, label=f"Chunk min ({chunk_size})")

    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title(title)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()
    plt.figtext(x=0.99, y=0.01, s=f"Variance: {variance:.2f}", ha="right", fontsize=8)
    plt.tight_layout()

    
    if save_path is not None:
        plt.savefig(save_path, dpi=200)

def plot_q_heatmaps(env, Q, title_prefix="Q(s,a)", save_path=None):
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
        ax.set_title(f"{title_prefix}: {ARROW[action]}")
        ax.set_xticks(range(env.W))
        ax.set_yticks(range(env.H))
        ax.invert_yaxis()

        for y in range(env.H):
            for x in range(env.W):
                if not np.isnan(grid[y, x]):
                    ax.text(x, y, f"{grid[y, x]:.2f}", ha="center", va="center", fontsize=8)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200)


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
