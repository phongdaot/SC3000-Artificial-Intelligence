import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from mdp import GridWorld5x5Env
np.random.seed(42)  # for reproducibility

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
        plt.savefig(save_path, dpi=400)

def compare_learning_curves(returns1, returns2, chunk_size=50, label1="Method 1", label2="Method 2", title="Learning Curve Comparison", save_path=None):
    plt.figure(figsize=(8, 3))

    variance1 = np.var(returns1)
    variance2 = np.var(returns2)

    # Compute chunk minima for both methods
    def compute_chunk_minima(returns):
        chunk_min = []
        chunk_x = []
        for i in range(0, len(returns), chunk_size):
            chunk = returns[i:i + chunk_size]
            if len(chunk) == 0:
                continue
            chunk_min.append(np.min(chunk))
            chunk_x.append(i + len(chunk) - 1)
        return chunk_x, chunk_min

    if len(returns1) >= chunk_size:
        chunk_x1, chunk_min1 = compute_chunk_minima(returns1)
        plt.plot(chunk_x1, chunk_min1, linewidth=1, label=f"{label1} Chunk min ({chunk_size})")

    if len(returns2) >= chunk_size:
        chunk_x2, chunk_min2 = compute_chunk_minima(returns2)
        plt.plot(chunk_x2, chunk_min2, linewidth=1, label=f"{label2} Chunk min ({chunk_size})")

    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title(title)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()
    plt.figtext(x=0.99, y=0.01, s=f"{label1} Var: {variance1:.2f}, {label2} Var: {variance2:.2f}", ha="right", fontsize=8)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=400)

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
        plt.savefig(save_path, dpi=400)


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


def plot_value_function(env, V, title="Value Function", save_path=None):
    grid = value_to_grid(env, V)

    plt.figure(figsize=(6, 5))
    im = plt.imshow(grid, origin="lower")
    plt.colorbar(im, label="V(s)")
    plt.title(title)
    plt.xticks(range(env.W))
    plt.yticks(range(env.H))

    for y in range(env.H):
        for x in range(env.W):
            s = (x, y)
            if s in env.blocks:
                plt.text(x, y, "#", ha="center", va="center", fontsize=14, fontweight="bold")
            elif s == env.goal:
                plt.text(x, y, "G", ha="center", va="center", fontsize=14, fontweight="bold", color="white")
            else:
                plt.text(x, y, f"{grid[y, x]:.2f}", ha="center", va="center", fontsize=9, color="white")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=400)

def plot_policy(env, policy, title="Policy", save_path=None):
    plt.figure(figsize=(6, 5))
    plt.xlim(-0.5, env.W - 0.5)
    plt.ylim(-0.5, env.H - 0.5)
    plt.xticks(range(env.W))
    plt.yticks(range(env.H))
    plt.grid(True)
    plt.gca().set_aspect("equal")
    plt.title(title)

    for y in range(env.H):
        for x in range(env.W):
            s = (x, y)

            if s in env.blocks:
                plt.fill_between([x - 0.5, x + 0.5], y - 0.5, y + 0.5, color="black")
                plt.text(x, y, "#", ha="center", va="center", color="white", fontsize=14, fontweight="bold")
                continue

            if s == env.goal:
                plt.fill_between([x - 0.5, x + 0.5], y - 0.5, y + 0.5, color="lightgreen")
                plt.text(x, y, "G", ha="center", va="center", fontsize=14, fontweight="bold")
                continue

            if s == env.start:
                plt.fill_between([x - 0.5, x + 0.5], y - 0.5, y + 0.5, color="lightblue", alpha=0.4)

            a = policy.get(s, None)
            plt.text(x, y, ARROW.get(a, "?"), ha="center", va="center", fontsize=18)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=400)
    
def compare_value_functions(env, V1, V2, label1="Value Iteration", label2="Policy Iteration", save_path=None):
    grid1 = value_to_grid(env, V1)
    grid2 = value_to_grid(env, V2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, grid, title in zip(
        axes,
        [grid1, grid2],
        [label1, label2]
    ):
        im = ax.imshow(grid, origin="lower")
        ax.set_title(title)
        ax.set_xticks(range(env.W))
        ax.set_yticks(range(env.H))

        for y in range(env.H):
            for x in range(env.W):
                s = (x, y)
                if s in env.blocks:
                    ax.text(x, y, "#", ha="center", va="center", fontsize=14, fontweight="bold")
                elif not np.isnan(grid[y, x]):
                    ax.text(x, y, f"{grid[y, x]:.2f}", ha="center", va="center", fontsize=8, color="white")

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=720)

def compare_policies_plot(env, pi1, pi2, pi3, pi4, label1="PI1", label2="PI2", label3="PI3", label4="PI4", save_path=None):
    plt.figure(figsize=(6, 5))
    plt.xlim(-0.5, env.W - 0.5)
    plt.ylim(-0.5, env.H - 0.5)
    plt.xticks(range(env.W))
    plt.yticks(range(env.H))
    plt.grid(True)
    plt.gca().set_aspect("equal")
    plt.title(f"Policy Comparisons:")

    for y in range(env.H):
        for x in range(env.W):
            s = (x, y)

            if s in env.blocks:
                plt.fill_between([x - 0.5, x + 0.5], y - 0.5, y + 0.5, color="black")
                plt.text(x, y, "#", ha="center", va="center", color="white", fontsize=14, fontweight="bold")
                continue

            if s == env.goal:
                plt.fill_between([x - 0.5, x + 0.5], y - 0.5, y + 0.5, color="lightgreen")
                plt.text(x, y, "G", ha="center", va="center", fontsize=14, fontweight="bold")
                continue

            a1, a2, a3, a4 = pi1.get(s), pi2.get(s), pi3.get(s), pi4.get(s)

            plt.fill_between([x - 0.5, x + 0.5], y - 0.5, y + 0.5, color="lightgray", alpha=0.5)

            # Top row: pi1 (blue) / pi2 (orange)
            plt.text(x - 0.08, y + 0.18, ARROW.get(a1, "?"), ha="right", va="center", fontsize=11, color="tab:blue")
            plt.text(x,        y + 0.18, "/",                ha="center", va="center", fontsize=11, color="black")
            plt.text(x + 0.08, y + 0.18, ARROW.get(a2, "?"), ha="left",  va="center", fontsize=11, color="tab:orange")

            # Bottom row: pi3 (green) / pi4 (red)
            plt.text(x - 0.08, y - 0.18, ARROW.get(a3, "?"), ha="right", va="center", fontsize=11, color="tab:green")
            plt.text(x,        y - 0.18, "/",                ha="center", va="center", fontsize=11, color="black")
            plt.text(x + 0.08, y - 0.18, ARROW.get(a4, "?"), ha="left",  va="center", fontsize=11, color="tab:red")

    legend_handles = [
        Line2D([0], [0], color="tab:blue",   lw=2, label=f"{label1} (top-left)"),
        Line2D([0], [0], color="tab:orange", lw=2, label=f"{label2} (top-right)"),
        Line2D([0], [0], color="tab:green",  lw=2, label=f"{label3} (bottom-left)"),
        Line2D([0], [0], color="tab:red",    lw=2, label=f"{label4} (bottom-right)"),
    ]
    plt.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=2,
        frameon=False,
        title="Arrow Colors"
    )

    plt.tight_layout(rect=[0, 0.1, 1, 1])
    if save_path:
        plt.savefig(save_path, dpi=400)
    
# =========================================================
# Comparison Stats
# =========================================================
def compare_policies_stats(env, pi1, pi2, label1="VI", label2="PI"):
    same = 0
    total = 0
    different_states = []

    for s in all_states(env):
        if s in env.blocks or env.is_terminal_state(s):
            continue

        total += 1
        if pi1.get(s) == pi2.get(s):
            same += 1
        else:
            different_states.append((s, pi1.get(s), pi2.get(s)))

    print(f"{label1} vs {label2}: {same}/{total} states identical ({100 * same / total:.2f}%)")

    if different_states:
        print("States with different actions:")
        for s, a1, a2 in different_states:
            print(f"  {s}: {label1}={ARROW.get(a1, '?')}, {label2}={ARROW.get(a2, '?')}")
    else:
        print("Policies are identical on all non-terminal, non-block states.")

def all_states(env: GridWorld5x5Env):
    """Return all states in coordinate form."""
    return [env._id_to_state(sid) for sid in range(env.observation_space.n)]


def non_terminal_states(env: GridWorld5x5Env):
    """Return all non-terminal, non-block states."""
    return [
        s for s in all_states(env)
        if (s not in env.blocks) and (not env.is_terminal_state(s))
    ]


def value_to_grid(env, V):
    """Convert value dict V[(x,y)] to 2D grid for plotting."""
    grid = np.full((env.H, env.W), np.nan, dtype=float)

    for y in range(env.H):
        for x in range(env.W):
            s = (x, y)
            if s in env.blocks:
                continue
            grid[y, x] = V.get(s, 0.0)

    return grid