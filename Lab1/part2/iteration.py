import numpy as np
import matplotlib.pyplot as plt
from mdp import GridWorld5x5Env
import os

SAVE_DIR = "results/task1"
# =========================================================
# Constants
# =========================================================
ARROW = {
    0: "↑",   # Up
    1: "←",   # Left
    2: "↓",   # Down
    3: "→",   # Right
}

# =========================================================
# Utilities
# =========================================================
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


# =========================================================
# Planning Algorithms
# =========================================================
def value_iteration(env: GridWorld5x5Env, theta=1e-6, max_iters=10000):
    """
    Value Iteration:
        V_{k+1}(s) = max_a sum_{s',r} P(s',r|s,a)[r + gamma V_k(s')]

    Returns:
        V       : optimal value function
        policy  : greedy policy derived from V
        history : max Bellman residual per sweep
    """
    states = all_states(env)
    V = {s: 0.0 for s in states}
    history = []

    for _ in range(max_iters):
        delta = 0.0
        V_new = V.copy()

        for s in states:
            if s in env.blocks or env.is_terminal_state(s):
                continue

            old_v = V[s]
            best_q = -np.inf

            for a in range(env.action_space.n):
                q = 0.0
                for prob, s2, r, done in env.transitions(s, a):
                    q += prob * (r + env.gamma * (0.0 if done else V[s2]))
                best_q = max(best_q, q)

            V_new[s] = best_q
            delta = max(delta, abs(old_v - V_new[s]))

        V = V_new
        history.append(delta)

        if delta < theta:
            break

    # Extract greedy policy
    policy = {}
    for s in states:
        if s in env.blocks or env.is_terminal_state(s):
            continue

        best_q = -np.inf
        best_a = None

        for a in range(env.action_space.n):
            q = 0.0
            for prob, s2, r, done in env.transitions(s, a):
                q += prob * (r + env.gamma * (0.0 if done else V[s2]))

            if q > best_q:
                best_q = q
                best_a = a

        policy[s] = best_a

    return V, policy, history


def policy_evaluation(env: GridWorld5x5Env, policy, theta=1e-6, max_iters=10000):
    """
    Iterative policy evaluation for a fixed deterministic policy.
    """
    states = all_states(env)
    V = {s: 0.0 for s in states}

    for _ in range(max_iters):
        delta = 0.0

        for s in states:
            if s in env.blocks or env.is_terminal_state(s):
                continue

            old_v = V[s]
            a = policy[s]

            v = 0.0
            for prob, s2, r, done in env.transitions(s, a):
                v += prob * (r + env.gamma * (0.0 if done else V[s2]))

            V[s] = v
            delta = max(delta, abs(old_v - v))

        if delta < theta:
            break

    return V


def policy_iteration(env: GridWorld5x5Env, theta=1e-6, max_policy_iters=1000):
    """
    Policy Iteration:
        1. Policy Evaluation
        2. Policy Improvement
        Repeat until stable

    Returns:
        V       : value function of final policy
        policy  : optimal policy
    """
    states = all_states(env)

    # Initialize random deterministic policy
    policy = {}
    for s in states:
        if s in env.blocks or env.is_terminal_state(s):
            continue
        policy[s] = np.random.randint(env.action_space.n)

    for _ in range(max_policy_iters):
        V = policy_evaluation(env, policy, theta=theta)

        policy_stable = True

        for s in states:
            if s in env.blocks or env.is_terminal_state(s):
                continue

            old_a = policy[s]
            best_q = -np.inf
            best_a = old_a

            for a in range(env.action_space.n):
                q = 0.0
                for prob, s2, r, done in env.transitions(s, a):
                    q += prob * (r + env.gamma * (0.0 if done else V[s2]))

                if q > best_q:
                    best_q = q
                    best_a = a

            policy[s] = best_a
            if best_a != old_a:
                policy_stable = False

        if policy_stable:
            return V, policy

    return V, policy


# =========================================================
# Printing Helpers
# =========================================================
def print_value_grid(env, V, fmt="{:7.2f}"):
    """Print value function in grid form."""
    for y in reversed(range(env.H)):
        row = []
        for x in range(env.W):
            s = (x, y)
            if s in env.blocks:
                row.append("   #   ")
            elif s == env.goal:
                row.append("   G   ")
            else:
                row.append(fmt.format(V.get(s, 0.0)))
        print(" ".join(row))


def print_policy_grid(env, policy):
    """Print deterministic policy as arrows."""
    for y in reversed(range(env.H)):
        row = []
        for x in range(env.W):
            s = (x, y)
            if s in env.blocks:
                row.append(" # ")
            elif s == env.goal:
                row.append(" G ")
            elif s == env.start:
                row.append(" S ")
            else:
                a = policy.get(s, None)
                row.append(f" {ARROW.get(a, '?')} ")
        print("".join(row))


def overlay_policy_on_values(env, V, policy, vfmt="{:6.2f}"):
    """Print arrow + value in each state."""
    for y in reversed(range(env.H)):
        row = []
        for x in range(env.W):
            s = (x, y)
            if s in env.blocks:
                row.append(" #### ")
            elif s == env.goal:
                row.append(" GGGG ")
            else:
                a = policy.get(s, None)
                arrow = ARROW.get(a, "?")
                row.append(f"{arrow}{vfmt.format(V.get(s, 0.0))}")
        print(" ".join(row))


# =========================================================
# Plotting
# =========================================================
def plot_convergence(deltas, theta=None, logy=True, title="Value Iteration Convergence", save_path=None):
    plt.figure(figsize=(7, 5))
    plt.plot(deltas, marker='o', markersize=3, linewidth=1.2)
    plt.xlabel("Iteration (sweep)")
    plt.ylabel("Bellman residual (max |ΔV|)")
    plt.title(title)

    if logy:
        plt.yscale("log")

    if theta is not None:
        plt.axhline(theta, linestyle="--", linewidth=1, label=f"theta={theta}")
        plt.legend()

    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()

    if save_path:
        plt.savefig(os.path.join(SAVE_DIR, save_path), dpi=200)


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
        plt.savefig(os.path.join(SAVE_DIR, save_path), dpi=200)


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
        plt.savefig(os.path.join(SAVE_DIR, save_path), dpi=200)
    


def compare_value_functions(env, V1, V2, label1="Value Iteration", label2="Policy Iteration", save_path=None):
    grid1 = value_to_grid(env, V1)
    grid2 = value_to_grid(env, V2)
    diff = np.abs(grid1 - grid2)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, grid, title in zip(
        axes,
        [grid1, grid2, diff],
        [label1, label2, "|Difference|"]
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
        plt.savefig(os.path.join(SAVE_DIR, save_path), dpi=200)


def compare_policies_plot(env, pi1, pi2, label1="VI", label2="PI", save_path=None):
    plt.figure(figsize=(6, 5))
    plt.xlim(-0.5, env.W - 0.5)
    plt.ylim(-0.5, env.H - 0.5)
    plt.xticks(range(env.W))
    plt.yticks(range(env.H))
    plt.grid(True)
    plt.gca().set_aspect("equal")
    plt.title(f"Policy Comparison: {label1} vs {label2}")

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

            a1 = pi1.get(s, None)
            a2 = pi2.get(s, None)

            if a1 == a2:
                plt.fill_between([x - 0.5, x + 0.5], y - 0.5, y + 0.5, color="lightgray", alpha=0.5)
                txt = ARROW.get(a1, "?")
            else:
                plt.fill_between([x - 0.5, x + 0.5], y - 0.5, y + 0.5, color="salmon", alpha=0.7)
                txt = f"{ARROW.get(a1, '?')}/{ARROW.get(a2, '?')}"

            plt.text(x, y, txt, ha="center", va="center", fontsize=14)

    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(SAVE_DIR, save_path), dpi=200)
    


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


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    # Assignment Task 1 uses stochastic dynamics
    env = GridWorld5x5Env(gamma=0.9, stochastic=True)

    # -----------------------------
    # Value Iteration
    # -----------------------------
    V_vi, pi_vi, vi_history = value_iteration(env, theta=1e-6)

    print("=== VALUE ITERATION: VALUE FUNCTION ===")
    print_value_grid(env, V_vi)

    print("\n=== VALUE ITERATION: POLICY ===")
    print_policy_grid(env, pi_vi)

    print("\n=== VALUE ITERATION: OVERLAY ===")
    overlay_policy_on_values(env, V_vi, pi_vi)

    # -----------------------------
    # Policy Iteration
    # -----------------------------
    V_pi, pi_pi = policy_iteration(env, theta=1e-6)

    print("\n=== POLICY ITERATION: VALUE FUNCTION ===")
    print_value_grid(env, V_pi)

    print("\n=== POLICY ITERATION: POLICY ===")
    print_policy_grid(env, pi_pi)

    print("\n=== POLICY ITERATION: OVERLAY ===")
    overlay_policy_on_values(env, V_pi, pi_pi)

    # -----------------------------
    # Comparison
    # -----------------------------
    print("\n=== POLICY COMPARISON ===")
    compare_policies_stats(env, pi_vi, pi_pi, label1="Value Iteration", label2="Policy Iteration")

    # -----------------------------
    # Plots
    # -----------------------------
    plot_convergence(
        vi_history,
        theta=1e-6,
        logy=True,
        title="Value Iteration Convergence",
        save_path="value_iteration_convergence.png"
    )

    plot_value_function(
        env, V_vi,
        title="Value Function from Value Iteration",
        save_path="value_function_vi.png"
    )

    plot_value_function(
        env, V_pi,
        title="Value Function from Policy Iteration",
        save_path="value_function_pi.png"
    )

    plot_policy(
        env, pi_vi,
        title="Policy from Value Iteration",
        save_path="policy_vi.png"
    )

    plot_policy(
        env, pi_pi,
        title="Policy from Policy Iteration",
        save_path="policy_pi.png"
    )

    compare_value_functions(
        env, V_vi, V_pi,
        label1="Value Iteration",
        label2="Policy Iteration",
        save_path="value_function_comparison.png"
    )

    compare_policies_plot(
        env, pi_vi, pi_pi,
        label1="VI",
        label2="PI",
        save_path="policy_comparison.png"
    )