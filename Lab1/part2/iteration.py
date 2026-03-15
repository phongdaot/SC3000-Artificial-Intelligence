import numpy as np
import matplotlib.pyplot as plt
from mdp import GridWorld5x5Env
import os
from utils import plot_value_function, plot_policy, compare_value_functions, compare_policies_plot, compare_policies_stats, all_states, value_to_grid

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

def value_iteration(env: GridWorld5x5Env, theta=1e-6, max_iterations=10000):
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

    for _ in range(max_iterations):
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


def policy_evaluation(env: GridWorld5x5Env, policy, theta=1e-6, max_iterations=10000):
    """
    Iterative policy evaluation for a fixed deterministic policy.
    """
    states = all_states(env)
    V = {s: 0.0 for s in states}

    for _ in range(max_iterations):
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


def policy_iteration(env: GridWorld5x5Env, theta=1e-6, max_iterations=1000):
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

    for _ in range(max_iterations):
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