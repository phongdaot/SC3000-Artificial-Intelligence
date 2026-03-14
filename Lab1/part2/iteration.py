import numpy as np
import gymnasium as gym
from mdp import GridWorld5x5Env
import matplotlib.pyplot as plt
import numpy as np

def all_states(env: GridWorld5x5Env):
    return [env._id_to_state(s) for s in range(env.observation_space.n)]

def plot_convergence(deltas, theta=None, logy=True, title="Value Iteration Convergence"):
    plt.figure()
    plt.plot(deltas, marker='o', markersize=2, linewidth=1)
    plt.xlabel("Iteration (sweep)")
    plt.ylabel("Bellman residual (max |ΔV|)")
    plt.title(title)

    if logy:
        plt.yscale("log")

    if theta is not None:
        plt.axhline(theta, linestyle="--", linewidth=1)

    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.savefig("iteration_convergence")


def value_iteration(env: GridWorld5x5Env, theta=1e-6, max_iters=10000):
    V = {s: 0.0 for s in all_states(env)}
    policy = {}
    history = []

    for it in range(max_iters):
        delta = 0.0
        G = {s: 0.0 for s in all_states(env)}

        deltas = []
        for s in all_states(env):
            if env.is_terminal_state(s):
                continue

            old = V[s]
            best = (-1e18, 0)

            for a in range(env.action_space.n):
                q = 0.0
                for prob, s2, r, done in env.transitions(s, a):
                    q += prob * (r + env.gamma * (0.0 if done else V[s2]))

                best = max(best, (q, a))

            G[s], policy[s] = best

            deltas.append(abs(old - G[s]))
            delta = max(delta, abs(old - G[s]))

        V = G
        history.append(np.average(deltas))
        if delta < theta:
            break

    return V, policy, history
 

def policy_evaluation(env, policy, theta=1e-6):
    V = {s: 0.0 for s in all_states(env)}
    while True:
        delta = 0.0
        for s in all_states(env):
            if env.is_terminal_state(s):
                continue
            old = V[s]
            a = policy[s]
            v = 0.0
            for prob, s2, r, done in env.transitions(s, a):
                v += prob * (r + env.gamma * (0.0 if done else V[s2]))
            V[s] = v
            delta = max(delta, abs(old - v))
        if delta < theta:
            break
    return V

def policy_iteration(env):
    # init random policy
    policy = {s: np.random.randint(env.action_space.n) for s in all_states(env) if not env.is_terminal_state(s)}

    while True:
        V = policy_evaluation(env, policy)
        stable = True
        for s in all_states(env):
            if env.is_terminal_state(s):
                continue
            old_a = policy[s]
            best_q = -1e18
            best_a = old_a
            for a in range(env.action_space.n):
                q = 0.0
                for prob, s2, r, done in env.transitions(s, a):
                    q += prob * (r + env.gamma * (0.0 if done else V[s2]))
                if q > best_q:
                    best_q, best_a = q, a
            policy[s] = best_a
            if best_a != old_a:
                stable = False
        if stable:
            return V, policy


env = GridWorld5x5Env(gamma=0.9, stochastic=False)


ARROW = {
    0: "↑", # Up
    1: "←", # Left
    2: "↓", # Down
    3: "→", # Right
}

def print_value_grid(env, V, fmt="{:7.2f}"):
    """Print V(s) as a 5x5 grid."""
    for y in reversed(range(env.H)):
        row = []
        for x in range(env.W):
            s = (x, y)
            if s in env.blocks:
                row.append(" # ")
            elif s == env.goal:
                row.append(" G ")
            else:
                row.append(fmt.format(V.get(s, 0.0)))
        print(" ".join(row))

def print_policy_grid(env, policy):
    """Print policy arrows as a 5x5 grid."""
    for y in reversed(range(env.H)):
        row = []
        for x in range(env.W):
            s = (x, y)
            if s in env.blocks:
                row.append(" # ")
            elif s == env.goal:
                row.append(" G ")
            else:
                a = policy.get(s, None)
                row.append(f" {ARROW.get(a,'?')} ")
        print("".join(row))

def overlay_policy_on_values(env, V, policy, vfmt="{:6.2f}"):
    """Print a combined grid: arrow + value in each cell."""
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
                row.append(f"{arrow}{vfmt.format(V.get(s,0.0))}")
        print(" ".join(row))

V_vi, pi_vi, deltas = value_iteration(env, theta=1e-6)
print("=== Value Function V*(s) ===")
print_value_grid(env, V_vi)

print("\n=== Optimal Policy π*(s) ===")
print_policy_grid(env, pi_vi)

print("\n=== Overlay (arrow + value) ===")
overlay_policy_on_values(env, V_vi, pi_vi)


plot_convergence(deltas, theta=1e-6, logy=True, title="Value Iteration (deterministic) Convergence")
print("Converged in", len(deltas), "iterations")