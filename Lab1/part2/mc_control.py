import numpy as np
from collections import defaultdict
from tqdm import tqdm
from mdp import GridWorld5x5Env
import matplotlib.pyplot as plt
from utils import eps_greedy_action_from_Q, greedy_policy_from_Q, plot_learning_curve, plot_q_heatmaps, print_policy_from_stateid, compare_policies_plot
from iteration import value_iteration, policy_iteration
import os



SAVE_DIR = "results/task2"

# -----------------------
# Monte Carlo Control (First-Visit, ε-soft behavior)
# -----------------------
def mc_control(env: GridWorld5x5Env, episodes=2000, eps=0.1, gamma=0.9, max_steps=500):
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


# Run
# -----------------------
if __name__ == "__main__":
    env = GridWorld5x5Env(gamma=0.9, stochastic=True, seed=42) # Task 2 setting (unknown stochastic dynamics)
    Q_mc, pi_mc, returns, lengths = mc_control(env, episodes=20000, eps=0.1, gamma=0.9, max_steps=500)
    print("=== MC Learned Greedy Policy (trained with ε=0.1) ===")
    print_policy_from_stateid(env, pi_mc)

    plot_learning_curve(returns, chunk_size=50, title="MC Control: Episode Return vs Episodes", save_path=os.path.join(SAVE_DIR, "mc_control_learning_curve.png"))
    plot_q_heatmaps(env, Q_mc, title_prefix="MC Control Q(s,a)", save_path=os.path.join(SAVE_DIR, "mc_control_q_heatmaps.png"))