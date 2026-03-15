from collections import defaultdict
from tqdm import tqdm
from mdp import GridWorld5x5Env
import matplotlib.pyplot as plt
from utils import eps_greedy_action_from_Q, greedy_policy_from_Q, plot_learning_curve, plot_q_heatmaps, print_policy_from_stateid
import os

SAVE_DIR = "results/task3"

def q_learning(env, episodes=2000, eps=0.1, alpha=0.1, gamma=0.9, max_steps=500):
    """
    Tabular Q-learning with ε-greedy exploration.
    
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

    for ep in tqdm(range(episodes)):
        obs, info = env.reset()
        total_r = 0.0

        for t in range(max_steps):
            s = int(obs)

            # ε-greedy action selection
            a = eps_greedy_action_from_Q(Q, s, n_actions, eps=eps)

            obs2, r, terminated, truncated, info = env.step(a)
            s2 = int(obs2)
            done = terminated or truncated

            if done:
                target = r
            else:
                target = r + gamma * max(Q[(s2, ap)] for ap in range(n_actions))

            # Q-learning update
            Q[(s, a)] += alpha * (target - Q[(s, a)])

            total_r += r
            obs = obs2

            if done:
                break

        returns.append(total_r)
        lengths.append(t + 1)

    # Extract greedy policy
    states_ids = [env._state_to_id(s) for s in env.states if s != env.goal]
    pi_greedy = greedy_policy_from_Q(Q, states_ids, n_actions)

    return Q, pi_greedy, returns, lengths

if __name__ == "__main__":
    env = GridWorld5x5Env(gamma=0.9, stochastic=False)
    Q, pi_greedy, returns, lengths = q_learning(env, episodes=20000, eps=0.1, alpha=0.1, gamma=0.9, max_steps=500)

    print("=== Q-Learning Learned Greedy Policy (trained with ε=0.1) ===")
    print_policy_from_stateid(env, pi_greedy)

    plot_learning_curve(returns, chunk_size=50, title="Q-Learning: Episode Return vs Episodes", save_path=os.path.join(SAVE_DIR, "q_learning_learning_curve.png"))
    plot_q_heatmaps(env, Q, title_prefix="Q-Learning Q(s,a)", save_path=os.path.join(SAVE_DIR, "q_learning_q_heatmaps.png"))