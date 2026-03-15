from iteration import value_iteration, policy_iteration
from mc_control import mc_control
from q_learning import SAVE_DIR, q_learning
from mdp import GridWorld5x5Env
from utils import compare_policies_plot, compare_learning_curves, compare_value_functions
import os
import numpy as np

SAVE_DIR = "results/comparisons"

if __name__ == "__main__":
    env = GridWorld5x5Env(gamma=0.9, stochastic=True) # Deterministic setting for fair comparison of planning vs learning
    # np.random.seed(42) # for reproducibility of learning algorithms

    # Run Value Iteration
    V_vi, pi_vi, history_vi = value_iteration(env)
    
    # Run Policy Iteration
    V_pi, pi_pi = policy_iteration(env)
    
    # Run Monte Carlo Control
    Q_mc, pi_mc, returns_mc, lengths_mc = mc_control(env, episodes=20000, eps=0.1, gamma=0.9, max_steps=500)
    pi_mc_xy = {env._id_to_state(s_id): a for s_id, a in pi_mc.items()} # convert from state_id to (x,y) state
    Q_mc_xy = {(env._id_to_state(s_id), a): q for (s_id, a), q in Q_mc.items()} # convert from state_id to (x,y) state
    
    # Run Q-Learning
    Q_q, pi_q, returns_q, lengths_q = q_learning(env, episodes=20000, eps=0.1, alpha=0.1, gamma=0.9, max_steps=500)
    pi_q_xy = {env._id_to_state(s_id): a for s_id, a in pi_q.items()} # convert from state_id to (x,y) state
    Q_q_xy = {(env._id_to_state(s_id), a): q for (s_id, a), q in Q_q.items()} # convert from state_id to (x,y) state
    
    # Compare policies
    compare_policies_plot(env, pi_vi, pi_pi, pi_mc_xy, pi_q_xy, "Value Iteration", "Policy Iteration", "MC Control", "Q-Learning", save_path=os.path.join(SAVE_DIR, "policy_comparison.png"))
    
    # Compare value functions
    compare_value_functions(env, V_vi, V_pi, "Value Iteration", "Policy Iteration", save_path=os.path.join(SAVE_DIR, "value_function_comparison.png"))
    
    # Compare learning curves
    compare_learning_curves(returns_mc, returns_q, label1="MC Control (ε=0.1)", label2="Q-Learning (ε=0.1)", title="Episode Return Comparison", save_path=os.path.join(SAVE_DIR, "learning_curve_comparison.png"))
    