from environments.MDP import Minigrid_MDP
from environments.LMDP import Minigrid
import numpy as np
import time
from utils import value_iteration, plot_greedy_policy, plot_value_function, plot_path, plot_episode_length, plot_episode_throughput, shortest_path_length, compare_throughputs
from models.qlearning import QLearning, Qlearning_training
from models.zlearning import compute_Pu_sparse, plot_sample_path, ZLearning, Zlearning_training, compare_Zlearning_estimates, show_Z, show_Pu, power_iteration, value_function_to_Z
from scipy.sparse import csr_matrix
from itertools import permutations


if __name__ == "__main__":
    walls = [(14,1), (1,8), (5, 5), (12, 5), (8, 7), (2,5), (3,5), (4,5), (6,5), (7,5), (8,5), (9,5), (10,5), (11,5), (13,5), (15,9)]
    # MDP
    minigrid_mdp = Minigrid_MDP(grid_size=17, walls = walls)
    minigrid_mdp.print_attributes()

    gamma = 0.95
    epsilon = 1e-6
    Q0 = np.zeros((minigrid_mdp.n_states, minigrid_mdp.n_actions))

    # Value Iteration MDP
    start_time = time.time()
    Q, opt_policy, n_steps = value_iteration(Q0, minigrid_mdp, epsilon, gamma)
    print("Value iteration took: ", n_steps, " steps before converging with epsilon:", epsilon)
    print("--- %s minutes and %s seconds ---" % (int((time.time() - start_time)/60), int((time.time() - start_time) % 60)))
    opt_lengths = list(shortest_path_length(minigrid_mdp,opt_policy, s) for s in range(minigrid_mdp.n_states))
    # plot_greedy_policy(opt_policy, minigrid_mdp, print_policy=True)
    # plot_value_function(Q, minigrid_mdp, print_values=True, file = "value_function.txt")
    plot_path(minigrid_mdp, opt_policy, path = 'plots\MDP_value_iteration_path.gif')

    # Q-learning
    # print("Q-learning training...")
    # qlearning = QLearning(minigrid_mdp, gamma=0.95, learning_rate=0.25, epsilon_min = 0)
    # start_time = time.time()
    # Q_est, est_policy, lengths, throughputs = Qlearning_training(qlearning, opt_lengths, n_steps=int(5e5))
    # print("--- %s minutes and %s seconds ---" % (int((time.time() - start_time)/60), int((time.time() - start_time) % 60)))
    # #plot_greedy_policy(est_policy, minigrid_mdp, print_policy=True, estimated=True)
    # plot_value_function(Q_est, minigrid_mdp, print_values=True, file = "QLearning_value_function.txt")
    # plot_path(minigrid_mdp, est_policy, path = 'plots\MDP_QLearning_path.gif')
    # plot_episode_length(lengths, shortest_path_length(minigrid_mdp,opt_policy), plot_batch=True)
    # q_averaged_throughputs = plot_episode_throughput(throughputs, shortest_path_length(minigrid_mdp,opt_policy), plot_batch=True)

    # LMDP
    minigrid = Minigrid(grid_size=17, walls=walls)
    minigrid.print_attributes()

    # Power Iteration LMDP
    e = 1e-30
    lmbda = 1
    Z0 = np.ones(minigrid.n_states)
    Z, n_steps = power_iteration(minigrid, lmbda = lmbda, epsilon=e)
    print("Value iteration took: ", n_steps, " steps before converging with epsilon:", e)
    print("\n\n")
    show_Z(Z,minigrid, print_Z=True, plot_Z = False, file = "Z_function_power_iteration.txt")
    PU = compute_Pu_sparse(Z, csr_matrix(minigrid.P0))
    # with open("PU_power_iteration.txt", "w") as f: # Print the transition matrix from power iteration
    #     for i in minigrid.S:
    #         for j in PU[i].indices:
    #                 if PU[i,j] != 0: f.write("Pu[{}][{}]: {}\n".format(minigrid.states[i], minigrid.states[j], PU[i,j]))
    #show_Pu(minigrid, PU, print_Pu=False, plot_Pu = False, is_sparse=True)
    plot_sample_path(minigrid, PU, path = 'plots\LMDP_power_iteration_path.gif')

    # Embedded MDP
    mdp_minigrid = minigrid.embedding_to_MDP()
    gamma = 0.95
    epsilon = 1e-30
    Q0 = np.zeros((mdp_minigrid.n_states, mdp_minigrid.n_actions))
    start_time = time.time()
    Q2, greedy_policy, n_steps = value_iteration(Q0, mdp_minigrid, epsilon, gamma)
    print("Value iteration took: ", n_steps, " steps before converging with epsilon:", epsilon)
    print("--- %s minutes and %s seconds ---" % (int((time.time() - start_time)/60), int((time.time() - start_time) % 60)))
    # plot_greedy_policy(greedy_policy, mdp_minigrid, print_policy=True)
    plot_value_function(Q2, mdp_minigrid, print_values=True, file = "value_function_embedded.txt")
    plot_path(mdp_minigrid, greedy_policy, path = 'plots\MDP_embedded_value_iteration_path.gif')
    Z2 = value_function_to_Z(Q2, lmbda = lmbda)
    show_Z(Z2,mdp_minigrid, print_Z=True, plot_Z = False, file = "Z_function_embedded.txt")
 
    # Z-Learning
    # print("Z-learning training...")
    # zlearning = ZLearning(minigrid, lmbda=1)
    # start_time = time.time()
    # Z_est, z_lengths, z_throughputs = Zlearning_training(zlearning, opt_lengths, n_steps=int(5e5))
    # print("--- %s minutes and %s seconds ---" % (int((time.time() - start_time)/60), int((time.time() - start_time) % 60)))
    # print("Total Absolute Error: ", np.sum(np.abs(PU[0:-4]-zlearning.Pu[0:-4])))
    # show_Z(Z_est[-1],minigrid, print_Z=True, plot_Z = False, file = "Z_function_zlearning.txt")
    # plot_sample_path(minigrid, zlearning.Pu, path = 'plots\LMDP_Z_learning_path.gif')
    # z_averaged_throughputs = plot_episode_throughput(z_throughputs, shortest_path_length(minigrid_mdp,opt_policy), plot_batch=True)
    # compare_throughputs(z_averaged_throughputs, q_averaged_throughputs, minigrid.grid_size, name1 = 'Z Learning', name2 = 'Q Learning')
    
    # with open("PU_zlearning", "w") as f: # Print the transition matrix from Z-learning
    #     for i in minigrid.S:
    #         for j in zlearning.Pu[i].indices:
    #                 if zlearning.Pu[i,j] != 0: f.write("Pu[{}][{}]: {}\n".format(minigrid.states[i], minigrid.states[j], zlearning.Pu[i,j]))

    # for i in minigrid.S: # Print the transition matrix differences between Z-learning and Power Iteration
    #        for j in (PU[i].indices):
    #             if np.abs(PU[i,j] -zlearning.Pu[i,j]) >= 0.1: 
    #                 print("Pu[", minigrid.states[i], "][", minigrid.states[j], "]: ", PU[i,j])
    #                 print("ZLearning Pu[", minigrid.states[i], "][", minigrid.states[j], "]: ", zlearning.Pu[i,j])