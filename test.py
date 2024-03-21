from environments.MDP import Minigrid_MDP
from environments.LMDP import Minigrid
import numpy as np
import time
from utils import Minigrid_MDP_Plots, plot_episode_length, plot_episode_throughput, compare_throughputs, plot_throughput, plot_value_per_hyperparameter
from models.qlearning import QLearning, Qlearning_training
from models.zlearning import ZLearning, Zlearning_training
from lmdp_plots import Minigrid_LMDP_Plots
from scipy.sparse import csr_matrix


if __name__ == "__main__":
    grid_size = 15
    walls = [(14,1), (1,8), (5, 5), (12, 5), (8, 7), (2,5), (3,5), (4,5), (6,5), (7,5), (8,5), (9,5), (10,5), (11,5), (13,5), (15,9)]
    # MDP
    minigrid_mdp = Minigrid_MDP(grid_size=grid_size, walls = walls)
    minigrid_mdp_plots = Minigrid_MDP_Plots(minigrid_mdp)
    # minigrid_mdp.print_attributes()
    #minigrid_mdp.render()

    gamma = 1
    epsilon = 1e-10

    # Value Iteration MDP
    # start_time = time.time()
    Q, opt_policy, n_steps = minigrid_mdp.value_iteration(epsilon, gamma)
    # print("Value iteration took: ", n_steps, " steps before converging with epsilon:", epsilon)
    # print("--- %s minutes and %s seconds ---" % (int((time.time() - start_time)/60), int((time.time() - start_time) % 60)))
    # opt_lengths = list(minigrid_mdp.shortest_path_length(opt_policy, s) for s in range(minigrid_mdp.n_states))
    # # plot_greedy_policy(opt_policy, minigrid_mdp, print_policy=True)
    #minigrid_mdp_plots.plot_value_function(Q, print_values=True, file = "value_function.txt")
    #minigrid_mdp_plots.plot_path(opt_policy, path = 'plots\MDP_value_iteration_path.gif')

    # Q-learning
    # print("Q-learning training...")
    # qlearning = QLearning(minigrid_mdp, gamma=gamma, learning_rate=0.25, learning_rate_decay=0.99999, learning_rate_min=0.005, epsilon=1, epsilon_decay=0.9995, epsilon_min = 0)
    # start_time = time.time()
    # Q_est, est_policy, lengths, throughputs = Qlearning_training(qlearning, n_steps=int(8e5))
    # print(qlearning.learning_rate, qlearning.epsilon)
    # print("--- %s minutes and %s seconds ---" % (int((time.time() - start_time)/60), int((time.time() - start_time) % 60)))
    # #plot_greedy_policy(est_policy, minigrid_mdp, print_policy=True, estimated=True)
    # minigrid_mdp_plots.plot_value_function(Q_est, print_values=True, file = "QLearning_value_function.txt")
    # #minigrid_mdp_plots.plot_path(est_policy, path = 'plots\MDP_QLearning_path.gif')
    # print(opt_policy - est_policy)
    # print("Total Absolute Error: ", np.sum(np.abs(Q-Q_est))/np.sum(np.abs(Q)))
    # with open("Q_error.txt", "w") as f: # Print the transition matrix from power iteration
    #     for i in range(minigrid_mdp.n_states):
    #         f.write("Q[{}] error: {}\n".format(minigrid_mdp.states[i], Q[i] - Q_est[i]))
    # #plot_episode_length(lengths, minigrid_mdp.shortest_path_length(opt_policy), plot_batch=True)
    # q_averaged_throughputs = plot_episode_throughput(throughputs, minigrid_mdp.shortest_path_length(opt_policy), plot_batch=True)

    # LMDP
    minigrid = Minigrid(grid_size=grid_size, walls=walls)
    minigrid_plots = Minigrid_LMDP_Plots(minigrid)

    # Power Iteration LMDP
    lmbda = 1
    Z, n_steps = minigrid.power_iteration(lmbda = lmbda, epsilon=epsilon)
    print("Power iteration took: ", n_steps, " steps before converging with epsilon:", epsilon)
    print("\n\n")
    #minigrid_plots.show_Z(Z, print_Z=True, plot_Z = False, file = "Z_function_power_iteration.txt")
    PU = minigrid.compute_Pu(Z)
    with open("PU_power_iteration.txt", "w") as f: # Print the transition matrix from power iteration
        for i in minigrid.S:
            for j in PU[i].indices:
                    if PU[i,j] != 0: f.write("Pu[{}][{}]: {}\n".format(minigrid.states[i], minigrid.states[j], PU[i,j]))
    # minigrid_plots.plot_sample_path(PU, path = 'plots\LMDP_power_iteration_path.gif')
    V = minigrid.Z_to_V(Z)
    # with open("value_function_power_iteration.txt", "w") as f: # Print the transition matrix from power iteration
    #     for i in range(minigrid.n_states):
    #         f.write("V[{}]: {}\n".format(minigrid.states[i], V[i]))

    # Z-Learning 
    # print("Z-learning training...")
    # zlearning = ZLearning(minigrid, lmbda=1, learning_rate=0.25, learning_rate_decay=0.99999, learning_rate_min=0.0005)
    # start_time = time.time()
    # Z_est, z_lengths, z_throughputs = Zlearning_training(zlearning, n_steps=int(8e5))
    # print("--- %s minutes and %s seconds ---" % (int((time.time() - start_time)/60), int((time.time() - start_time) % 60)))
    # print("Mean Absolute Error: ", np.sum(np.abs(PU-zlearning.Pu))/np.sum(np.abs(PU)) # Normalized error
    # print("Mean Absolute Z Error: ", np.sum(np.abs(Z-Z_est[-1]))/np.sum(np.abs(Z))) # Normalized error
    # minigrid_plots.show_Z(Z_est[-1], print_Z=True, plot_Z = False, file = "Z_function_zlearning.txt")
    # #minigrid_plots.plot_sample_path(zlearning.Pu, path = 'plots\LMDP_Z_learning_path.gif')
    # with open("PU_zlearning.txt", "w") as f: # Print the transition matrix from power iteration
    #     for i in minigrid.S:
    #         for j in zlearning.Pu[i].indices:
    #                 if zlearning.Pu[i,j] != 0: f.write("Pu[{}][{}]: {}\n".format(minigrid.states[i], minigrid.states[j], zlearning.Pu[i,j]))
    # z_averaged_throughputs = plot_episode_throughput(z_throughputs, minigrid_mdp.shortest_path_length(opt_policy), plot_batch=True)

    # Embedded MDP
    mdp_minigrid = minigrid.embedding_to_MDP()
    minigrid_mdp_embedded_plots = Minigrid_MDP_Plots(mdp_minigrid)
    start_time = time.time()
    Q2, opt_policy2, n_steps = mdp_minigrid.value_iteration(epsilon, gamma)
    print("Value iteration took: ", n_steps, " steps before converging with epsilon:", epsilon)
    print("--- %s minutes and %s seconds ---" % (int((time.time() - start_time)/60), int((time.time() - start_time) % 60)))
    # # plot_greedy_policy(greedy_policy, mdp_minigrid, print_policy=True)
    #minigrid_mdp_embedded_plots.plot_value_function(Q2, print_values=True, file = "value_function_embedded.txt")
    # #minigrid_mdp_embedded_plots.plot_path(opt_policy2, path = 'plots\MDP_embedded_value_iteration_path.gif')
    print("Total embedding error: ", np.sum(np.abs(V-Q2.max(axis=1))))

    # Q-learning Embedded MDP
    # print("Q-learning training...")
    # qlearning = QLearning(mdp_minigrid, gamma=gamma, learning_rate=0.25, learning_rate_decay=0.99999, learning_rate_min=0.0005, epsilon=1, epsilon_decay=0.9995, epsilon_min = 0)
    # start_time = time.time()
    # Q_est, est_policy, lengths, throughputs = Qlearning_training(qlearning, n_steps=int(8e5))
    # print("--- %s minutes and %s seconds ---" % (int((time.time() - start_time)/60), int((time.time() - start_time) % 60)))
    # print(opt_policy2 - est_policy)
    # minigrid_mdp_embedded_plots.plot_value_function(Q_est, print_values=True, file = "QLearning_embedded_value_function.txt")
    # print("Total Absolute Error: ", np.sum(np.abs(Q2-Q_est))/np.sum(np.abs(Q2)))
    # #plot_greedy_policy(est_policy, minigrid_mdp, print_policy=True, estimated=True)
    # #minigrid_mdp_embedded_plots.plot_path(est_policy, path = 'plots\MDP_QLearning_path.gif')
    # #plot_episode_length(lengths, minigrid_mdp.shortest_path_length(opt_policy), plot_batch=True)
    # q_averaged_throughputs2 = plot_episode_throughput(throughputs, minigrid_mdp.shortest_path_length(opt_policy), plot_batch=True)

    epsilons = [1, 0.9]#, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    q_throughputs = []
    q_names = []
    q_errors = []
    q_policy_differences = []

    for epsilon in epsilons:
        qlearning = QLearning(mdp_minigrid, gamma=gamma, learning_rate=0.25, learning_rate_decay=0.99999, learning_rate_min=0.0005, epsilon=epsilon, epsilon_decay=0.9995, epsilon_min = 0)
        Q_est, est_policy, _, throughputs = Qlearning_training(qlearning, n_steps=int(8e5))
        q_averaged_throughputs = plot_episode_throughput(throughputs, minigrid_mdp.shortest_path_length(opt_policy), plot_batch=True)
        q_throughputs.append(q_averaged_throughputs)
        q_names.append('Epsilon: ' + str(epsilon))
        q_errors.append(np.sum(np.abs(Q2-Q_est))/np.sum(np.abs(Q2)))
        q_policy_differences.append(np.sum(opt_policy2 != est_policy))


    #compare_throughputs(q_averaged_throughputs, q_averaged_throughputs2, minigrid.grid_size, name1 = 'Q Learning', name2 = 'Q Learning Embedded')
    plot_throughput(q_throughputs, minigrid.grid_size, names = q_names, save_path='plots\Q_epsilon')
    #compare_throughputs(throughputs = q_throughputs, grid_size = minigrid.grid_size, names = q_names, save_path='plots\Q_epsilon_old')
    #plot_value_per_hyperparameter(q_errors, epsilons, title = 'Value Function Approximation Error by Exploration', xlabel = 'Epsilon', ylabel = 'Approximation Error', save_path = 'plots\Q_epsilon_error')
    #plot_value_per_hyperparameter(q_policy_differences, epsilons, title = 'Policy Differences by Exploration', xlabel = 'Epsilon', ylabel = 'Policy Differences', save_path = 'plots\Q_epsilon_policy_difference')

    # with open("PU_zlearning", "w") as f: # Print the transition matrix from Z-learning
    #     for i in minigrid.S:
    #         for j in zlearning.Pu[i].indices:
    #                 if zlearning.Pu[i,j] != 0: f.write("Pu[{}][{}]: {}\n".format(minigrid.states[i], minigrid.states[j], zlearning.Pu[i,j]))

    # for i in minigrid.S: # Print the transition matrix differences between Z-learning and Power Iteration
    #        for j in (PU[i].indices):
    #             if np.abs(PU[i,j] -zlearning.Pu[i,j]) >= 0.05: 
    #                 print("Pu[", minigrid.states[i], "][", minigrid.states[j], "]: ", PU[i,j])
    #                 print("ZLearning Pu[", minigrid.states[i], "][", minigrid.states[j], "]: ", zlearning.Pu[i,j])