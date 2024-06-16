from environments.minigrids import Minigrid_LMDP, Minigrid_MDP
from environments.blackjack import Black_Jack_MDP
import numpy as np
import time
from utils.plot import Plotter, Minigrid_MDP_Plotter
from algs.zlearning import ZLearning, Zlearning_training
from algs.qlearning import QLearning, Qlearning_training
from utils.lmdp_plot import Minigrid_LMDP_Plotter
from scipy.sparse import csr_matrix
import cProfile


import gymnasium as gym
import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    grid_size = 15
    walls = [(14,1), (1,8), (5, 5), (12, 5), (8, 7), (2,5), (3,5), (4,5), (6,5), (7,5), (8,5), (9,5), (10,5), (11,5), (13,5), (15,9)]
    lavas = []#(7,1), (1,4), (1,14)]
    objects = {"walls": walls, "lavas": lavas}

    simple15_map = [
        "########",
        "#A     #",
        "#      #",
        "#  W   #",
        "#      #",
        "#      #",
        "#L    G#",
        "########"
    ]

    hill15_map = [
        "#################",
        "#A              #",
        "# WLLLLLLLLLLLL #",
        "# WWWWWWWWWWWWL #",
        "# W   W   W  WL #",
        "# W W W W W  WL #",
        "# W W W W W  WL #",
        "# W W W W W  WL #",
        "# W W W W W  WL #",
        "# W W W W W  WL #",
        "# W W W W W  WL #",
        "# W W W W W  WL #",
        "# W W W W W  WL #",
        "# W W W W W  WL #",
        "# W W W W W  WL #",
        "#   W   W      G#",
        "#################"
    ]

    maze13_map = [
        "###############",
        "#A    W       #",
        "# W W W W W W #",
        "# W   W W W W #",
        "# W W W W W W #",
        "# W W W W   W #",
        "# W W W WWWWW #",
        "# W     W     #",
        "# W WWWWW WWW #",
        "# W W   W   W #",
        "# WWW W W W W #",
        "#     W W W W #",
        "# WWWWW W W WW#",
        "#     W W    G#",
        "###############"
    ]

    maze20_map = [
        "####################",
        "#A    W     W     G#",
        "#W W  W W W W W W  #",
        "#W W      W W W W  #",
        "#W W W W WWW  WWWWW#",
        "#W   W W   W   W   #",
        "#WWWWWWWWW WWW WW W#",
        "#  W   W     W W   #",
        "#  W W WWW W W W  W#",
        "#  W W   W W W     #",
        "#  W WW    WWWWW WW#",
        "#    W       W W W #",
        "#WWWWW WWWWW W W W #",
        "#W     W     W   W #",
        "#W WWWWW WWWWW WWW #",
        "#W     W     W     #",
        "#WWWWWWW WWWWWWW WW#",
        "#        W        W#",
        "#W W W W   W W W   #",
        "####################"
    ]

    rooms_map = [
        "######################",
        "#             W     G#",
        "#      W      W      #",
        "#      W      W      #",
        "#      W      W      #",
        "#      W      W      #",
        "#      W      W      #",
        "#WWWW WW WWWWWWW WWWWW#",
        "#      W      W      #",
        "#      W      W      #",
        "#      W      W      #",
        "#      W  A   W      #",
        "#      W      W      #",
        "#      W      W      #",
        "#W WWWWWWWWWWWWWWWWW W#",
        "#      W             #",
        "#      W      W      #",
        "#      W      W      #",
        "#      W      W      #",
        "#             W      #",
        "#      W      W      #",
        "######################"
    ]

    grid_map = hill15_map
    grid_size = len(grid_map)-2 if grid_map is not None else grid_size

    gamma = 1
    epsilon = 1e-10
    n_iters = int(8e5)
    lmbda = 1

    # MDP
    minigrid_mdp = Minigrid_MDP(grid_size=grid_size, objects=objects, map=grid_map, gamma=gamma)
    minigrid_mdp_plots = Minigrid_MDP_Plotter(minigrid_mdp)
    #minigrid_mdp.render()
    minigrid_lmdp = Minigrid_LMDP(grid_size=grid_size, objects=objects, map=grid_map, lmbda=lmbda)
    minigrid_mdp_embedded,_ = minigrid_lmdp.embedding_to_MDP()

    # Q, opt_policy, n_steps = minigrid_mdp.value_iteration(epsilon=1e-10, gamma=gamma)
    # V = np.max(Q, axis=1)

    # qlearning = QLearning(minigrid_mdp, gamma=gamma, epsilon=1, epsilon_decay=0.9995, epsilon_min = 0, c = 200)
    # Q_est, est_policy, q_throughputs, q_rewards= Qlearning_training(qlearning, n_steps=n_iters)
    # V = np.max(Q_est, axis=1)
    # print("Total Square Error: ", np.sum(np.square(Q-Q_est)))
    # qlearning = QLearning(minigrid_mdp_embedded, gamma=gamma, epsilon=1, epsilon_decay=0.9, epsilon_min = 0, c = 200)
    # Q_est, est_policy, q_throughputs2, q_rewards2= Qlearning_training(qlearning, n_steps=n_iters)
    # V2 = np.max(Q_est, axis=1)
    # print("Total Square Error: ", np.sum(np.square(Q-Q_est)))
    # qlearning = QLearning(minigrid_mdp, gamma=gamma, epsilon=1, epsilon_decay=0.5, epsilon_min = 0, c = 200)
    # Q_est, est_policy, q_throughputs3, q_rewards3= Qlearning_training(qlearning, n_steps=n_iters)
    # V3 = np.max(Q_est, axis=1)
    # print("Total Square Error: ", np.sum(np.square(Q-Q_est)))
    
    # minigrid_mdp_plots.plot_values_vs_indices(range(1, len(q_rewards)+1), [q_rewards, q_rewards2], names = ["Q-G", "Q-GM"], save_path='plots\comparison_c_throughputs', title="Simple 15x15 Grid", scale='symlog')
    # minigrid_mdp_plots.plot_values_vs_indices(range(1, len(q_throughputs)+1), [q_throughputs, q_throughputs2], names = ["Q-G", "Q-GM"], save_path='plots\comparison_c', title="Simple 15x15 Grid")
    # minigrid_mdp_plots.plot_throughput([q_throughputs, q_throughputs2, q_throughputs3], grid_size, names = ["Q-L", "Q-LM", "Q-LP"], smooth_window=10000, save_path='plots\comparison_c')
    # minigrid_mdp_plots.plot_rewards([q_rewards, q_rewards2, q_rewards3], grid_size, names = ["Q-L", "Q-LM", "Q-LP"], smooth_window=10000, save_path='plots\comparison_c_rewards')
    # minigrid_mdp_plots.plot_grids(minigrid_mdp, grid_size, [V, V2, V3], names = ["Q-L", "Q-LM", "Q-LP"])

    throughputs = []
    rewards = []

    for i in range(5):
        qlearning = QLearning(minigrid_mdp_embedded, gamma=gamma, epsilon=1, epsilon_decay=0.9995, epsilon_min = 0, c = 200)
        Q_est, est_policy, q_throughputs, q_rewards= Qlearning_training(qlearning, n_steps=n_iters)
        throughputs.append(q_throughputs)
        rewards.append(q_rewards)
    
    q_throughputs = np.mean(throughputs, axis=0)
    q_rewards = np.mean(rewards, axis=0)
    for q_reward in rewards:
        minigrid_mdp_plots.plot_episode_reward(q_reward)
    V_q = np.max(Q_est, axis=1)

    minigrid_mdp_plots.plot_episode_throughput(q_throughputs)
    minigrid_mdp_plots.plot_episode_reward(q_rewards)

    zlearning = ZLearning(minigrid_lmdp, lmbda=1, c=10000)
    Z_est, z_throughputs, z_rewards = Zlearning_training(zlearning, n_steps=n_iters)
    V_z = zlearning.Z_to_V(Z_est)
    minigrid_mdp_plots.plot_episode_reward(z_rewards)
    minigrid_mdp_plots.plot_episode_throughput(z_throughputs)
    zlearning_naive = ZLearning(minigrid_lmdp, lmbda=1, c=10000, naive=True)
    Z_est_naive, z_throughputs_naive, z_rewards_naive = Zlearning_training(zlearning_naive, n_steps=n_iters)
    V_z_naive = zlearning.Z_to_V(Z_est_naive)
    minigrid_mdp_plots.plot_episode_throughput(z_throughputs_naive)
    minigrid_mdp_plots.plot_episode_reward(z_rewards_naive)
    
    minigrid_mdp_plots.plot_rewards([q_rewards, z_rewards, z_rewards_naive], grid_size, names = ["Q-L", "Z-L", "NZ-L"], smooth_window=10000, save_path='plots\comparison_c_throughputs')
    minigrid_mdp_plots.plot_throughput([q_throughputs, z_throughputs, z_throughputs_naive], grid_size, names = ["Q-L", "Z-L", "NZ-L"], smooth_window=10000, save_path='plots\comparison_c')
    minigrid_mdp_plots.plot_grids(minigrid_mdp, grid_size, [V_q, V_z, V_z_naive], names = ["Q-L", "Z-L", "NZ-L"])
    
    
    
    
    # Q-learning
    # print("Q-learning training...")
    # # qlearning = QLearning(black_jack, gamma=gamma, epsilon=1, epsilon_decay=0.995, epsilon_min = 0, c = 200, reset_randomness = 1)
    # # Q_est, est_policy, lengths, throughputs = Qlearning_training(qlearning, n_steps=n_iters)
    # # #plot_greedy_policy(est_policy, minigrid_mdp, print_policy=True, estimated=True)
    # minigrid_mdp_plots.plot_value_function(Q_est, print_values=True, file = "QLearning_value_function.txt")
    # #minigrid_mdp_plots.plot_path(est_policy, path = 'videos\MDP_QLearning_path.gif')
    # # # print(opt_policy - est_policy)
    # print("Total Square Error: ", np.sum(np.square(Q-Q_est)))
    # # with open("Q_error.txt", "w") as f: # Print the transition matrix from power iteration
    # #     for i in range(minigrid_mdp.n_states):
    # #         f.write("Q[{}] error: {}\n".format(minigrid_mdp.states[i], Q[i] - Q_est[i]))
    # # print("Policy Differences: ", np.sum(np.abs(opt_policy-est_policy)))
    # # with open("policy_error.txt", "w") as f: # Print the transition matrix from power iteration
    # #     for i in range(minigrid_mdp.n_states):
    # #         f.write("Pi[{}]: {}\n".format(minigrid_mdp.states[i], opt_policy[i] - est_policy[i]))
    # with open("nsa.txt", "w") as f: # Print the transition matrix from power iteration
    #     for i in range(minigrid_mdp.n_states):
    #         f.write("Nsa[{}]: {}\n".format(minigrid_mdp.states[i], qlearning.Nsa[i]))
    # q_averaged_throughputs = minigrid_mdp_plots.plot_episode_throughput(throughputs)#, minigrid_mdp.shortest_path_length())

    # # LMDP
    # minigrid = Minigrid_LMDP(grid_size=grid_size, objects=objects, map=grid_map, lmbda=lmbda)
    # minigrid_plots = Minigrid_LMDP_Plotter(minigrid)

    # # Power Iteration 
    # Z, n_steps = minigrid.power_iteration(lmbda = lmbda, epsilon=epsilon)
    # # #print("Power iteration took: ", n_steps, " steps before converging with epsilon:", epsilon)
    # # #minigrid_plots.show_Z(Z, print_Z=True, plot_Z = False, file = "Z_function_power_iteration.txt")
    # PU = minigrid.compute_Pu(Z)
    # with open("PU_power_iteration.txt", "w") as f: # Print the transition matrix from power iteration
    #     for i in range(minigrid.n_nonterminal_states):
    #         for j in PU[i].indices:
    #                 if PU[i,j] != 0: f.write("Pu[{}][{}]: {}\n".format(minigrid.states[i], minigrid.states[j], PU[i,j]))
    # #minigrid_plots.plot_sample_path(PU, path = 'videos\LMDP_power_iteration_path.gif')
    # #V = minigrid.Z_to_V(Z)
    # # # with open("value_function_power_iteration.txt", "w") as f: # Print the transition matrix from power iteration
    # # #     for i in range(minigrid.n_states):
    # # #         f.write("V[{}]: {}\n".format(minigrid.states[i], V[i]))
    

    # # Z-Learning 
    # print("Z-learning training...")
    # zlearning = ZLearning(minigrid, lmbda=1, c = 200)
    # start_time = time.time()
    # Z_est, z_lengths, z_throughputs = Zlearning_training(zlearning, n_steps=n_iters)
    # print("--- %s minutes and %s seconds ---" % (int((time.time() - start_time)/60), int((time.time() - start_time) % 60)))
    # z_averaged_throughputs = minigrid_mdp_plots.plot_episode_throughput(z_throughputs, minigrid_mdp.shortest_path_length())
    # with open("PU_Z_learning.txt", "w") as f: # Print the transition matrix from power iteration
    #     for i in range(minigrid.n_nonterminal_states):
    #         for j in zlearning.Pu[i].indices:
    #                 if zlearning.Pu[i,j] != 0: f.write("Pu[{}][{}]: {}\n".format(minigrid.states[i], minigrid.states[j], zlearning.Pu[i,j])) 
    # print("Mean Absolute Error Pu: ", np.sum(np.abs(PU-zlearning.Pu))) 
    # # # print("Mean Absolute Z Error: ", np.sum(np.abs(Z-Z_est[-1]))/np.sum(np.abs(Z))) # Normalized error
    # # # minigrid_plots.show_Z(Z_est[-1], print_Z=True, plot_Z = False, file = "Z_function_zlearning.txt")
    # with open("Pu_error.txt", "w") as f: # Print the transition matrix from power iteration
    #     for i in range(minigrid.n_nonterminal_states):
    #         for j in zlearning.Pu[i].indices:
    #             f.write("Pu[{}][{}] error: {}\n".format(minigrid.states[i], minigrid.states[j], PU[i,j]- zlearning.Pu[i,j]))
    # minigrid_plots.plot_sample_path(PU, path = 'videos\LMDP_ZLearning_path.gif')
    # minigrid_mdp_plots.plot_throughput([throughputs, z_throughputs], minigrid_mdp.grid_size, ["Q-Learning", "Z-Learning"])

    # #Use prophiler for Zlearning_trainnig
    # cProfile.run('Zlearning_training(zlearning, n_steps=int(5e5))', sort='tottime')
