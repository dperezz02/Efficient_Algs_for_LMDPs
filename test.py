from frameworks.mdp import Minigrid_MDP
from frameworks.lmdp import Minigrid
import numpy as np
import time
from utils.plot import Plotter, Minigrid_MDP_Plotter
from algs.zlearning import ZLearning, Zlearning_training
from algs.qlearning import QLearning, Qlearning_training
from utils.lmdp_plot import Minigrid_LMDP_Plotter
from scipy.sparse import csr_matrix
import cProfile

if __name__ == "__main__":
    
    grid_size = 15
    walls = [(14,1), (1,8), (5, 5), (12, 5), (8, 7), (2,5), (3,5), (4,5), (6,5), (7,5), (8,5), (9,5), (10,5), (11,5), (13,5), (15,9)]
    lavas = [(7,1), (1,4), (14,14)]
    objects = {"walls": walls, "lavas": lavas}


    # MDP
    minigrid_mdp = Minigrid_MDP(grid_size=grid_size, objects=objects)
    minigrid_mdp_plots = Minigrid_MDP_Plotter(minigrid_mdp)
    #minigrid_mdp.render()

    gamma = 1
    epsilon = 1e-10
    n_iters = int(8e5)
    lmbda = 1

    Q, opt_policy, n_steps = minigrid_mdp.value_iteration(epsilon=1e-10, gamma=gamma)
    # minigrid_mdp_plots.plot_value_function(Q, print_values=True, file = "value_function.txt")
    # minigrid_mdp_plots.plot_path(opt_policy, path = 'videos\MDP_value_iteration_path.gif')


    # Q-learning
    print("Q-learning training...")
    qlearning = QLearning(minigrid_mdp, gamma=gamma, epsilon=1, epsilon_decay=0.9995, epsilon_min = 0, c = 200, reset_randomness = 0)
    Q_est, est_policy, lengths, throughputs = Qlearning_training(qlearning, n_steps=n_iters)
    # #plot_greedy_policy(est_policy, minigrid_mdp, print_policy=True, estimated=True)
    # minigrid_mdp_plots.plot_value_function(Q_est, print_values=True, file = "QLearning_value_function.txt")
    # minigrid_mdp_plots.plot_path(est_policy, path = 'videos\MDP_QLearning_path.gif')
    # # print(opt_policy - est_policy)
    # print("Total Square Error: ", np.sum(np.square(Q-Q_est)))
    # with open("Q_error.txt", "w") as f: # Print the transition matrix from power iteration
    #     for i in range(minigrid_mdp.n_states):
    #         f.write("Q[{}] error: {}\n".format(minigrid_mdp.states[i], Q[i] - Q_est[i]))
    # print("Policy Differences: ", np.sum(np.abs(opt_policy-est_policy)))
    # with open("policy_error.txt", "w") as f: # Print the transition matrix from power iteration
    #     for i in range(minigrid_mdp.n_states):
    #         f.write("Pi[{}]: {}\n".format(minigrid_mdp.states[i], opt_policy[i] - est_policy[i]))
    # with open("nsa.txt", "w") as f: # Print the transition matrix from power iteration
    #     for i in range(minigrid_mdp.n_states):
    #         f.write("Nsa[{}]: {}\n".format(minigrid_mdp.states[i], qlearning.Nsa[i]))
    q_averaged_throughputs = minigrid_mdp_plots.plot_episode_throughput(throughputs,minigrid_mdp.shortest_path_length())

    # LMDP
    minigrid = Minigrid(grid_size=grid_size, objects=objects)
    minigrid_plots = Minigrid_LMDP_Plotter(minigrid)

    # Power Iteration 
    Z, n_steps = minigrid.power_iteration(lmbda = lmbda, epsilon=epsilon)
    # #print("Power iteration took: ", n_steps, " steps before converging with epsilon:", epsilon)
    # #minigrid_plots.show_Z(Z, print_Z=True, plot_Z = False, file = "Z_function_power_iteration.txt")
    PU = minigrid.compute_Pu(Z)
    with open("PU_power_iteration.txt", "w") as f: # Print the transition matrix from power iteration
        for i in range(minigrid.n_nonterminal_states):
            for j in PU[i].indices:
                    if PU[i,j] != 0: f.write("Pu[{}][{}]: {}\n".format(minigrid.states[i], minigrid.states[j], PU[i,j]))
    minigrid_plots.plot_sample_path(PU, path = 'videos\LMDP_power_iteration_path.gif')
    #V = minigrid.Z_to_V(Z)
    # # with open("value_function_power_iteration.txt", "w") as f: # Print the transition matrix from power iteration
    # #     for i in range(minigrid.n_states):
    # #         f.write("V[{}]: {}\n".format(minigrid.states[i], V[i]))
    

    # Z-Learning 
    print("Z-learning training...")
    zlearning = ZLearning(minigrid, lmbda=1, c = 200)
    start_time = time.time()
    Z_est, z_lengths, z_throughputs = Zlearning_training(zlearning, n_steps=n_iters)
    print("--- %s minutes and %s seconds ---" % (int((time.time() - start_time)/60), int((time.time() - start_time) % 60)))
    z_averaged_throughputs = minigrid_mdp_plots.plot_episode_throughput(z_throughputs, minigrid_mdp.shortest_path_length())
    with open("PU_Z_learning.txt", "w") as f: # Print the transition matrix from power iteration
        for i in range(minigrid.n_nonterminal_states):
            for j in zlearning.Pu[i].indices:
                    if zlearning.Pu[i,j] != 0: f.write("Pu[{}][{}]: {}\n".format(minigrid.states[i], minigrid.states[j], zlearning.Pu[i,j]))
    print("Mean Absolute Error Z : ", np.sum(np.square(Z-Z_est))) 
    print("Mean Absolute Error Pu: ", np.sum(np.square(PU-zlearning.Pu))) 
    # # print("Mean Absolute Z Error: ", np.sum(np.abs(Z-Z_est[-1]))/np.sum(np.abs(Z))) # Normalized error
    # # minigrid_plots.show_Z(Z_est[-1], print_Z=True, plot_Z = False, file = "Z_function_zlearning.txt")
    minigrid_mdp_plots.plot_throughput([throughputs, z_throughputs], minigrid_mdp.grid_size, ["Q-Learning", "Z-Learning"])
    minigrid_plots.plot_sample_path(PU, path = 'videos\LMDP_ZLearning_path.gif')
