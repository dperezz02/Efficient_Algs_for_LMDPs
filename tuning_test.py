from environments.MDP import Minigrid_MDP
from environments.LMDP import Minigrid
import numpy as np
import time
from utils.plot import Plotter, Minigrid_MDP_Plotter
from models.zlearning import ZLearning, Zlearning_training
from models.qlearning import QLearning, Qlearning_training
from utils.lmdp_plot import Minigrid_LMDP_Plotter
from scipy.sparse import csr_matrix

if __name__ == "__main__":

    grid_size = 15
    walls = [(14,1), (1,8), (5, 5), (12, 5), (8, 7), (2,5), (3,5), (4,5), (6,5), (7,5), (8,5), (9,5), (10,5), (11,5), (13,5), (15,9)]
    
    # MDP
    minigrid_mdp = Minigrid_MDP(grid_size=grid_size, walls = walls)
    minigrid_mdp_plots = Minigrid_MDP_Plotter(minigrid_mdp)

    gamma = 1
    epsilon = 1e-10
    n_iters = int(4e5)

    # Value Iteration MDP
    Q, opt_policy, n_steps = minigrid_mdp.value_iteration(epsilon, gamma)
    
    # LMDP
    minigrid = Minigrid(grid_size=grid_size, walls=walls)
    minigrid_plots = Minigrid_LMDP_Plotter(minigrid)

    # Power Iteration LMDP
    lmbda = 1
    Z, n_steps = minigrid.power_iteration(lmbda = lmbda, epsilon=epsilon)
    print("Power iteration took: ", n_steps, " steps before converging with epsilon:", epsilon)
    print("\n\n")
    PU = minigrid.compute_Pu(Z)
    V = minigrid.Z_to_V(Z)

    # Embedded MDP
    mdp_minigrid = minigrid.embedding_to_MDP()
    minigrid_mdp_embedded_plots = Minigrid_MDP_Plotter(mdp_minigrid)
    start_time = time.time()
    Q2, opt_policy2, n_steps = mdp_minigrid.value_iteration(epsilon, gamma)
    print("Value iteration took: ", n_steps, " steps before converging with epsilon:", epsilon)
    print("--- %s minutes and %s seconds ---" % (int((time.time() - start_time)/60), int((time.time() - start_time) % 60)))

    # epsilons = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    # q_throughputs = []
    # q_names = []
    # q_errors = []
    # q_policy_differences = []

    # for epsilon in epsilons:
    #     qlearning = QLearning(mdp_minigrid, gamma=gamma, learning_rate=0.25, learning_rate_decay=0.99995, learning_rate_min=0.0005, epsilon=epsilon, epsilon_decay=0.9995, epsilon_min = 0)
    #     Q_est, est_policy, _, throughputs = Qlearning_training(qlearning, n_steps=n_iters)
    #     q_throughputs.append(throughputs)
    #     q_names.append('Epsilon: ' + str(epsilon))
    #     q_errors.append(np.sum(np.abs(Q2-Q_est))/np.sum(np.abs(Q2)))
    #     q_policy_differences.append(np.sum(opt_policy2 != est_policy))


    # minigrid_mdp_embedded_plots.plot_throughput(q_throughputs, minigrid.grid_size, names = q_names, save_path='plots\Q_epsilon2')
    #minigrid_mdp_embedded_plots.plot_value_per_hyperparameter(q_errors, epsilons, title = 'Value Function Approximation Error by Exploration', xlabel = 'Epsilon', ylabel = 'Approximation Error', save_path = 'plots\Q_epsilon_error2')
    #minigrid_mdp_embedded_plots.plot_value_per_hyperparameter(q_policy_differences, epsilons, title = 'Policy Differences by Exploration', xlabel = 'Epsilon', ylabel = 'Policy Differences', save_path = 'plots\Q_epsilon_policy_difference2')

    # epsilon_decays = [0.9995, 0.999, 0.995, 0.99]
    # q_throughputs = []
    # q_names = []
    # q_errors = []
    # q_policy_differences = []
    
    # for epsilon_decay in epsilon_decays:
    #     qlearning = QLearning(mdp_minigrid, gamma=gamma, learning_rate=0.25, learning_rate_decay=0.99999, learning_rate_min=0.0005, epsilon=1, epsilon_decay=epsilon_decay, epsilon_min = 0)
    #     Q_est, est_policy, _, throughputs = Qlearning_training(qlearning, n_steps=n_iters)
    #     # Get the number of iterations needed to converge according to throughputs
    #     q_throughputs.append(throughputs)
    #     q_names.append('Epsilon Decay: ' + str(epsilon_decay))
    #     q_errors.append(np.sum(np.abs(Q2-Q_est))/np.sum(np.abs(Q2)))
    #     q_policy_differences.append(np.sum(opt_policy2 != est_policy))



    # minigrid_mdp_embedded_plots.plot_throughput(q_throughputs, minigrid.grid_size, names = q_names, save_path='plots\Q_epsilon_decays2')
    # minigrid_mdp_embedded_plots.plot_value_per_hyperparameter(q_errors, epsilon_decays, title = 'Value Function Approximation Error by Exploration Decay', xlabel = 'Epsilon Decay', ylabel = 'Approximation Error', save_path = 'plots\Q_epsilon_decay_error2')
    # minigrid_mdp_embedded_plots.plot_value_per_hyperparameter(q_policy_differences, epsilon_decays, title = 'Policy Differences by Exploration Decay', xlabel = 'Epsilon Decay', ylabel = 'Policy Differences', save_path = 'plots\Q_epsilon_decay_policy_difference2')


    #learning_rate_decays = [1, 0.9999, 0.995, 0.99]
    #learning_rates = [1, 0.75, 0.5, 0.25, 0.1, 0.05, 0.01]
    #gammas = [1, 0.99, 0.95, 0.9, 0]
    lambdas = [1, 0.1, 0.01, 0.001]
    q_throughputs = []
    q_names = []
    q_errors = []
    q_policies = []
    
    for lmbda in lambdas:
        #qlearning = QLearning(mdp_minigrid, gamma=gamma, learning_rate=0.25, learning_rate_decay=0.9999, learning_rate_min=0.0005, epsilon=1, epsilon_decay=0.9995, epsilon_min = 0)
        zlearning = ZLearning(minigrid, lmbda=lmbda, learning_rate=0.25, learning_rate_decay=0.9999, learning_rate_min=0.0005)
        #Q_est, est_policy, _, throughputs = Qlearning_training(qlearning, n_steps=n_iters)
        _, _, z_throughputs = Zlearning_training(zlearning, n_steps=n_iters)
        Pu = zlearning.Pu
        q_throughputs.append(z_throughputs)
        q_names.append('Lambda: ' + str(lmbda))
        #q_errors.append(np.sum(np.abs(Q2-Q_est))/np.sum(np.abs(Q2)))
        #q_policies.append(1-np.sum(opt_policy2 != est_policy)/mdp_minigrid.n_states)
        q_policies.append(1-np.sum(np.abs(PU-zlearning.Pu))/np.sum(PU))


    minigrid_mdp_embedded_plots.plot_throughput(q_throughputs, minigrid.grid_size, names = q_names, save_path='plots\Z_lmbda')
    #minigrid_mdp_embedded_plots.plot_value_per_hyperparameter(q_errors, gammas, title = 'Value Function Approximation Error by Discount Rate', xlabel = 'Discount Rate', ylabel = 'Approximation Error', save_path = 'plots\Q_discount_rate_error')
    minigrid_mdp_embedded_plots.plot_value_per_hyperparameter(q_policies, lambdas, title = 'PU Approximation by Temperature Parameter', xlabel = 'Lambda', ylabel = 'PU Approximation', save_path = 'plots\Z_lambda_pu')

    #TODO: gamma lambda