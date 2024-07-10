from environments.minigrids import Minigrid_LMDP, Minigrid_MDP
from environments.simplegrid import SimpleGrid_LMDP, SimpleGrid_MDP
import numpy as np
import time
from utils.plot import Plotter, Minigrid_MDP_Plotter
from algs.zlearning import ZLearning, Zlearning_training
from algs.qlearning import QLearning, Qlearning_training
from utils.lmdp_plot import Minigrid_LMDP_Plotter
from scipy.sparse import csr_matrix
import random

if __name__ == "__main__":

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

    def generate_random_walls_and_lavas(grid_size, wall_percentage, lava_percentage):
        walls = set()
        lavas = set()

        # Ensure the start and goal positions are not filled
        forbidden_positions = {(1, 1), (grid_size, grid_size)}

        # Calculate the number of walls and lavas based on the percentage
        num_states = grid_size * grid_size
        num_walls = int(num_states * wall_percentage / 100)
        num_lavas = int(num_states * lava_percentage / 100)

        # Randomly generate wall positions
        while len(walls) < num_walls:
            x = random.randint(1, grid_size)
            y = random.randint(1, grid_size)
            if (x, y) not in forbidden_positions and (x, y) not in lavas:
                walls.add((x, y))

        # Randomly generate lava positions
        while len(lavas) < num_lavas:
            x = random.randint(1, grid_size)
            y = random.randint(1, grid_size)
            if (x, y) not in forbidden_positions and (x, y) not in walls:
                lavas.add((x, y))

        return {"walls": list(walls), "lavas": list(lavas)}

    grid_size = 10
    wall_percentage = 10
    lava_percentage = 10
    objects = generate_random_walls_and_lavas(grid_size, wall_percentage, lava_percentage)
    objects = {"walls":[], "lavas":[]}

    grid_map = hill15_map
    grid_size = len(grid_map)-2 if grid_map is not None else grid_size


    # MDP
    simplegrid_mdp = SimpleGrid_MDP(grid_size)
    minigrid_mdp2 = Minigrid_MDP(grid_size, objects=objects, map = grid_map)
    #minigrid_mdp.render()
    
    #minigrid_lmdp, error = minigrid_mdp.embedding_to_LMDP()
    minigrid_lmdp = Minigrid_LMDP(grid_size, objects=objects, map = grid_map)
    minigrid_mdp, error = minigrid_lmdp.embedding_to_MDP()
    print(error)
    # minigrid_lmdp, error = minigrid_mdp.embedding_to_LMDP()
    # print(error)
    # minigrid_mdp, error = minigrid_lmdp.embedding_to_MDP()
    # print(error)
    minigrid_mdp_plots = Minigrid_MDP_Plotter(minigrid_mdp)
    
    
    Z, n_steps = minigrid_lmdp.power_iteration(lmbda = 1, epsilon=1e-10)
    V2 = minigrid_lmdp.Z_to_V(Z)

    minigrid_mdp_plots.plot_minigrid(minigrid_mdp, grid_size, V2)




    gamma = 1
    epsilon = 1e-10
    n_iters = int(2e5)
    lmbda = 1

    # Value Iteration MDP
    Q, opt_policy, n_steps = minigrid_mdp.value_iteration(epsilon, gamma)
    V = np.max(Q, axis=1) 

    qlearning = QLearning(minigrid_mdp, gamma=gamma, c=200, epsilon=1, epsilon_decay=0.999, epsilon_min = 0, reset_randomness=0.1)
    Q_est, V_error, est_policy, throughputs, rewards = Qlearning_training(qlearning, n_steps=n_iters, V=V)

    V_est = np.max(Q_est, axis=1)

    qlearning = QLearning(minigrid_mdp2, gamma=gamma, c=200, epsilon=1, epsilon_decay=0.999, epsilon_min = 0, reset_randomness=0.1)
    Q_est2, V_error2, est_policy2, throughputs2, rewards2 = Qlearning_training(qlearning, n_steps=n_iters, V=V)

    #minigrid_mdp_plots.plot_rewards_and_errors([V_error, V_error2], [rewards, rewards2], [r"$\epsilon$-Q-L", r"$\epsilon_{d}$-Q-L"], title="Minigrid  domain, grid size=15x15")
    

    zlearning = ZLearning(minigrid_lmdp, lmbda=lmbda, c=10000, reset_randomness=0.1)
    Z, V_error2, z_throughputs, z_rewards = Zlearning_training(zlearning, n_steps=n_iters, V=V2)
    V_est2 = minigrid_lmdp.Z_to_V(Z)

    print(np.mean(np.square(V_est2 - V2)))

    PU = minigrid_lmdp.compute_Pu(Z)

    with open("V_opt.txt", "w") as f:
        for i in range(minigrid_lmdp.n_nonterminal_states):
            f.write("V_Z[{}]: {}\n".format(minigrid_lmdp.states[i], V2[i]))

    with open("Pu.txt", "w") as f:
        for i in range(minigrid_lmdp.n_nonterminal_states):
            for j in PU[i].indices:
                f.write("Pu[{}, {}] -> {}\n".format(minigrid_lmdp.states[i], minigrid_lmdp.states[j], PU[i,j]))

    with open("V_z_true.txt", "w") as f:
        for i in range(minigrid_lmdp.n_nonterminal_states):
            f.write("V_Z[{}]: {}\n".format(minigrid_lmdp.states[i], V_est2[i]))

    with open("V_error.txt", "w") as f:
        for i in range(minigrid_lmdp.n_nonterminal_states):
            f.write("V_error[{}]: {}\n".format(minigrid_lmdp.states[i], np.mean(np.square(V_est2[i] - V2[i]))))

    # # with open("V_z.txt", "w") as f:
    # #     for i in range(1, grid_size+1):
    # #         for j in range(1, grid_size+1):
    # #             if (i,j,0) in minigrid_mdp.states:
    # #                 v = max(V_est2[minigrid_lmdp.state_to_index[(i,j,0)]], V_est2[minigrid_lmdp.state_to_index[(i,j,1)]], V_est2[minigrid_lmdp.state_to_index[(i,j,2)]], V_est2[minigrid_lmdp.state_to_index[(i,j,3)]] )
    # #                 f.write("V_z[{}]: {}\n".format((i,j), v))


    minigrid_mdp_plots.plot_rewards_and_errors([V_error, V_error2], [rewards2, z_rewards], ["Q-L", "Z-L"], title="Minigrid (hill cliff) domain, grid size=13x13")

    minigrid_mdp_plots.plot_grids_with_policies(env=minigrid_mdp, grid_size= grid_size, value_functions = [V, V_est, V_est2], names = ["Optimal Value Function", "Q-learning", "Z-learning"])



    # with open("V.txt", "w") as f: 
    #     for i in range(1, grid_size+1):
    #         for j in range(1, grid_size+1):
    #             if (i,j,0) in minigrid_mdp.states:
    #                 v = (V[minigrid_mdp.state_to_index[(i,j,0)]]+ V[minigrid_mdp.state_to_index[(i,j,1)]]+ V[minigrid_mdp.state_to_index[(i,j,2)]]+ V[minigrid_mdp.state_to_index[(i,j,3)]] )/4
    #                 f.write("V[{}]: {}\n".format((i,j), v))

    # with open("V_est.txt", "w") as f: 
    #     for i in range(1, grid_size+1):
    #         for j in range(1, grid_size+1):
    #             if (i,j,0) in minigrid_mdp.states:
    #                 v = (V_est[minigrid_mdp.state_to_index[(i,j,0)]]+ V_est[minigrid_mdp.state_to_index[(i,j,1)]]+ V_est[minigrid_mdp.state_to_index[(i,j,2)]]+ V_est[minigrid_mdp.state_to_index[(i,j,3)]] )/4
    #                 f.write("V[{}]: {}\n".format((i,j), v))

    # # # minigrid_mdp_plots.plot_rewards_and_errors([V_error, V_error+50],[rewards, rewards-500], ["0.999", "0.995"], title=r"Minigrid domain, grid size=10x10, $\epsilon=1$")

    # minigrid_mdp_plots.plot_grids_with_policies(env=minigrid_mdp, grid_size= grid_size, value_functions = [V, V_est])


    
    # LMDP
    # minigrid = Minigrid_LMDP(grid_size=grid_size)
    # minigrid_plots = Minigrid_LMDP_Plotter(minigrid)
    # minigrid = SimpleGrid_LMDP(grid_size)

    # walls = [(14,1), (1,8), (5, 5), (12, 5), (8, 7), (2,5), (3,5), (4,5), (6,5), (7,5), (8,5), (9,5), (10,5), (11,5), (13,5), (15,9)]
    # lavas = [(7,1), (1,4), (14,14)]

    # Z, n_steps = minigrid.power_iteration(lmbda = lmbda, epsilon=epsilon)
    # V = minigrid.Z_to_V(Z)
    # minigrid_plots.plot_grid(minigrid, grid_size, V, walls=walls, lavas=lavas)
    #V = minigrid.Z_to_V(Z)

    # Embedded MDP
    #mdp_minigrid, _ = minigrid.embedding_to_MDP()
    #minigrid_mdp_embedded_plots = Minigrid_MDP_Plotter(mdp_minigrid)
    # Q2, opt_policy2, n_steps = mdp_minigrid.value_iteration(epsilon, gamma)
    # V2 = np.max(Q2, axis=1)

    # qlearning = QLearning(minigrid_mdp, gamma=gamma, epsilon=1, epsilon_decay=0.9995, epsilon_min = 0, c = 100)
    # Q_est, est_policy, _, throughputs = Qlearning_training(qlearning, n_steps=n_iters)
    # print(np.sum(opt_policy- est_policy))
    # print(np.mean(np.square(V - np.max(Q_est, axis=1))))
    # print(opt_policy[opt_policy != est_policy], est_policy[opt_policy != est_policy])
    # print(throughputs[-1])


    lambda_values = [0.1, 1, 2, 5]

    z_rewards = []
    z_names = []
    z_errors = []
    z_values = []
    z_values.append(V)
    z_names.append("Optimal Value Function")

    for c in lambda_values:
        zlearning = ZLearning(minigrid_lmdp, lmbda=c, c=10000)
        Z, V_error, throughputs, rewards = Zlearning_training(zlearning, n_steps=n_iters, V=V)
        V_est = minigrid_lmdp.Z_to_V(Z)
        print(V_error[-1], rewards[-1])

        z_rewards.append(rewards)
        z_names.append(r'$\lambda$=' + str(c))
        z_errors.append(V_error)
        z_values.append(V_est)

    minigrid_mdp_plots.plot_rewards_and_errors(z_errors, z_rewards, z_names[1:], title=r"Minigrid (hill cliff) domain, grid size$=13x13$")
    
    minigrid_mdp_plots.plot_grids_with_policies(env=minigrid_mdp, grid_size= grid_size, value_functions = z_values, names = z_names)




    # minigrid_plots.plot_throughput(throughputs2, grid_size, names = z_names, smooth_window=10000, save_path='plots\Z_c_throughputs')
    # minigrid_plots.plot_throughput(throughputs, grid_size, names = z_names, smooth_window=10000, save_path='plots\Z_c')
    # minigrid_plots.plot_value_per_hyperparameter(v_errors, c_values, title = 'Value Function Approximation Error by Learning Rate in Z-learning', xlabel = 'C', ylabel = 'Approximation Error', save_path = 'plots\Z_c_error')
    # minigrid_plots.plot_value_per_hyperparameter(v_mses, c_values, title = 'Value Function MSE Error by Learning Rate in Z-learning', xlabel = 'C', ylabel = 'Mean Squared Errpr', save_path = 'plots\Z_c_mse')
    # #minigrid_mdp_embedded_plots.plot_value_per_hyperparameter(q_policy_differences, c_values, title = 'Policy Differences by Learning Rate', xlabel = 'C', ylabel = 'Policy Differences', save_path = 'plots\Q_embedded_c_policy_difference')

    # epsilon_decays = [0.9999, 0.9995, 0.999, 0.995, 0.99]
    # epsilon_decays = [0.999, 0.99]
    # q_rewards = []
    # q_names = []
    # q_errors = []
    # q_values = []
    # q_values.append(V)
    # q_names.append("Optimal Value Function")
    
    # for epsilon_decay in epsilon_decays:
    #     qlearning = QLearning(minigrid_mdp, gamma=gamma, c=200, epsilon=1, epsilon_decay=epsilon_decay, epsilon_min = 0)
    #     Q_est, V_error, est_policy, throughputs, rewards = Qlearning_training(qlearning, n_steps=n_iters, V=V)
    #     V_est = np.max(Q_est, axis=1)
    #     q_rewards.append(rewards)
    #     q_names.append(r'$\epsilon_{decay}$=' + str(epsilon_decay))
    #     q_errors.append(V_error)
    #     q_values.append(V_est)
    
    # minigrid_mdp_plots.plot_rewards_and_errors(q_errors, q_rewards, q_names[1:], title=r"Minigrid (hill cliff) domain, grid size$=15x15$, $\epsilon=1$")

    # minigrid_mdp_plots.plot_grids_with_policies(env=minigrid_mdp, grid_size= grid_size, value_functions = q_values, names = q_names)



    # minigrid_mdp_embedded_plots.plot_throughput(q_throughputs, minigrid.grid_size, names = q_names, save_path='plots\Q_epsilon_decays2')
    # minigrid_mdp_embedded_plots.plot_value_per_hyperparameter(q_errors, epsilon_decays, title = 'Value Function Approximation Error by Exploration Decay', xlabel = 'Epsilon Decay', ylabel = 'Approximation Error', save_path = 'plots\Q_epsilon_decay_error2')
    # minigrid_mdp_embedded_plots.plot_value_per_hyperparameter(q_policy_differences, epsilon_decays, title = 'Policy Differences by Exploration Decay', xlabel = 'Epsilon Decay', ylabel = 'Policy Differences', save_path = 'plots\Q_epsilon_decay_policy_difference2')


    #learning_rate_decays = [1, 0.9999, 0.995, 0.99]
    #learning_rates = [1, 0.75, 0.5, 0.25, 0.1, 0.05, 0.01]
    #gammas = [1, 0.99, 0.95, 0.9, 0]
    # lambdas = [1, 0.1, 0.01, 0.001]
    # q_throughputs = []
    # q_names = []
    # q_errors = []
    # q_policies = []
    
    # for lmbda in lambdas:
    #     #qlearning = QLearning(mdp_minigrid, gamma=gamma, learning_rate=0.25, learning_rate_decay=0.9999, learning_rate_min=0.0005, epsilon=1, epsilon_decay=0.9995, epsilon_min = 0)
    #     zlearning = ZLearning(minigrid, lmbda=lmbda, learning_rate=0.25, learning_rate_decay=0.9999, learning_rate_min=0.0005)
    #     #Q_est, est_policy, _, throughputs = Qlearning_training(qlearning, n_steps=n_iters)
    #     _, _, z_throughputs = Zlearning_training(zlearning, n_steps=n_iters)
    #     Pu = zlearning.Pu
    #     q_throughputs.append(z_throughputs)
    #     q_names.append('Lambda: ' + str(lmbda))
    #     #q_errors.append(np.sum(np.abs(Q2-Q_est))/np.sum(np.abs(Q2)))
    #     #q_policies.append(1-np.sum(opt_policy2 != est_policy)/mdp_minigrid.n_states)
    #     q_policies.append(1-np.sum(np.abs(PU-zlearning.Pu))/np.sum(PU))


    # minigrid_mdp_embedded_plots.plot_throughput(q_throughputs, minigrid.grid_size, names = q_names, save_path='plots\Z_lmbda')
    # #minigrid_mdp_embedded_plots.plot_value_per_hyperparameter(q_errors, gammas, title = 'Value Function Approximation Error by Discount Rate', xlabel = 'Discount Rate', ylabel = 'Approximation Error', save_path = 'plots\Q_discount_rate_error')
    # minigrid_mdp_embedded_plots.plot_value_per_hyperparameter(q_policies, lambdas, title = 'PU Approximation by Temperature Parameter', xlabel = 'Lambda', ylabel = 'PU Approximation', save_path = 'plots\Z_lambda_pu')
