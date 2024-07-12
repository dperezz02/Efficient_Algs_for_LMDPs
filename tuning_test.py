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

    simple6_map = [
        "########",
        "#A     #",
        "#      #",
        "#  W   #",
        "#      #",
        "#      #",
        "#L    G#",
        "########"
    ]

    hill13_map = [
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

    maze18_map = [
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

    rooms18_map = [
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

    grid_size = 15
    wall_percentage = 10
    lava_percentage = 10
    objects = generate_random_walls_and_lavas(grid_size, wall_percentage, lava_percentage)
    #objects = {"walls":[], "lavas":[]}

    grid_map = hill13_map
    grid_size = len(grid_map)-2 if grid_map is not None else grid_size


    # MDP
    minigrid_mdp = Minigrid_MDP(grid_size, objects=objects, map = grid_map)
    #minigrid_mdp.render()
    
    #minigrid_lmdp, error = minigrid_mdp.embedding_to_LMDP()
    minigrid_lmdp = Minigrid_LMDP(grid_size, objects=objects, map = grid_map)

    #minigrid_mdp, error = minigrid_lmdp.embedding_to_MDP()
    #print(error)
    #minigrid_lmdp, error = minigrid_mdp.embedding_to_LMDP()
    #print(error)
    # minigrid_mdp, error = minigrid_lmdp.embedding_to_MDP()
    # print(error)
    minigrid_mdp_plots = Minigrid_MDP_Plotter(minigrid_mdp)
    
    
    # Z, n_steps = minigrid_lmdp.power_iteration(lmbda = 1, epsilon=1e-10)
    # V2 = minigrid_lmdp.Z_to_V(Z)

    gamma = 1
    epsilon = 1e-10
    n_iters = int(7e4)
    lmbda = 1

    # Value Iteration MDP
    # Q, opt_policy, n_steps = minigrid_mdp.value_iteration(epsilon, gamma)
    # V = np.max(Q, axis=1) 
    # 

    Z, _ = minigrid_lmdp.power_iteration(lmbda = lmbda, epsilon=epsilon)
    V = minigrid_lmdp.Z_to_V(Z)
    minigrid_mdp_plots.plot_minigrid(minigrid_mdp, grid_size, V)

    # qlearning = QLearning(minigrid_mdp, gamma=gamma, c=200, epsilon=1, epsilon_decay=0.999, epsilon_min = 0, reset_randomness=0)
    # Q_est, V_error, est_policy, throughputs, rewards = Qlearning_training(qlearning, n_steps=n_iters, V=V)
    # V_est = np.max(Q_est, axis=1)

    # minigrid_mdp_plots.plot_rewards_and_errors([V_error, V_error-5], [rewards, rewards-5], ["A", "B"], title=r"Minigrid domain, grid size$=15x15$")
    


    lmbda_values = [0.1, 0.5, 1, 1.5, 2, 5]

    q_rewards = []
    names = []
    q_errors = []
    values = []
    values.append(V)
    names.append("Optimal Value Function")

    for lmbda in lmbda_values:
        # qlearning = QLearning(minigrid_mdp, gamma=gamma, c=5000000, epsilon=1, epsilon_decay=0.999, epsilon_min = 0, reset_randomness=0)
        # Q_est, V_error, est_policy, throughputs, rewards = Qlearning_training(qlearning, n_steps=n_iters, V=V)
        # V_est = np.max(Q_est, axis=1)

        zlearning = ZLearning(minigrid_lmdp, lmbda=lmbda, c=10000)
        Z, V_error, throughputs, rewards = Zlearning_training(zlearning, n_steps=n_iters, V=V)
        V_est = minigrid_lmdp.Z_to_V(Z)

        q_rewards.append(rewards)
        names.append(r'$\lambda$=' + str(lmbda))
        q_errors.append(V_error)
        values.append(V_est)

    minigrid_mdp_plots.plot_rewards_and_errors(q_errors, q_rewards, names[1:], title=r"Minigrid (hill-cliff) domain, grid size$=13x13$")
    
    minigrid_mdp_plots.plot_grids_with_policies(env=minigrid_mdp, grid_size= grid_size, value_functions = values, names = names)
