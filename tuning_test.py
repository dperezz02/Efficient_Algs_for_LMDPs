from environments.minigrids import Minigrid_LMDP, Minigrid_MDP, generate_random_walls_and_lavas
from environments.simplegrid import SimpleGrid_LMDP, SimpleGrid_MDP
import numpy as np
import time
from utils.plot import Plotter, Minigrid_MDP_Plotter
from algs.zlearning import ZLearning, Zlearning_training
from algs.qlearning import QLearning, Qlearning_training
from utils.lmdp_plot import Minigrid_LMDP_Plotter
from scipy.sparse import csr_matrix

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


if __name__ == "__main__":

    grid_size = 15
    wall_percentage = 10
    lava_percentage = 10
    objects = generate_random_walls_and_lavas(grid_size, wall_percentage, lava_percentage)
    #objects = {"walls":[], "lavas":[]}

    grid_map = hill13_map
    grid_size = len(grid_map)-2 if grid_map is not None else grid_size


    # MDP
    minigrid_mdp = Minigrid_MDP(grid_size, objects=objects, map = grid_map)

    minigrid_lmdp = Minigrid_LMDP(grid_size, objects=objects, map = grid_map)

    minigrid_mdp_plots = Minigrid_MDP_Plotter(minigrid_mdp)
    

    gamma = 1
    epsilon = 1e-10
    n_iters = int(7e4)
    lmbda = 1

    Z, _ = minigrid_lmdp.power_iteration(lmbda = lmbda, epsilon=epsilon)
    V = minigrid_lmdp.Z_to_V(Z)
    minigrid_mdp_plots.plot_minigrid(minigrid_mdp, grid_size, V)

    lmbda_values = [0.5, 1, 1.5, 2, 5]

    q_rewards = []
    names = []
    q_errors = []
    values = []
    values.append(V)
    names.append("Optimal Value Function")

    for lmbda in lmbda_values:

        zlearning = ZLearning(minigrid_lmdp, lmbda=lmbda, c=10000)
        Z, V_error, throughputs, rewards = Zlearning_training(zlearning, n_steps=n_iters, V=V)
        V_est = minigrid_lmdp.Z_to_V(Z)

        q_rewards.append(rewards)
        names.append(r'$\lambda$=' + str(lmbda))
        q_errors.append(V_error)
        values.append(V_est)

    minigrid_mdp_plots.plot_rewards_and_errors(q_errors, q_rewards, names[1:], title=r"Minigrid (hill-cliff) domain, grid size$=13x13$")
    
    minigrid_mdp_plots.plot_grids_with_policies(env=minigrid_mdp, grid_size= grid_size, value_functions = values, names = names)
