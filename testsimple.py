from frameworks.simplegrid import SimpleGrid, SimpleGrid_MDP
from utils.plot import Plotter, Minigrid_MDP_Plotter
import numpy as np
import time

if __name__ == "__main__":
    grid_size = 2

    g = SimpleGrid(grid_size)
    # Z, nsteps = g.power_iteration(1, 1e-10)
    # #print(Z)
    # #print(g.Z_to_V(Z))
    # Pu = g.compute_Pu(Z)

    #mdp_minigrid, embedding_mse = g.embedding_to_MDP()
    #print("Embedding Mean Squared Error: ", embedding_mse)

    #lmdp, embedding_mse = mdp_minigrid.embedding_to_LMDP()
    #print("Embedding Mean Squared Error: ", embedding_mse)
    #print(lmdp.R)

    #Z, nsteps = lmdp.power_iteration(1, 1e-10)
    #print(lmdp.Z_to_V(Z))
    #print("Total embedding error (MDP -> LMDP): ", np.sum(np.abs(lmdp.Z_to_V(Z) - Q.max(axis=1))))

    # mdp2 = lmdp.embedding_to_MDP()
    # Q, pi, n = mdp2.value_iteration(1e-10, 1)
    # #print(Q.max(axis=1))
    # print("Total embedding error (LMDP -> MDP): ", np.sum(np.abs(lmdp.Z_to_V(Z) - Q.max(axis=1))))

    # lmdp2 = mdp2.embedding_to_LMDP()
    # Z, nsteps = lmdp2.power_iteration(1, 1e-10)
    # #print(lmdp2.Z_to_V(Z))
    # print("Total embedding error (MDP -> LMDP): ", np.sum(np.abs(lmdp2.Z_to_V(Z) - Q.max(axis=1))))

    # mdp3 = lmdp2.embedding_to_MDP()
    # Q, pi, n = mdp3.value_iteration(1e-10, 1)
    # #print(Q.max(axis=1))
    # print("Total embedding error (LMDP -> MDP): ", np.sum(np.abs(lmdp2.Z_to_V(Z) - Q.max(axis=1))))

    # mdp = SimpleGrid_MDP(grid_size)
    # minigrid_mdp_plots = Minigrid_MDP_Plotter(mdp)

    # mdp_minigrid2, embedding_mse = g.embedding_to_MDP()
    # print("Total embedding error (LMDP -> MDP): ", embedding_mse)
    # mdp2 = lmdp.embedding_to_MDP()
    # Q, pi, n = mdp2.value_iteration(1e-10, 1)
    # print(Q.max(axis=1))
    # print(mdp_minigrid2.P)

    #Use a prophiler
    # import cProfile
    # g = SimpleGrid(100)
    # mdp_g, _ = g.embedding_to_MDP()
    # lmdp_g, mse = mdp_g.embedding_to_LMDP()
    # print("Embedding Mean Squared Error: ", mse)
    