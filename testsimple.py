from frameworks.simplegrid import SimpleGrid, SimpleGrid_MDP
from utils.plot import Plotter, Minigrid_MDP_Plotter
import numpy as np
import time

if __name__ == "__main__":
    grid_size = 5

    #g = SimpleGrid(grid_size)
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

    mdp = SimpleGrid_MDP(grid_size)
    minigrid_mdp_plots = Minigrid_MDP_Plotter(mdp)
    
    # start_time = time.time()
    # minigrid_lmdp1, embedding_mse = mdp.embedding_to_LMDP()
    # print("Execution time: ", time.time() - start_time)
    # print("Embedding Mean Squared Error: ", embedding_mse)
    # start_time = time.time()
    # minigrid_lmdp4, embedding_mse4 = mdp.embedding_to_LMDP(todorov=True)
    # print("Execution time: ", time.time() - start_time)
    # print("Embedding Mean Squared Error: ", embedding_mse4)
    

    # minigrid_mdp_plots.plot_embedding_error_scatter(mdp, [minigrid_lmdp1, minigrid_lmdp4], ["TS", "TE"], save_path = "plots\embedding_error_scatter_simple.png")

    grid_sizes = [2,3,5,10,15,20,30,40,50]
    mses = [[],[],[]]#,[]]
    for grid_size in grid_sizes:
        print(grid_size)
        mdp = SimpleGrid_MDP(grid_size)
        _, embedding_mse = mdp.embedding_to_LMDP()
        mses[0].append(embedding_mse)
        print(embedding_mse)
        # _, embedding_mse = mdp.embedding_to_LMDP(binary=True)
        # mses[1].append(embedding_mse)
        _, embedding_mse = mdp.embedding_to_LMDP(iterative=True)
        mses[1].append(embedding_mse)
        print(embedding_mse)
        # _, embedding_mse = mdp.embedding_to_LMDP(todorov=True)
        # mses[2].append(embedding_mse)
    
    minigrid_mdp_plots.plot_mse_vs_grid_size([4,9,25,100,225,400,900,1600,2500], mses, ["TS", "TSA"], save_path = "plots/")
    # lmdp, embedding_mse = mdp.embedding_to_LMDP()
    # print("Total embedding error (Deterministic MDP -> LMDP): ", embedding_mse)
    #print(lmdp.R)
    
    
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
    


    # for grid_size in [2, 3, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
    #     g = SimpleGrid(grid_size)
    #     mdp_g, _ = g.embedding_to_MDP()

    #     start_time = time.time()
    #     for i in range(10):
    #         mdp_g.embedding_to_LMDP()
    #     end_time = time.time()
    #     print("Execution time for vectorized embedding in grid size", grid_size, ": ", (end_time - start_time) / 10)

    #     start_time = time.time()
    #     for i in range(10):
    #         mdp_g.embedding_to_LMDP_loop()
    #     end_time = time.time()
    #     print("Execution time for loop embedding in grid size", grid_size, ": ", (end_time - start_time) / 10)