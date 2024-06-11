from frameworks.simplegrid import SimpleGrid, SimpleGrid_MDP
import numpy as np
import time

if __name__ == "__main__":
    #grid_size = 2

    # g = SimpleGrid(grid_size)
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
    # #print(mdp.P, mdp.R)
    # #Q, pi, n = mdp.value_iteration(1e-10, 1)
    # #print(Q.max(axis=1))

    # lmdp, embedding_mse = mdp.embedding_to_LMDP()
    # #Z, nsteps = lmdp.power_iteration(1, 1e-10)
    # #print(lmdp.Z_to_V(Z))
    # print("Total embedding error (Deterministic MDP -> LMDP): ", embedding_mse)
    # #print(lmdp.R)+
    
    
    # mdp_minigrid2, embedding_mse = g.embedding_to_MDP()
    # print("Total embedding error (LMDP -> MDP): ", embedding_mse)
    # mdp2 = lmdp.embedding_to_MDP()
    # Q, pi, n = mdp2.value_iteration(1e-10, 1)
    # print(Q.max(axis=1))
    #print(mdp_minigrid2.P)

    #Use a prophiler
    # import cProfile
    # g = SimpleGrid(100)
    # mdp_g, _ = g.embedding_to_MDP()
    # cProfile.run('mdp_g.embedding_to_LMDP()', sort='tottime')
    # cProfile.run('mdp_g.embedding_to_LMDP_loop()', sort='tottime')
    


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


    g = SimpleGrid(2)
    mdp_g, _ = g.embedding_to_MDP()
    lmdp1, error = mdp_g.embedding_to_LMDP()
    lmdp2,  error2 = mdp_g.embedding_to_LMDP_loop()
    print(lmdp1.P0 -lmdp2.P0)
    print(lmdp1.R- lmdp2.R)
    print("Error vectorized: ", error)
    print("Error loop: ", error2)