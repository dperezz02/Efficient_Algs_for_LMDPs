from environments.simplegrid import SimpleGrid, SimpleGrid_MDP
import numpy as np

if __name__ == "__main__":
    grid_size = 2

    g = SimpleGrid(grid_size)
    Z, nsteps = g.power_iteration(1, 1e-10)
    #print(g.Z_to_V(Z))
    Pu = g.compute_Pu(Z)

    mdp_minigrid, embedding_rmse = g.embedding_to_MDP()
    print("Embedding Mean Squared Error: ", np.round(embedding_rmse*100,10), "%")


    #lmdp = mdp.embedding_to_LMDP()
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
    #print(mdp.P, mdp.R)
    #Q, pi, n = mdp.value_iteration(1e-10, 1)
    #print(Q.max(axis=1))

    lmdp, rmse = mdp.embedding_to_LMDP()
    #Z, nsteps = lmdp.power_iteration(1, 1e-10)
    #print(lmdp.Z_to_V(Z))
    #print("Total embedding error (LMDP -> MDP): ", np.sum(np.abs(lmdp.Z_to_V(Z) - Q.max(axis=1)))/np.sum(np.abs(Q.max(axis=1))), "%")
    #print(lmdp.R)

    mdp_minigrid2, embedding_rmse = g.embedding_to_MDP()
    print("Embedding Mean Squared Error: ", np.round(embedding_rmse*100,5), "%")
    # mdp2 = lmdp.embedding_to_MDP()
    # Q, pi, n = mdp2.value_iteration(1e-10, 1)
    # print(Q.max(axis=1))

    # lmdp2 = mdp2.embedding_to_LMDP()
    # Z, nsteps = lmdp2.power_iteration(1, 1e-10)
    # print(lmdp2.Z_to_V(Z))