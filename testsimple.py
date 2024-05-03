from environments.simplegrid import SimpleGrid, SimpleGrid_MDP
import numpy as np

if __name__ == "__main__":
    grid_size = 3
    g = SimpleGrid(grid_size)
    Z, nsteps = g.power_iteration(1, 1e-10)
    print(g.Z_to_V(Z))
    #print(g.P0)
    Pu = g.compute_Pu(Z)

    mdp = g.embedding_to_MDP()
    Q, pi, n = mdp.value_iteration(1e-10, 1)
    #print(mdp.R)
    print(Q.max(axis=1))
    #print(mdp.P)

    lmdp = mdp.embedding_to_LMDP()
    Z, nsteps = lmdp.power_iteration(1, 1e-10)
    #print("R:", lmdp.R)
    #print("P0:", lmdp.P0)
    print(g.Z_to_V(Z))

    mdp2 = lmdp.embedding_to_MDP()
    Q, pi, n = mdp2.value_iteration(1e-10, 1)
    print(Q.max(axis=1))
    #print(mdp2.P)

    lmdp2 = mdp2.embedding_to_LMDP()
    #print("R:", lmdp2.R)
    #print("P0:", lmdp2.P0)
    Z, nsteps = lmdp2.power_iteration(1, 1e-10)
    print(lmdp2.Z_to_V(Z))
    #print(lmdp2.P0)

    mdp3 = lmdp2.embedding_to_MDP()
    Q, pi, n = mdp3.value_iteration(1e-10, 1)
    print(Q.max(axis=1))
    #print(mdp3.P)

    # mdp = SimpleGrid_MDP(grid_size)
    # #print(mdp.P, mdp.R)
    # Q, pi, n = mdp.value_iteration(1e-10, 1)
    # print(Q.max(axis=1))

    #D = mdp.P[0][:,:-1] + 1e-10
    #D = np.ones(D.shape) / 3
    # D = [[0.95, 0.025, 0.025],
    #      [0.025, 0.95, 0.025],
    #      [0.025, 0.005, 0.95],
    #      [0.95, 0.025, 0.025]]
    # ba = -mdp.R[0][:]  -np.sum(D * np.log(D), axis = 1)

    # # print(D, ba)
    # # print("Entropy: ", -np.sum(D * np.log(D), axis = 1))

    # # # Use a linear solver to get c from D@c = ba
    # # #c = np.linalg.solve(D, ba)
    # # c = np.linalg.pinv(D) @ ba

    # # print(c)

    # # q = -np.log(np.sum(np.exp(-c)))

    # # print(q)

    # # m = q - c
    # # print(m)

    # # print("R:", -q, "P0:", np.exp(m))

    # lmdp = mdp.embedding_to_LMDP()
    # # print(lmdp.R)
    # # print(lmdp.P0)
    # Z, nsteps = lmdp.power_iteration(1, 1e-10)
    # print(lmdp.Z_to_V(Z))

    # mdp2 = lmdp.embedding_to_MDP()
    # Q, pi, n = mdp2.value_iteration(1e-10, 1)
    # print(Q.max(axis=1))

    # lmdp2 = mdp2.embedding_to_LMDP()
    # Z, nsteps = lmdp2.power_iteration(1, 1e-10)
    # print(lmdp2.Z_to_V(Z))
