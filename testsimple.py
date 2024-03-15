from environments.simplegrid import SimpleGrid
import numpy as np

if __name__ == "__main__":
    g = SimpleGrid(2)
    Z, nsteps = g.power_iteration(1, 1e-10)
    print(Z)
    print(g.Z_to_V(Z))
    Pu = g.compute_Pu(Z)

    print(g.P0[0])
    print(Pu[0])
    print(Pu[0].data)
    print(g.P0[0][Pu[0].indices])
    print(Pu[0].data/g.P0[0][Pu[0].indices])
    print(np.log(Pu[0].data/g.P0[0][Pu[0].indices]))
    print(np.sum(np.log(Pu[0].data/g.P0[0][Pu[0].indices])))
    print(g.R[0] - np.sum(Pu[0].data*np.log(Pu[0].data/g.P0[0][Pu[0].indices])))

    print(g.R[1] - np.sum(Pu[1].data*np.log(Pu[1].data/g.P0[1][Pu[1].indices])))
    print(g.R[2] - np.sum(Pu[2].data*np.log(Pu[2].data/g.P0[2][Pu[2].indices])))
    
    mdp = g.embedding_to_MDP()
    print(mdp.R)
    # print(mdp.P)
    # Q, pi, n = mdp.value_iteration(1e-10, 1)
    # print(Q)
    # print(Q.max(axis=1))
