from environments.simplegrid import SimpleGrid, SimpleGrid_MDP
import numpy as np

if __name__ == "__main__":
    g = SimpleGrid(10)
    Z, nsteps = g.power_iteration(1, 1e-10)
    print(g.Z_to_V(Z))
    #print(g.P0)
    Pu = g.compute_Pu(Z)

    mdp = g.embedding_to_MDP()
    Q, pi, n = mdp.value_iteration(1e-10, 1)
    print(Q.max(axis=1))
    #print(mdp.P)

    lmdp = mdp.embedding_to_LMDP()
    Z, nsteps = lmdp.power_iteration(1, 1e-10)
    print(g.Z_to_V(Z))
    #print(lmdp.P0)

    mdp2 = lmdp.embedding_to_MDP()
    Q, pi, n = mdp2.value_iteration(1e-10, 1)
    print(Q.max(axis=1))
    #print(mdp2.P)

    lmdp2 = mdp2.embedding_to_LMDP()
    Z, nsteps = lmdp2.power_iteration(1, 1e-10)
    print(lmdp2.Z_to_V(Z))
    #print(lmdp2.P0)

    mdp3 = lmdp2.embedding_to_MDP()
    Q, pi, n = mdp3.value_iteration(1e-10, 1)
    print(Q.max(axis=1))
    #print(mdp3.P)

    mdp = SimpleGrid_MDP(10)
    Q, pi, n = mdp.value_iteration(1e-10, 1)
    print(Q.max(axis=1))

    lmdp = mdp.embedding_to_LMDP()
    print(lmdp.R)
    print(lmdp.P0)
    Z, nsteps = lmdp.power_iteration(1, 1e-10)
    print(lmdp.Z_to_V(Z))

    mdp2 = lmdp.embedding_to_MDP()
    Q, pi, n = mdp2.value_iteration(1e-10, 1)
    print(Q.max(axis=1))

    lmdp2 = mdp2.embedding_to_LMDP()
    Z, nsteps = lmdp2.power_iteration(1, 1e-10)
    print(lmdp2.Z_to_V(Z))
