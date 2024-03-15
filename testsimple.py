from environments.simplegrid import SimpleGrid
import numpy as np

if __name__ == "__main__":
    g = SimpleGrid(2)
    Z, nsteps = g.power_iteration(1, 1e-10)
    print(Z)
    print(g.Z_to_V(Z))
    Pu = g.compute_Pu(Z)

    mdp = g.embedding_to_MDP()
    Q, pi, n = mdp.value_iteration(1e-10, 1)
    print(Q.max(axis=1))
