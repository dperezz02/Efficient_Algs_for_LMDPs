from environments.simplegrid import SimpleGrid

if __name__ == "__main__":
    g = SimpleGrid(2)
    Z, nsteps = g.power_iteration(1, 1e-10)
    print(Z)
    print(g.Z_to_V(Z))
    print(g.compute_Pu(Z))
    
    mdp = g.embedding_to_MDP()
    print(mdp.R)
    print(mdp.P)
    Q, pi, n = mdp.value_iteration(1e-10, 1)
    print(Q)

    # print(Q.max(axis=1))
