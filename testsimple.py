from environments.simplegrid import SimpleGrid

if __name__ == "__main__":
    g = SimpleGrid(2)
    Z, nsteps = g.power_iteration(1, 1e-10)
    print(Z)
    
    mdp = g.embedding_to_MDP()
    Q, pi, n = mdp.value_iteration(1e-10, 1)
    print(Q)
