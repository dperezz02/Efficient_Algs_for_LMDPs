from environments.simplegrid import SimpleGrid_LMDP, SimpleGrid_MDP

if __name__ == "__main__":
    grid_size = 15

    g = SimpleGrid_LMDP(grid_size)

    mdp_minigrid, embedding_mse = g.embedding_to_MDP()
    print("Embedding Mean Squared Error: ", embedding_mse)

    lmdp, embedding_mse = mdp_minigrid.embedding_to_LMDP()
    print("Embedding Mean Squared Error: ", embedding_mse)
 
    