import numpy as np
from environments.LMDP import LMDP

class SimpleGrid(LMDP):

    def __init__(self, size = 2):
        super().__init__(size * size, 1, 4)
        self.T = [size * size - 1]
        
        # construct transition probabilities
        for x in range(size):
            for y in range(size):
                state = x * size + y
                if state < self.n_nonterminal_states:
                    if x > 0:
                        self.P0[state][(x - 1) * size + y] += 1
                    else:
                        self.P0[state][state] += 1
                    if x + 1 < size:
                        self.P0[state][(x + 1) * size + y] += 1
                    else:
                        self.P0[state][state] += 1
                    if y > 0:
                        self.P0[state][x * size + y - 1] += 1
                    else:
                        self.P0[state][state] += 1
                    if y + 1 < size:
                        self.P0[state][x * size + y + 1] += 1
                    else:
                        self.P0[state][state] += 1

                    self.P0[state][:] /= np.sum(self.P0[state])

        # construct reward function
        self.R[0:self.n_nonterminal_states] = -1
