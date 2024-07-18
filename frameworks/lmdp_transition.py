import numpy as np
from frameworks.mdp import MDP
from scipy.sparse import csr_matrix, isspmatrix_csr
from frameworks.lmdp import LMDP


class LMDP_transition(LMDP):
    def __init__(self, n_states, n_terminal_states, lmbda = 1, s0 = 0):
        self.n_states = n_states
        self.n_nonterminal_states = n_states - n_terminal_states
        self.P0 = np.zeros((self.n_nonterminal_states, n_states))
        self.R = np.zeros((self.n_nonterminal_states, n_states)) # Assuming terminal states are at the end of the state space
        self.s0 = s0
        self.lmbda = lmbda

    def act(self, current_state, P):
        """Transition function."""

        # Check if the transition matrix is sparse 
        if isspmatrix_csr(P):
            next_state = np.random.choice(P[current_state].indices, p=P[current_state].data) # Using sparse matrix
        else:
            next_state = np.random.choice(self.n_states, p=P[current_state])
        reward = self.R[current_state, next_state]
        terminal = next_state >= self.n_nonterminal_states
        return next_state, reward, terminal
    
    def power_iteration(self, lmbda = None, epsilon = 1e-10):
        """Power iteration algorithm to compute the optimal Z function."""

        lmbda = self.lmbda if lmbda is None else lmbda

        P0 = self.P0 if isspmatrix_csr(self.P0) else csr_matrix(self.P0)
        
        Z = np.ones(self.n_states)
        V_diff = np.arange(self.n_states)
        n_steps = 0
        
        O = csr_matrix(np.exp(self.R / lmbda))
        G = P0.multiply(O) #Doubt 1
        ZT = np.exp(np.mean(self.R[:, self.n_nonterminal_states:], axis=0) / lmbda) #Doubt 2

        while max(V_diff) - min(V_diff) > epsilon:
            TZ = G @ Z
            TZ = np.concatenate((TZ, ZT))
            V_diff = self.Z_to_V(TZ) - self.Z_to_V(Z)
            Z = TZ
            n_steps += 1

        return Z, n_steps