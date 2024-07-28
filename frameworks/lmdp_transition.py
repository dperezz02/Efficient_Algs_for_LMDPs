import numpy as np
from frameworks.mdp import MDP
from scipy.sparse import csr_matrix, isspmatrix_csr
from frameworks.lmdp import LMDP


class LMDP_transition(LMDP):
    def __init__(self, n_states, n_terminal_states, lmbda = 1, s0 = 0):
        self.n_states = n_states
        self.n_nonterminal_states = n_states - n_terminal_states
        self.P0 = np.zeros((self.n_nonterminal_states, n_states))
        self.R = np.zeros((self.n_nonterminal_states, n_states))
        self.J = np.zeros(n_terminal_states)
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
        R = self.R if isspmatrix_csr(self.R) else csr_matrix(self.R)
        
        Z = np.ones(self.n_states)
        V_diff = np.arange(self.n_states)
        n_steps = 0
        
        O = csr_matrix((np.exp(R.data/lmbda), R.indices, R.indptr), shape=R.shape)
        G = P0.multiply(O)
        ZT = np.exp(self.J / lmbda)

        while max(V_diff) - min(V_diff) > epsilon:
            TZ = G @ Z
            TZ = np.concatenate((TZ, ZT))
            V_diff = self.Z_to_V(TZ) - self.Z_to_V(Z)
            Z = TZ
            n_steps += 1

        return Z, n_steps
    
    
    def embedding_to_MDP(self, lmbda = None):
        """Embed the LMDP into an MDP."""

        lmbda = self.lmbda if lmbda is None else lmbda
        
        # Extract the number of actions from nonzero transition probabilities
        P0 = self.P0.toarray() if isspmatrix_csr(self.P0) else self.P0
        #R = self.R.toarray() if isspmatrix_csr(self.R) else self.R
        n_actions = np.max((P0 > 0).sum(axis=1))
        mdp = MDP(self.n_states, self.n_states - self.n_nonterminal_states, n_actions)
        Z_opt, _ = self.power_iteration(lmbda)
        Pu = self.compute_Pu(Z_opt)

        # Compute the transition probabilities
        n_next_states_per_row = np.diff(Pu.indptr)
        n_next_states = np.unique(n_next_states_per_row)

        # Iterate through all possible transition dimensionalities to avoid heterogeneous matrices
        for next_states in n_next_states:
            source_states = np.where(n_next_states_per_row == next_states)[0]
            source_states_repeated = np.repeat(source_states, next_states)
            indices = Pu[source_states].indices.reshape(-1, next_states)

            for a in range(mdp.n_actions):
                rolled_indices = np.roll(indices, -a, axis=1).flatten()
                mdp.P[source_states_repeated, a, rolled_indices] = Pu[source_states].data

        for state in range(self.n_nonterminal_states):
            for a in range(mdp.n_actions):
                #mdp.R[state,a] = np.dot(mdp.P[state, a, Pu[state].indices], R[state, Pu[state].indices]) - lmbda * np.dot(mdp.P[state, a, Pu[state].indices], np.log(mdp.P[state, a, Pu[state].indices]/P0[state, Pu[state].indices]))
                mdp.R[state,a] = self.R[state, Pu[state].indices].dot(mdp.P[state, a, Pu[state].indices]) - lmbda * np.dot(mdp.P[state, a, Pu[state].indices], np.log(mdp.P[state, a, Pu[state].indices] / P0[state, Pu[state].indices]))

        # Compute the embedding error
        V_lmdp = self.Z_to_V(Z_opt)
        Q, _, _ = mdp.value_iteration(gamma=1)
        V_mdp = Q.max(axis=1)
        embedding_mse = np.mean(np.square(V_lmdp - V_mdp))

        return mdp, embedding_mse