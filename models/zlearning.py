import numpy as np
from lmdp_plots import compute_Pu_sparse
from scipy.sparse import csr_matrix

#-------------------------------
# Z-Learning implementation
# ------------------------------

class ZLearning:
    """
    Implements Z-learning algorithm
    """
    def __init__(self, lmdp, lmbda=1, learning_rate=1, min_learning_rate=0.0005, learning_rate_decay=0.99999, reset_randomness = 0.0):
        self.lmdp = lmdp
        self.lmbda = lmbda
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.Z = np.ones(lmdp.n_states)
        self.P0 = csr_matrix(lmdp.P0)
        self.Pu = self.P0
        self.reset_randomness = reset_randomness
        self.state = self.lmdp.s0
        self.episode_end = False

    def get_Z(self, r, x):
        """
        :param r: reward of current state
        :param x: current state (in index format)
        :return: Delta update
        """
        Gz = self.P0[x].dot(self.Z)

        zjk = np.exp(r / self.lmbda) * Gz

        return zjk - self.Z[x]

    def step(self):

        # Get Delta
        r = self.lmdp.R[self.state]
        delta = self.get_Z(r, self.state)

        # Update Z
        self.Z[self.state] += self.learning_rate * delta

        # Compute Pu
        self.Pu = compute_Pu_sparse(self.Z,self.P0)

        # Sample next state
        self.state, _ , self.episode_end = self.lmdp.act(self.state, self.Pu)

        if self.episode_end:
            self.state = np.random.choice(self.n_states) if np.random.rand() < self.reset_randomness else self.lmdp.s0
            self.learning_rate = np.maximum(self.learning_rate * self.learning_rate_decay, self.min_learning_rate)

def Zlearning_training(zlearning: ZLearning, opt_lengths, n_steps = int(5e5)):
    tt = 0
    l0 = 0
    s0 = 0
    z_lengths = []
    z_throughputs = np.zeros(n_steps)

    Z_est = np.zeros((n_steps, zlearning.lmdp.n_states))
    while tt < n_steps:
        zlearning.step()
        Z_est[tt, :] = zlearning.Z
        tt += 1

        if zlearning.episode_end:
            z_lengths.append(tt-l0)
            #z_throughputs[l0:tt] = 1/(tt-l0)
            z_throughputs[l0:tt] = opt_lengths[s0]/(tt-l0)
            l0 = tt
            s0 = zlearning.state

        if tt % 10000 == 0:
            print("Step: ", tt)

    if l0 != tt: z_throughputs[l0:tt] = z_throughputs[l0-1]

    zlearning.Pu = compute_Pu_sparse(zlearning.Z,zlearning.P0)

    return Z_est, z_lengths, z_throughputs
