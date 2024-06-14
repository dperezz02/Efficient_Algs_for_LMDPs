import numpy as np
from scipy.sparse import csr_matrix, isspmatrix_csr
import time

#-------------------------------
# Z-Learning implementation
# ------------------------------

class ZLearning:
    """
    Implements Z-learning algorithm
    """
    def __init__(self, lmdp, lmbda=1, c = 1, reset_randomness = 0.0):
        self.lmdp = lmdp
        self.lmbda = lmbda
        self.learning_rate = 1
        self.c = c
        self.n_episodes = 0
        self.Z = np.ones(lmdp.n_states)
        self.Z[self.lmdp.n_nonterminal_states:] = np.exp(self.lmdp.R[self.lmdp.n_nonterminal_states:] / lmbda)
        self.Pu = self.lmdp.P0
        self.reset_randomness = reset_randomness
        self.state = self.lmdp.s0
        self.r = 0
        self.episode_end = False
        self.at_goal = False


    def get_Z(self, r, x):
        """
        :param r: reward of current state
        :param x: current state (in index format)
        :return: Delta update
        """
        Gz = self.lmdp.P0[x].dot(self.Z)

        zjk = np.exp(r / self.lmbda) * Gz

        return zjk - self.Z[x]

    def step(self):

        # Get Delta
        self.r = self.lmdp.R[self.state]
        delta = self.get_Z(self.r, self.state)

        # Update Z
        self.Z[self.state] += self.learning_rate * delta

        # Compute Pu
        self.Pu = self.lmdp.compute_Pu(self.Z)

        # Sample next state
        self.state, _ , self.episode_end = self.lmdp.act(self.state, self.Pu)
        self.at_goal = self.episode_end

        if self.episode_end:
            self.n_episodes += 1
            self.learning_rate = self.c / (self.c + self.n_episodes)
            self.at_goal = self.lmdp.R[self.state] == 0
            self.state = np.random.choice(self.lmdp.n_nonterminal_states) if np.random.rand() < self.reset_randomness else self.lmdp.s0

def Zlearning_training(zlearning: ZLearning, n_steps = int(5e5)):
    tt = 0
    l0 = 0
    s0 = zlearning.lmdp.s0
    opt_paths = list(zlearning.lmdp.shortest_path_length(s) for s in range(zlearning.lmdp.n_states)) if zlearning.reset_randomness != 0 else None
    z_lengths = []
    cumulative_reward = 0
    z_throughputs = np.zeros(n_steps)

    Z_est = np.zeros((n_steps, zlearning.lmdp.n_states))
    start_time = time.time()
    while tt < n_steps:
        zlearning.step()
        Z_est[tt, :] = zlearning.Z
        cumulative_reward += zlearning.r
        tt += 1

        if zlearning.episode_end:
            z_lengths.append(tt-l0)  if zlearning.reset_randomness == 0 else z_lengths.append((tt-l0)/opt_paths[s0]) 
            cumulative_reward = cumulative_reward if zlearning.at_goal else cumulative_reward + np.min(zlearning.mdp.R[zlearning.state])
            z_throughputs[l0:tt] = cumulative_reward if zlearning.reset_randomness == 0 else cumulative_reward*opt_paths[s0]/(tt-l0)
            l0 = tt
            s0 = zlearning.state
            cumulative_reward = 0

        if tt % 10000 == 0:

            elapsed_time = time.time() - start_time
            estimated_total_time = (elapsed_time / tt) * n_steps
            estimated_remaining_time = estimated_total_time - elapsed_time

            print(f"Step: {tt}/{n_steps}, Time: {elapsed_time/60:.2f}m, ETA: {estimated_remaining_time/60:.2f}m")


    if l0 != tt: z_throughputs[l0:tt] = z_throughputs[l0-1]

    zlearning.Pu = zlearning.lmdp.compute_Pu(zlearning.Z)

    return Z_est, z_lengths, z_throughputs
