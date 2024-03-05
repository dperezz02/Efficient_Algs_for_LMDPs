import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import imageio
import os

def power_iteration(env, lmbda, epsilon=1e-30):
    Z = np.ones(env.n_states)
    Z_diff = np.arange(env.n_states)
    n_steps = 0

    G = np.diag(np.exp(env.R[env.S] / lmbda))
    ZT = np.exp(env.R[env.T] / lmbda)

    while max(Z_diff) - min(Z_diff) > epsilon:
        TZ = G @ env.P0 @ Z
        TZ = np.concatenate((TZ, ZT))
        Z_diff = TZ - Z
        Z = TZ
        n_steps += 1

    return Z, n_steps

def compute_Pu(Z, P0):
    Pu = P0 * Z  # Element-wise multiplication of P0 and Z
    # Normalize each row of the matrix
    row_sums = Pu.sum(axis=1, keepdims=True)
    Pu /= row_sums
    return Pu

def compute_Pu_sparse(Z, P0):
    # Element-wise multiplication of P0 and Z
    Pu = P0.multiply(Z)
    # Normalize each row of the matrix
    row_sums = Pu.sum(axis=1)
    Pu = Pu.multiply(csr_matrix(1.0 / row_sums))
    return Pu


def show_Z(Z, lmdp, print_Z=False, plot_Z=True, file = "Z_value.txt"):
    if print_Z:
        with open(file, "w") as f:
            for s in range(lmdp.n_states):
                #print("Z value in state ", lmdp.states[s], ": ", Z[s])
                f.write("Z({}): {}\n".format(lmdp.states[s], Z[s]))
    if plot_Z:
        plt.figure(figsize=(5, 2*lmdp.grid_size))
        im = plt.imshow(Z.reshape((lmdp.n_cells,4)), cmap='viridis', origin='lower')
        plt.title("Z(i)")
        plt.xlabel("Orientation")
        plt.ylabel("Cell Position")
        y_labels = [f"{i//lmdp.grid_size + 1}x{i%lmdp.grid_size + 1}" for i in range(lmdp.n_cells)]
        # Adjust the range of values for the colorbar legend
        colorbar_min = Z.min() 
        colorbar_max = Z.max()
        colorbar_ticks = np.linspace(colorbar_min, colorbar_max, 10)
        x_labels = [f"{i}" for i in range(lmdp.n_orientations)]
        plt.yticks(range(lmdp.n_cells), y_labels)
        plt.xticks(range(lmdp.n_orientations), x_labels)
        plt.colorbar(im, ticks=colorbar_ticks)  # Add colorbar for reference
        plt.show()

def show_Pu(lmdp, PU, print_Pu=False, plot_Pu=True, is_sparse=False):
    if print_Pu:
        for i in range(lmdp.n_states):
           for j in (PU[i].indices if is_sparse else range(lmdp.n_states)):
                if PU[i,j] != 0: print("Pu[", lmdp.states[i], "][", lmdp.states[j], "]: ", PU[i,j])

    if plot_Pu:
        if is_sparse: PU = PU.todense()
        # Create a heatmap
        plt.figure(figsize=(15*lmdp.grid_size, 10*lmdp.grid_size))
        plt.imshow(PU, cmap='viridis', origin='lower')

        # Add colorbar for reference
        cbar = plt.colorbar()
        cbar.set_label("Pij(u*)")

        # Add title and labels
        plt.title("Controlled Transition Probabilities")
        plt.xlabel("Output State j")
        plt.ylabel("Input State i")
        plt.yticks(range(lmdp.n_states), lmdp.states, fontsize=10)
        plt.xticks(range(lmdp.n_states), lmdp.states, fontsize=10, rotation='vertical')

        plt.show()

def plot_sample_path(lmdp, PU, start = 0, path = 'LMDP_PU_path.gif'):
    lmdp.s0 = start if lmdp.s0 != start else lmdp.s0
    s = lmdp.reset()
    done = False
    with imageio.get_writer(path, mode='I', duration=0.2) as writer:
        writer.append_data(lmdp.env.render())
        while not done:
            s, _, done = lmdp.step(s, PU)
            writer.append_data(lmdp.env.render())
    os.startfile(path) #for windows 

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

def compare_Zlearning_estimates(zlearning, Z_opt, Pu_opt):
    for i in range(zlearning.lmdp.n_states):
        print("state:", zlearning.lmdp.states[i])
        print("true: ", Z_opt[i])
        print("est: ", zlearning.Z[i])
        print("----------------------------")

    show_Pu(zlearning.lmdp, zlearning.Pu, print_Pu = True, plot_Pu = False, is_sparse = True)
    print("Total Absolute Error: ", np.sum(np.abs(Pu_opt[0:-4]-zlearning.Pu[0:-4])))