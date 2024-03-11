import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import imageio
import os
#TODO: convert to V to compare epsilon as in value iteration.
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

def compare_Zlearning_estimates(zlearning, Z_opt, Pu_opt):
    for i in range(zlearning.lmdp.n_states):
        print("state:", zlearning.lmdp.states[i])
        print("true: ", Z_opt[i])
        print("est: ", zlearning.Z[i])
        print("----------------------------")

    show_Pu(zlearning.lmdp, zlearning.Pu, print_Pu = True, plot_Pu = False, is_sparse = True)
    print("Total Absolute Error: ", np.sum(np.abs(Pu_opt[0:-4]-zlearning.Pu[0:-4])))

def value_function_to_Z(V, lmbda=1):
    Z = np.exp(V / lmbda)
    return Z

def value_function_to_V(Z, lmbda=1):
    V = lmbda*np.log(Z)
    return V