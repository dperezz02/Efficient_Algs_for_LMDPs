import numpy as np
import matplotlib.pyplot as plt
import imageio
import os

def state_visited_policy(policy, tmp_env):
    state, _ = tmp_env.reset()
    states_visited = np.zeros((tmp_env.n_states))
    states_visited[state] = 1
    n_steps = 0
    while True:
        n_steps += 1
        next_state, _, done = tmp_env.act(state, policy[state])
        states_visited[next_state] = 1
        if done:
            break
        else:
            state = next_state

        if n_steps > 100:
            break

    return states_visited

def minigrid_demo(minigrid):
    print("Welcome to our Minigrid Interactive Demo")
    action = np.random.choice(minigrid.n_actions)
    a = input(f"       Select an Action from {minigrid.actions} (default {action}): ")
    while a != 'e':
        if a and a.isdigit():
            action = int(a)
            _, reward, terminated, truncated, info = minigrid.env.step(action)
            print("Minigrid Step")
            print("       Action:", action)
            print("  Observation:", minigrid.observation())
            print("       Reward:", reward)
            print("         Done:", "terminated" if terminated else "truncated" if truncated else "False", )
            print("         Info:", info)
            minigrid.render()
        else:
            print("Invalid action.")
        a = input(f"       Select an Action from {minigrid.actions} (default {action}, press 'e' to exit): ")


def bellman_operator(Q, env, gamma=0.95):
    TQ = np.zeros((env.n_states, env.n_actions))

    greedy_policy = np.zeros(env.n_states, dtype=int)

    for s in range(env.n_states):
        if env.terminal(s):
            TQ[s,:] = env.R[s,:]
        else:
            for a in env.actions:
                prob = env.P[s, a]
                reward = float(env.R[s][a])

                TQ[s, a] = np.sum(prob * (reward + gamma * Q.max(axis=1)))

    greedy_policy = np.argmax(TQ, axis=1)

    return TQ, greedy_policy

def value_iteration(Q0, env, epsilon=1e-6, gamma = 0.95, max_iters=10000):
    n_steps = 0
    Q = Q0

    for i in range(max_iters):
        n_steps += 1
        TQ, greedy_policy = bellman_operator(Q, env, gamma)

        err = np.abs(TQ - Q).max()
        if err < epsilon:
            return TQ, greedy_policy, n_steps

        Q = TQ

    return Q, greedy_policy, n_steps


def plot_greedy_policy(greedy_policy, mdp, estimated=False, print_policy=False):
    if print_policy: print(greedy_policy)
    states_visited = state_visited_policy(greedy_policy, mdp)
    plt.figure(figsize=(5, 2*mdp.grid_size))
    plt.imshow(states_visited.reshape((mdp.n_cells,4)), cmap="Greys")
    if estimated: plt.title("Estimated policy \n ( The states in black represent the states selected by the estimated policy)")
    else: plt.title("Optimal policy \n ( The states in black represent the states selected by the Optimal policy)")
    plt.xlabel("Orientation")
    plt.ylabel("Cell Position")
    y_labels = [f"{i//mdp.grid_size + 1}x{i%mdp.grid_size + 1}" for i in range(mdp.n_cells)]
    x_labels = [f"{i}" for i in range(mdp.n_orientations)]
    plt.yticks(range(mdp.n_cells), y_labels)
    plt.xticks(range(mdp.n_orientations), x_labels)
    plt.show()   

def plot_value_function(Q, mdp, print_values=False, file = "value_function.txt"):
    V = Q.max(axis=1)
    if print_values: 
        with open(file, "w") as f:
            for s in range(mdp.n_states):
                f.write("V({}): {}\n".format(mdp.states[s], V[s]))
                f.write("Q({}): {}\n".format(mdp.states[s], Q[s]))

    plt.figure(figsize=(5, 2*mdp.grid_size))
    im = plt.imshow(V.reshape((mdp.n_cells,4)), vmin= V.min(), vmax=V.max())
    colorbar_ticks = np.linspace(V.max(), V.min(), 10)
    plt.colorbar(im, ticks=colorbar_ticks) 
    plt.title("V(s)")
    plt.xlabel("Orientation")
    plt.ylabel("Cell Position")
    y_labels = [f"{i//mdp.grid_size + 1}x{i%mdp.grid_size + 1}" for i in range(mdp.n_cells)]
    x_labels = [f"{i}" for i in range(mdp.n_orientations)]
    plt.yticks(range(mdp.n_cells), y_labels)
    plt.xticks(range(mdp.n_orientations), x_labels)
    plt.show()

def plot_path(mdp, policy, start = 0, path='MDP_policy_path.gif'):
    mdp.s0 = start if mdp.s0 != start else mdp.s0
    s = mdp.reset()
    done = False
    with imageio.get_writer(path, mode='I', duration=0.2) as writer:
        writer.append_data(mdp.env.render())
        while not done:
            s, _, done = mdp.step(s, policy[s])
            writer.append_data(mdp.env.render())
    os.startfile(path)  # for windows

def shortest_path_length(mdp, optimal_policy, s=0):
    done = mdp.terminal(s)
    n_steps = 0
    while not done:
        s, _, done = mdp.act(s, optimal_policy[s])
        n_steps += 1
    return n_steps

def plot_greedy_policy_square(greedy_policy, mdp):
    states_visited = state_visited_policy(greedy_policy, mdp)
    _, axs = plt.subplots(mdp.grid_size, mdp.grid_size, figsize=(30, 80))

    # Calculate the number of orientations per cell
    num_orientations = mdp.n_orientations

    # Iterate over each cell
    for row in range(mdp.grid_size):
        for col in range(mdp.grid_size):
            ax = axs[row, col]

            # Calculate the indices for the orientations of the current cell
            base_index = mdp.state_to_index[(col + 1, row + 1, 0)]
            indices = [base_index + o for o in range(num_orientations)]

            # Create an empty grid for the current cell
            cell_grid = np.zeros((num_orientations, num_orientations))

            # Mark visited states within the cell
            for index in indices:
                if states_visited[index] == 1:
                    orientation = index - base_index
                    cell_grid[orientation % num_orientations, orientation // num_orientations] = 1

            # Plot the grid for the current cell
            ax.imshow(cell_grid, cmap="Greys", origin='upper')

            # Set title and remove axes ticks
            ax.set_title(f"Cell ({col+1}, {row+1})")
            ax.axis("off")

    plt.suptitle("Greedy Policy\n(The states in black represent the states selected by the greedy policy)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
def plot_episode_length(lengths, opt_length, plot_batch=False, batch_size=50):
    print("Number of episodes: ", len(lengths))
    plt.plot(range(1, len(lengths)+1), lengths)
    plt.axhline(y=opt_length, color='r', linestyle='--', alpha=0.5)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length vs. Episode")
    plt.show()
    print("Last episode length: ",lengths[-1])
    if plot_batch:
        averaged_lengths = [sum(lengths[i:i+batch_size]) / batch_size for i in range(0, len(lengths)-len(lengths)%batch_size, batch_size)]
        averaged_lengths.append(np.mean(lengths[-(len(lengths) % batch_size):]))
        plt.plot(range(1, len(averaged_lengths)+1), averaged_lengths)
        plt.axhline(y=opt_length, color='r', linestyle='--', alpha=0.5)
        plt.xlabel("Batch of Episodes")
        plt.ylabel("Average Episode Length")
        plt.title("Episodes Batch Length vs. Batch of Episodes")
        plt.show()
        print("Last batch averaged length: ", averaged_lengths[-1])

def plot_episode_throughput(throughputs, opt_length, plot_batch=False, batch_size=1000):
    plt.plot(range(1, len(throughputs)+1), throughputs)
    plt.axhline(y=1/opt_length, color='r', linestyle='--', alpha=0.5)
    plt.xlabel("Time step")
    plt.ylabel("Throughput")
    plt.title("Throughput vs. Time step")
    plt.show()
    print("Last time step throughput: ",throughputs[-1])
    averaged_throughputs = throughputs

    if plot_batch:
        n_steps = len(throughputs)
        averaged_throughputs = np.zeros(n_steps)
        for b in range(0, n_steps, batch_size):
            averaged_throughputs[b:b+batch_size] = np.mean(throughputs[b:b+batch_size])
        plt.plot(range(1, len(averaged_throughputs)+1), averaged_throughputs)
        plt.axhline(y=1/opt_length, color='r', linestyle='--', alpha=0.5)
        plt.xlabel("Time step")
        plt.ylabel("Batch Averaged Throughput")
        plt.title("Batch Averaged Throughput vs. Time step")
        plt.show()
        print("Last batch averaged througput: ", averaged_throughputs[-1])
    
    return averaged_throughputs
        
def compare_throughputs(througput1, throughput2, grid_size, name1, name2):
    plt.plot(range(1, len(througput1)+1), througput1, label= name1)
    plt.plot(range(1, len(throughput2)+1), throughput2, label= name2)
    plt.axhline(y=1/(2*grid_size-1), color='r', linestyle='--', alpha=0.5)
    plt.xlabel("Time step")
    plt.ylabel("Averaged Throughput")
    plt.title("Minigrid " + str(grid_size) + "x"+ str(grid_size) + " " + name1 + " vs " + name2)
    plt.legend()
    plt.show()

def plot_convergence(Opt, Est, model = 'Z-learning'):
    diff = np.abs(Est - Opt).mean(axis=(1))
    plt.plot(diff)
    plt.xlabel('iteration')
    plt.ylabel('Error')
    plt.title(model + " convergence")
    plt.show()