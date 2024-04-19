import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
import seaborn as sns
import pandas as pd
import matplotlib as mpl
import pandas as pd

class Plotter:
    def __init__(self):
        try:
            # Try to enable TeX rendering
            plt.rcParams["text.usetex"] = True
        except FileNotFoundError as e:
            # If a FileNotFoundError occurs (related to TeX), disable TeX rendering
            print("Warning: TeX rendering unavailable. Using Matplotlib's default fonts.")
            plt.rcParams["text.usetex"] = False
        
        sns.set_context(context="paper", font_scale=1.2)
        sns.set_style({'font.family': 'serif'})

    def plot_throughput(self, throughputs, grid_size, names, smooth_window = 10000, save_path = 'plots\''):
        df = pd.DataFrame()
        for i in range(len(names)):
            temp_df = pd.DataFrame()
            temp_df['rewards'] = throughputs[i]
            temp_df['smoothed_rewards'] = temp_df['rewards'].rolling(window=smooth_window, min_periods=1, center=True).mean()
            temp_df['rewards'] = temp_df['rewards'] .rolling(window=int(smooth_window/5), min_periods=1, center=True).mean()
            temp_df['index'] = range(len(throughputs[i]))
            temp_df['name'] = names[i]
            df = pd.concat([df, temp_df.reset_index(drop=True)], ignore_index=True)
        
        ax = sns.lineplot(data=df, x='index', y='rewards', hue='name', alpha=0.5, legend=False)
        ax = sns.lineplot(data=df, x='index', y='smoothed_rewards', hue='name', alpha=1.0)
        ax.ticklabel_format(style="sci", scilimits=(0, 0), axis="x")
        ax.set(xlabel="Time Step", ylabel="Episodic Reward")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title("Episodic Throughput in Minigrid " + str(grid_size) + "x"+ str(grid_size) + "Walls")
        plt.tight_layout()
        plt.savefig(save_path + str(grid_size) + 'throughputs.png')
        plt.show()
        plt.clf()


    def plot_episode_length(self, lengths, opt_length, plot_batch=False, batch_size=50):
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

    def plot_episode_throughput(self, throughputs, opt_length, smooth_window=5000):
        plt.plot(range(1, len(throughputs)+1), throughputs, alpha=0.09, color='b')
        plt.axhline(y=1/opt_length, color='r', linestyle='--', alpha=0.5)
        throughputs_series = pd.Series(throughputs)
        smoothed_throughputs = throughputs_series.rolling(window=smooth_window, center=True).mean()
        plt.plot(range(1, len(smoothed_throughputs)+1), smoothed_throughputs, color='b')
        plt.xlabel("Time step")
        plt.ylabel("Averaged Throughput")
        plt.title("Averaged Throughput vs. Time step")
        plt.show()
        #print("Last batch averaged througput: ", smoothed_throughputs[-1])
    
        return smoothed_throughputs
            
    def compare_throughputs(self, throughputs, grid_size, names, save_path = 'plots\''):
        for i, throughput in enumerate(throughputs):
            plt.plot(range(1, len(throughput)+1), throughput, label= names[i])
        plt.axhline(y=1/(2*grid_size-1), color='r', linestyle='--', alpha=0.5)
        plt.xlabel("Time Step")
        plt.ylabel("Average Reward")
        plt.title("Average Reward per Time Step in Minigrid " + str(grid_size) + "x"+ str(grid_size))
        plt.legend()
        plt.savefig(save_path + str(grid_size) + 'throughputs.png')
        plt.show()

    def plot_value_per_hyperparameter(self, values, hyperparameters, title, xlabel, ylabel, save_path = 'plots\''):
        sns.lineplot(x=hyperparameters, y=values)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.savefig(save_path + title + '.png')
        plt.show()
        plt.clf()

    def plot_convergence(self, Opt, Est, model = 'Z-learning'):
        diff = np.abs(Est - Opt).mean(axis=(1))
        plt.plot(diff)
        plt.xlabel('iteration')
        plt.ylabel('Error')
        plt.title(model + " convergence")
        plt.show()

class Minigrid_MDP_Plotter(Plotter):

    def __init__(self, minigrid):
        super().__init__()
        self.minigrid = minigrid

    def minigrid_demo(self):
        print("Welcome to our Minigrid Interactive Demo")
        action = np.random.choice(self.minigrid.n_actions)
        a = input(f"       Select an Action from {self.minigrid.actions} (default {action}): ")
        while a != 'e':
            if a and a.isdigit():
                action = int(a)
                _, reward, terminated, truncated, info = self.minigrid.env.step(action)
                print("Minigrid Step")
                print("       Action:", action)
                print("  Observation:", self.minigrid.observation())
                print("       Reward:", reward)
                print("         Done:", "terminated" if terminated else "truncated" if truncated else "False", )
                print("         Info:", info)
                self.minigrid.render()
            else:
                print("Invalid action.")
            a = input(f"       Select an Action from {self.minigrid.actions} (default {action}, press 'e' to exit): ")

    def state_visited_policy(self, policy):
        state = self.minigrid.s0
        states_visited = np.zeros((self.minigrid.n_states))
        states_visited[state] = 1
        n_steps = 0
        while True:
            n_steps += 1
            next_state, _, done = self.minigrid.act(state, policy[state])
            states_visited[next_state] = 1
            if done:
                break
            else:
                state = next_state

            if n_steps > 100:
                break

        return states_visited

    def plot_greedy_policy(self, greedy_policy, estimated=False, print_policy=False):
        if print_policy: print(greedy_policy)
        states_visited = self.state_visited_policy(greedy_policy)
        plt.figure(figsize=(5, 2*self.minigrid.grid_size))
        plt.imshow(states_visited.reshape((self.minigrid.n_cells,4)), cmap="Greys")
        if estimated: plt.title("Estimated policy \n ( The states in black represent the states selected by the estimated policy)")
        else: plt.title("Optimal policy \n ( The states in black represent the states selected by the Optimal policy)")
        plt.xlabel("Orientation")
        plt.ylabel("Cell Position")
        y_labels = [f"{i//self.minigrid.grid_size + 1}x{i%self.minigrid.grid_size + 1}" for i in range(self.minigrid.n_cells)]
        x_labels = [f"{i}" for i in range(self.minigrid.n_orientations)]
        plt.yticks(range(self.minigrid.n_cells), y_labels)
        plt.xticks(range(self.minigrid.n_orientations), x_labels)
        plt.show()   

    def plot_value_function(self, Q, print_values=False, file = "value_function.txt"):
        V = Q.max(axis=1)
        if print_values: 
            with open(file, "w") as f:
                for s in range(self.minigrid.n_states):
                    f.write("V({}): {}\n".format(self.minigrid.states[s], V[s]))
                    f.write("Q({}): {}\n".format(self.minigrid.states[s], Q[s]))

        plt.figure(figsize=(5, 2*self.minigrid.grid_size))
        im = plt.imshow(V.reshape((self.minigrid.n_cells,4)), vmin= V.min(), vmax=V.max())
        colorbar_ticks = np.linspace(V.max(), V.min(), 10)
        plt.colorbar(im, ticks=colorbar_ticks) 
        plt.title("V(s)")
        plt.xlabel("Orientation")
        plt.ylabel("Cell Position")
        y_labels = [f"{i//self.minigrid.grid_size + 1}x{i%self.minigrid.grid_size + 1}" for i in range(self.minigrid.n_cells)]
        x_labels = [f"{i}" for i in range(self.minigrid.n_orientations)]
        plt.yticks(range(self.minigrid.n_cells), y_labels)
        plt.xticks(range(self.minigrid.n_orientations), x_labels)
        plt.show()

    def plot_path(self, policy, start = 0, path='MDP_policy_path.gif'):
        self.minigrid.s0 = start if self.minigrid.s0 != start else self.minigrid.s0
        s = self.minigrid.reset()
        done = False
        with imageio.get_writer(path, mode='I', duration=0.2) as writer:
            writer.append_data(self.minigrid.env.render())
            while not done:
                s, _, done = self.minigrid.step(s, policy[s])
                writer.append_data(self.minigrid.env.render())
        os.startfile(path)  # for windows

    def plot_greedy_policy_square(self, greedy_policy):
        states_visited = self.state_visited_policy(greedy_policy)
        _, axs = plt.subplots(self.minigrid.grid_size, self.minigrid.grid_size, figsize=(30, 80))

        # Calculate the number of orientations per cell
        num_orientations = self.minigrid.n_orientations

        # Iterate over each cell
        for row in range(self.minigrid.grid_size):
            for col in range(self.minigrid.grid_size):
                ax = axs[row, col]

                # Calculate the indices for the orientations of the current cell
                base_index = self.minigrid.state_to_index[(col + 1, row + 1, 0)]
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
    
