import numpy as np
import matplotlib.pyplot as plt
from gym.wrappers import OrderEnforcing
from frameworks.grid import CustomEnv
import frameworks
from scipy.sparse import csr_matrix, isspmatrix_csr

class MDP:
    def __init__(self, n_states, n_terminal_states, n_actions, gamma = 0.95):
        self.n_states = n_states
        self.n_nonterminal_states = n_states - n_terminal_states
        self.n_actions = n_actions
        self.P = np.zeros((self.n_nonterminal_states, n_actions, n_states))
        self.R = np.zeros((n_states, n_actions)) # Assuming terminal states are at the end of the state space
        self.s0 = 0
        self.gamma = gamma

    def act(self, current_state, action):
        """Transition function."""

        next_state = np.random.choice(self.n_states, p=self.P[current_state, action]) 
        reward = self.R[current_state, action]
        terminal = next_state >= self.n_nonterminal_states
        return next_state, reward, terminal
    
    def value_iteration(self, epsilon=1e-10, gamma = None):
        """Value iteration algorithm."""

        gamma = self.gamma if gamma is None else gamma

        Q = np.zeros((self.n_states, self.n_actions))
        V_diff = np.arange(self.n_states)
        n_steps = 0

        R = self.R[:self.n_nonterminal_states]
        P = gamma * self.P
        QT = self.R[self.n_nonterminal_states:]

        while max(V_diff) - min(V_diff) > epsilon:
            TQ = R + P @ Q.max(axis=1)
            TQ = np.concatenate((TQ, QT))
            V_diff = TQ.max(axis=1) - Q.max(axis=1)
            Q = TQ
            n_steps += 1

        policy = np.argmax(Q, axis=1)

        return Q, policy, n_steps
    
    def shortest_path_length(self, s = None):
        """Compute the shortest optimal path length from a given state to a terminal state.
        :param s: The starting state. """

        s = self.s0 if s is None else s

        _, policy, _ = self.value_iteration()

        done = s >= self.n_nonterminal_states
        n_steps = 0
        while not done:
            s, _, done = self.act(s, policy[s])
            n_steps += 1
        return n_steps

    def embedding_to_LMDP(self, lmbda = 1, gamma = None):
        """Embed the MDP into an LMDP."""

        gamma = self.gamma if gamma is None else gamma

        # Compute the value function of the original MDP without discounting
        Q, _, _ = self.value_iteration(gamma=gamma)
        V = Q.max(axis=1)

        # Create the LMDP
        lmdp = frameworks.lmdp.LMDP(self.n_states, self.n_states - self.n_nonterminal_states)
        # Check if all actions from all states are deterministic. Otherwise, the stochastic LMDP embedding will perform better
        is_deterministic = (np.count_nonzero(self.P, axis=2) == np.ones((self.n_nonterminal_states, self.n_actions))).all()

        # Apply the deterministic LMDP embedding
        if is_deterministic:

            lmdp.R = np.sum(self.R, axis = 1)/self.n_actions
            lmdp.P0 = np.sum(self.P, axis = 1)/self.n_actions

            # Update reward function with KL divergence (SPA embedding)
            Z, _ = lmdp.power_iteration(lmbda)
            Pu = lmdp.compute_Pu(Z, sparse=True)
            row_indices = np.repeat(np.arange(Pu.shape[0]), np.diff(Pu.indptr))
            log_ratio = np.log(Pu.data / lmdp.P0[row_indices, Pu.indices])
            product = Pu.data * log_ratio
            R = np.sum(self.R, axis = 1)/self.n_actions + lmbda * np.concatenate((np.bincount(row_indices, weights=product), lmdp.R[self.n_nonterminal_states:]))

            K_min = 0
            K_max = 1

            #Find the optimal K through ternary search
            while K_max - K_min > 1e-5:
                m1 = K_min + (K_max - K_min) / 3
                lmdp.R = m1 * R
                Z1, _ = lmdp.power_iteration(lmbda)
                mse1 = np.mean(np.square(lmdp.Z_to_V(Z1) - V))
                
                m2 = K_max - (K_max - K_min) / 3
                lmdp.R = m2 * R 
                Z2, _ = lmdp.power_iteration(lmbda)
                mse2 = np.mean(np.square(lmdp.Z_to_V(Z2) - V))
                if mse1 > mse2:
                    K_min = m1
                else:
                    K_max = m2

            lmdp.R = K_min * R

        # Apply the non-deterministic LMDP embedding (from Todorov et al. 2009)
        else:
            epsilon = 1e-10
            D = self.P
            # Find columns that contain any non-zero values
            cols_with_nonzero = np.any(D != 0, axis=1)
            unique_next_state_counts = np.unique(np.sum(cols_with_nonzero, axis=1))

            # Perform the vectorized embedding for each unique number of next states
            for next_state_count in unique_next_state_counts:

                source_states = np.where(np.sum(cols_with_nonzero, axis=1) == next_state_count)[0]
                source_states_repeated = np.repeat(source_states, next_state_count)
                next_states = np.where(cols_with_nonzero[source_states])[1]

                # Remove the columns that contain only zeros and keep only possible transitions
                D_count = np.array([D[i][:, cols_with_nonzero[i]] for i in source_states])

                # Substitute 0s in actual possible transitions columns with 'epsilon' and renormalize
                D_count[D_count == 0] = epsilon
                D_count /= D_count.sum(axis=2, keepdims=True)

                # Apply the Pseudo-Inverse method to both square and non-square matrices
                B = -self.R[source_states] -np.sum(D_count * np.log(D_count), axis = 2)
                pseudo_inverse_D = np.linalg.pinv(D_count)
                C = np.einsum('ijk,ik->ij', pseudo_inverse_D, B)

                R = np.log(np.sum(np.exp(-C), axis=1))
                M = - R[:, np.newaxis] - C

                # Assign the reward and initial state distribution to the LMDP in the corresponding states
                lmdp.R[source_states] = R
                lmdp.P0[source_states_repeated, next_states] = np.exp(M).flatten()

        
        embedding_mse = np.mean(np.square(lmdp.Z_to_V(lmdp.power_iteration(lmbda)[0]) - V))
        lmdp.P0 = csr_matrix(lmdp.P0)
        return lmdp, embedding_mse


class Minigrid_MDP(MDP):

    DIR_TO_VEC = [
        # Pointing right (positive X)
        np.array((1, 0)),
        # Down (positive Y)
        np.array((0, 1)),
        # Pointing left (negative X)
        np.array((-1, 0)),
        # Up (negative Y)
        np.array((0, -1)),
    ]

    def __init__(self, grid_size = 14, walls = [], lavas = [], dynamics = None):
        """Initialize the Minigrid MDP."""

        self.grid_size = grid_size
        self.n_orientations = 4
        self.actions = list(range(3))
        self.J = {"goal": 0, "lava": -grid_size*grid_size} # Determine reward function for terminal states
        n_states, n_terminal_states = self._create_environment(grid_size, walls, lavas)

        super().__init__(n_states = n_states, n_terminal_states = n_terminal_states, n_actions = len(self.actions))
        self.n_cells = int(self.n_states / self.n_orientations)

        if dynamics is None:
            self._create_P()
            self._reward_function()
        else:
            self.P = dynamics['P']
            self.R = dynamics['R']

    def _create_environment(self, grid_size, walls, lavas):
        """Create the Minigrid environment."""

        self.env = OrderEnforcing(CustomEnv(size=grid_size+2, walls=walls, lavas=lavas, render_mode="rgb_array"))
        self.env.reset()

        nonterminal_states = []
        terminal_states = []
        for x in range(1, self.env.grid.height):
            for y in range(1, self.env.grid.width):
                if self._is_valid_position(x, y):
                    for o in range(4):
                        state = (x, y, o)
                        if self._is_terminal(state):
                            terminal_states.append(state)
                        else:
                            nonterminal_states.append(state)

        self.states = nonterminal_states + terminal_states
        assert self.grid_size * self.grid_size - len(walls) == len(self.states) / self.n_orientations, "Invalid number of states"
        self.state_to_index = {state: index for index, state in enumerate(self.states)}
        return len(self.states), len(terminal_states)

    def _create_P(self):
        """Create the transition matrix for a deterministic MDP."""

        for state in range(self.n_nonterminal_states):
            for action in self.actions:
                next_state = self._state_step(self.states[state], action)
                self.P[state][action][self.state_to_index[next_state]] = 1.0

    def _reward_function(self, uniform_reward=True):
        """Create the reward function for a deterministic MDP.
        
        :uniform_reward: Whether all actions lead to the same reward or it depends on their next state."""

        for state in range(self.n_nonterminal_states):
            for action in self.actions:
                if uniform_reward:
                    self.R[state][action] = -1.0
                else:
                    next_state = self._state_step(self.states[state], action)
                    pos = self.env.grid.get(next_state[0], next_state[1])
                    self.R[state][action] = -1.0 if pos is None or pos.type != "goal" else 0.0
        
        for state in range(self.n_nonterminal_states, self.n_states):
            self.R[state] = self.J[self._state_type(self.states[state])]


    def _is_valid_position(self, x: int, y: int) -> bool:
        """Testing whether a coordinate is a valid location."""

        return (
            0 < x < self.env.width and
            0 < y < self.env.height and
            (self.env.grid.get(x, y) is None or self.env.grid.get(x, y).can_overlap())
        )
    
    def _state_type(self, state: tuple[int, int, int]) -> str:
        """Return the type of a state."""

        pos = self.env.grid.get(state[0], state[1])
        return pos.type if pos is not None else None
    
    def _is_terminal(self, state: tuple[int, int, int]) -> bool:
        """Check if a state is terminal."""

        at_goal = self._state_type(state) in self.J.keys()

        return at_goal

    def _state_step(self, state: tuple[int, int, int], action: int) -> tuple[int, int, int]:
        """Utility to move states one step forward, no side effect."""

        x, y, direction = state

        # Default transition to the sink failure state
        assert self._is_valid_position(x, y)

        # Transition left
        if action == self.env.actions.left:
            direction = (direction - 1) % 4
        # Transition right
        elif action == self.env.actions.right:
            direction = (direction + 1) % 4
        # Transition forward
        elif action == self.env.actions.forward:
            fwd_pos = np.array((x, y)) + self.DIR_TO_VEC[direction]
            if self._is_valid_position(*fwd_pos):
                x, y = fwd_pos
        # Error
        else:
            print(action)
            assert False, "Invalid action"

        return x, y, direction
    
    # Core Methods

    def reset(self, **kwargs):
        """Reset the environment to the initial state. Used mainly for framework rendering."""

        state = self.s0
        self.env.reset(**kwargs)
        if state != self.state_to_index[tuple(np.array((*self.env.unwrapped.agent_start_pos, self.env.unwrapped.agent_start_dir), dtype=np.int32))]:
            self.env.unwrapped.agent_pos = self.states[state][:2]
            self.env.unwrapped.agent_dir = self.states[state][2]
        assert self.state_to_index[tuple(self.observation())] == state
        return state
    
    def step(self, state, action):
        """Step function to interact with the environment."""

        next_state, reward, done = self.act(state, action)
        for a in self.actions:
            if self._state_step(self.states[state], a) == self.states[next_state]:
                self.env.step(a)
        return next_state, reward, done

    def observation(self): 
        """Transform observation."""

        obs = (*self.env.agent_pos, self.env.agent_dir)
        return np.array(obs, dtype=np.int32)
    
    def render(self):
        """Render the environment."""

        image = self.env.render()
        plt.imshow(image)
        plt.show()
    
    def embedding_to_LMDP(self):
        """Embed the Minigrid MDP into a Minigrid LMDP."""

        lmdp, embedding_rmse = super().embedding_to_LMDP()
        dynamics = {'P0': lmdp.P0, 'R': lmdp.R}
        lmdp_minigrid = frameworks.lmdp.Minigrid(self.grid_size, walls = self.env.walls, dynamics = dynamics)
        return lmdp_minigrid, embedding_rmse

    # Auxiliary Methods
    
    def print_attributes(self):
        print("Number of states: ", self.n_states)
        print("Number of actions: ", self.n_actions)
        print("Number of grid cells: ", self.n_cells)
    
    def print_environment(self):
        print("States: ", self.states)
        print("Actions: ", self.actions)
