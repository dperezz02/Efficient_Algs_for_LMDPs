import numpy as np
import matplotlib.pyplot as plt
from gym.wrappers import OrderEnforcing
from frameworks.grid import CustomEnv
from frameworks.mdp import MDP, Minigrid_MDP
from scipy.sparse import csr_matrix, isspmatrix_csr


class LMDP:
    def __init__(self, n_states, n_terminal_states):
        self.n_states = n_states
        self.n_nonterminal_states = n_states - n_terminal_states
        self.P0 = np.zeros((self.n_nonterminal_states, n_states))
        self.R = np.zeros(n_states) # Assuming terminal states are at the end of the state space
        self.s0 = 0 

    def act(self, current_state, P):
        """Transition function."""

        # Check if the transition matrix is sparse 
        if isspmatrix_csr(P):
            next_state = np.random.choice(P[current_state].indices, p=P[current_state].data) # Using sparse matrix
        else:
            next_state = np.random.choice(self.n_states, p=P[current_state])
        reward = self.R[next_state]
        terminal = next_state >= self.n_nonterminal_states
        return next_state, reward, terminal
    
    def power_iteration(self, lmbda = 1, epsilon = 1e-10):
        """Power iteration algorithm to compute the optimal Z function."""

        P0 = self.P0 if isspmatrix_csr(self.P0) else csr_matrix(self.P0)
        Z = np.ones(self.n_states)
        V_diff = np.arange(self.n_states)
        n_steps = 0
        
        G = csr_matrix(np.diag(np.exp(self.R[:self.n_nonterminal_states] / lmbda)))
        ZT = np.exp(self.R[self.n_nonterminal_states:] / lmbda)

        while max(V_diff) - min(V_diff) > epsilon:
            TZ = G @ P0 @ Z
            TZ = np.concatenate((TZ, ZT))
            V_diff = self.Z_to_V(TZ) - self.Z_to_V(Z)
            Z = TZ
            n_steps += 1

        return Z, n_steps
    
    
    def compute_Pu(self, Z, sparse = True):
        if sparse:
            P0 = csr_matrix(self.P0) if not isspmatrix_csr(self.P0) else self.P0
            Pu = P0.multiply(Z) # Element-wise multiplication of P0 and Z
            # Normalize each row of the matrix
            row_sums = Pu.sum(axis=1)
            Pu = Pu.multiply(csr_matrix(1.0 / row_sums))

        else:
            P0 = self.P0.toarray() if isspmatrix_csr(self.P0) else self.P0
            Pu = self.P0 * Z  # Element-wise multiplication of P0 and Z
            # Normalize each row of the matrix
            row_sums = Pu.sum(axis=1, keepdims=True)
            Pu /= row_sums
        return Pu
    
    def Z_to_V(self, Z, lmbda = 1):
        V = lmbda * np.log(Z)
        return V
  
    def embedding_to_MDP(self, lmbda = 1):
        """Embed the LMDP into an MDP."""
        
        # Extract the number of actions from nonzero transition probabilities
        P0 = self.P0.toarray() if isspmatrix_csr(self.P0) else self.P0
        n_actions = np.max((P0 > 0).sum(axis=1))
        mdp = MDP(self.n_states, self.n_states - self.n_nonterminal_states, n_actions)
        Z_opt, _ = self.power_iteration(lmbda)
        Pu = self.compute_Pu(Z_opt)

        # Compute the reward function
        row_indices = np.repeat(np.arange(Pu.shape[0]), np.diff(Pu.indptr))
        log_ratio = np.log(Pu.data / P0[row_indices, Pu.indices])
        product = Pu.data * log_ratio
        mdp.R = self.R - lmbda * np.concatenate((np.bincount(row_indices, weights=product), self.R[self.n_nonterminal_states:]))
        mdp.R = np.broadcast_to(mdp.R.reshape(-1, 1), (self.n_states, n_actions))

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

        # Compute the embedding error
        V_lmdp = self.Z_to_V(Z_opt)
        Q, _, _ = mdp.value_iteration(gamma=1)
        V_mdp = Q.max(axis=1)
        embedding_mse = np.mean(np.square(V_lmdp - V_mdp))

        return mdp, embedding_mse
    
    def embedding_to_MDP_loop(self, lmbda = 1):
        """Embed the LMDP into an MDP."""
        
        # Extract the number of actions from nonzero transition probabilities
        P0 = self.P0.toarray() if isspmatrix_csr(self.P0) else self.P0
        n_actions = np.max((P0 > 0).sum(axis=1))
        mdp = MDP(self.n_states, self.n_states - self.n_nonterminal_states, n_actions)
        Z_opt, _ = self.power_iteration(lmbda)
        Pu = self.compute_Pu(Z_opt)
        
        for i in range(mdp.n_states):
            mdp.R[i, :] = self.R[i] 
            if i < self.n_nonterminal_states:
                p0 = self.P0[i,Pu[i].indices].toarray()[0] if isspmatrix_csr(self.P0) else self.P0[i,Pu[i].indices]
                mdp.R[i, :] = self.R[i] - lmbda * np.sum(Pu[i].data * np.log(Pu[i].data / p0))
                data = Pu[i].data
                for a in range(mdp.n_actions):
                    mdp.P[i, a, (Pu[i].indices)] = np.roll(data,a)

        # Compute the embedding error
        V_lmdp = self.Z_to_V(Z_opt)
        Q, _, _ = mdp.value_iteration(gamma=1)
        V_mdp = Q.max(axis=1)
        embedding_mse = np.mean(np.square(V_lmdp - V_mdp))

        return mdp, embedding_mse
    
class Minigrid(LMDP):

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

    def __init__(self, grid_size = 14, walls = [], dynamics = None):
        """Initialize the minigrid environment."""

        self.grid_size = grid_size
        self.n_orientations = 4
        n_states, n_terminal_states = self._create_environment(grid_size, walls)
        
        super().__init__(n_states = n_states, n_terminal_states = n_terminal_states)
        self.n_cells = int(self.n_states / self.n_orientations)
        
        if dynamics is None:
            self._create_P0()
            self._reward_function()
        else:
            self.P0 = dynamics['P0']
            self.R = dynamics['R']

    def _create_environment(self, grid_size, walls):
        """Create the environment for the minigrid."""

        self.env = OrderEnforcing(CustomEnv(size=grid_size+2, walls=walls, render_mode="rgb_array"))
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

    def _create_P0(self, sparse = True):
        """Create the uncontrolled transition probabilities matrix for a stochastic LMDP."""

        transitions = [self.env.actions.left, self.env.actions.right, self.env.actions.forward]
        for state in range(self.n_nonterminal_states):
            for t in transitions:
                next_state = self.state_step(self.states[state], t)
                self.P0[state][self.state_to_index[next_state]] += 1/len(transitions)
        if sparse:
            self.P0 = csr_matrix(self.P0)

    def _reward_function(self):
        """Create the reward function for the minigrid environment."""

        for state in range(self.n_nonterminal_states):
            self.R[state] = -1.0

    def _is_valid_position(self, x: int, y: int) -> bool:
        """Testing whether a coordinate is a valid location."""

        return (
            0 < x < self.env.width and
            0 < y < self.env.height and
            (self.env.grid.get(x, y) is None or self.env.grid.get(x, y).can_overlap())
        )
    
    def _is_terminal(self, state: tuple[int, int, int]) -> bool:
        """Check if a state is terminal."""

        pos = self.env.grid.get(state[0], state[1])
        at_goal = pos is not None and pos.type == "goal"
        return at_goal

    def state_step(self, state: tuple[int, int, int], action: int) -> tuple[int, int, int]:
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
    
    def step(self, state, P):
        """Step function to interact with the environment."""

        next_state, reward, done = self.act(state, P)
        for action in self.actions:
            if self.state_step(self.states[state], action) == self.states[next_state]:
                self.env.step(action)
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
    
    def embedding_to_MDP(self):
        """Embed the Minigrid lMDP into a Minigrid MDP."""

        mdp, embedding_mse = super().embedding_to_MDP()
        dynamics = {'P': mdp.P, 'R': mdp.R}
        mdp_minigrid = Minigrid_MDP(self.grid_size, walls = self.env.walls, dynamics = dynamics)
        return mdp_minigrid, embedding_mse
    
    # Auxiliary Methods
    
    def print_attributes(self):
        print("Number of states: ", self.n_states)
        print("Number of actions: ", self.n_actions)
        print("Number of grid cells: ", self.n_cells)
    
    def print_environment(self):
        print("States: ", self.states)
        print("Actions: ", self.actions)
