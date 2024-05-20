import numpy as np
import matplotlib.pyplot as plt
from gym.wrappers import OrderEnforcing
from environments.grid import CustomEnv
from environments.mdp import MDP, Minigrid_MDP
from scipy.sparse import csr_matrix, isspmatrix_csr


class LMDP:
    def __init__(self, n_states, n_terminal_states):
        self.n_states = n_states
        self.n_nonterminal_states = n_states - n_terminal_states
        self.S = np.concatenate((np.ones(self.n_nonterminal_states, dtype=np.int8), np.zeros(n_terminal_states, dtype=np.int8)))
        self.P0 = np.zeros((self.n_nonterminal_states, n_states))
        self.R = np.zeros(n_states)
        self.s0 = 0 

    def act(self, current_state, P):
        """Transition function."""

        # Check if the transition matrix is sparse 
        if isspmatrix_csr(P):
            next_state = np.random.choice(P[current_state].indices, p=P[current_state].data) # Using sparse matrix
        else:
            next_state = np.random.choice(self.n_states, p=P[current_state])
        reward = self.R[next_state]
        terminal = self.S[next_state] == 0
        return next_state, reward, terminal
    
    def power_iteration(self, lmbda = 1, epsilon = 1e-10):
        """Power iteration algorithm to compute the optimal Z function."""

        Z = np.ones(self.n_states)
        V_diff = np.arange(self.n_states)
        n_steps = 0
        
        nonterminal_states = np.where(self.S)[0]
        G = np.diag(np.exp(self.R[nonterminal_states] / lmbda))
        ZT = np.exp(self.R[np.where(self.S == 0)[0]] / lmbda)

        while max(V_diff) - min(V_diff) > epsilon:
            TZ = G @ self.P0 @ Z
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
        mdp.S = self.S
        Z_opt, _ = self.power_iteration(lmbda)
        Pu = self.compute_Pu(Z_opt)

        # Compute the reward function
        row_indices = np.repeat(np.arange(Pu.shape[0]), np.diff(Pu.indptr))
        log_ratio = np.log(Pu.data / P0[row_indices, Pu.indices])
        product = Pu.data * log_ratio
        mdp.R = self.R - lmbda * np.concatenate((np.bincount(row_indices, weights=product), self.R[np.where(self.S == 0)[0]]))
        mdp.R = np.broadcast_to(mdp.R.reshape(-1, 1), (self.n_states, n_actions))

        # Compute the transition probabilities
        nnz_per_row = np.diff(Pu.indptr)
        source_states = np.repeat(np.arange(len(nnz_per_row)), nnz_per_row)
        indices = Pu.indices.reshape(-1, 3)

        for a in range(mdp.n_actions):
            rolled_indices = np.roll(indices, -a, axis=1).flatten()
            mdp.P[source_states, a, rolled_indices] = Pu.data

        # Compute the embedding error
        V_lmdp = self.Z_to_V(Z_opt)[np.where(self.S)[0]]
        Q, _, _ = mdp.value_iteration(gamma=1)
        V_mdp = Q.max(axis=1)[np.where(self.S)[0]]
        embedding_rmse = np.mean(np.square(V_lmdp - V_mdp)/np.square(V_lmdp))

        return mdp, embedding_rmse
    
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
        self._create_environment(grid_size, walls)
        
        super().__init__(n_states = len(self.states), n_terminal_states = np.count_nonzero(self.S == 0))
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

        self.states = [
            (x, y, o) for x in range(1, self.env.grid.height) for y in range(1, self.env.grid.width) for o in range(4)
            if self._is_valid_position(x, y)
        ]
        assert self.grid_size * self.grid_size - len(walls) == len(self.states) / self.n_orientations
        self.state_to_index = {state: index for index, state in enumerate(self.states)}
        
        # Keep track of the terminal and non-terminal states
        self.S = np.array([
            0 if self._is_terminal(s) else 1 for s in range(len(self.states))
        ])

    def _create_P0(self, sparse = True):
        """Create the uncontrolled transition probabilities matrix for a stochastic LMDP."""

        transitions = [self.env.actions.left, self.env.actions.right, self.env.actions.forward]
        for state in np.where(self.S)[0]:
            for t in transitions:
                next_state = self.state_step(self.states[state], t)
                self.P0[state][self.state_to_index[next_state]] += 1/len(transitions)
        if sparse:
            self.P0 = csr_matrix(self.P0)

    def _reward_function(self):
        """Create the reward function for the minigrid environment."""

        for state in np.where(self.S)[0]:
            self.R[state] = -1.0

    def _is_valid_position(self, x: int, y: int) -> bool:
        """Testing whether a coordinate is a valid location."""

        return (
            0 < x < self.env.width and
            0 < y < self.env.height and
            (self.env.grid.get(x, y) is None or self.env.grid.get(x, y).can_overlap())
        )
    
    def _is_terminal(self, x: int) -> bool:
        """Check if a state is terminal."""

        state = self.states[x]
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

        mdp, embedding_rmse = super().embedding_to_MDP()
        dynamics = {'P': mdp.P, 'R': mdp.R}
        mdp_minigrid = Minigrid_MDP(self.grid_size, walls = self.env.walls, dynamics = dynamics)
        return mdp_minigrid, embedding_rmse
    
    # Auxiliary Methods
    
    def print_attributes(self):
        print("Number of states: ", self.n_states)
        print("Number of actions: ", self.n_actions)
        print("Number of grid cells: ", self.n_cells)
    
    def print_environment(self):
        print("States: ", self.states)
        print("Actions: ", self.actions)
