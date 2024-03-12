import numpy as np
import matplotlib.pyplot as plt
from gym.wrappers import OrderEnforcing
from environments.grid import CustomEnv
from environments.MDP import MDP, Minigrid_MDP
from scipy.sparse import csr_matrix, isspmatrix_csr


class LMDP:
    def __init__(self, n_states, n_terminal_states, n_actions):
        self.n_states = n_states
        self.n_nonterminal_states = n_states - n_terminal_states
        self.n_actions = n_actions
        self.P0 = np.zeros((self.n_nonterminal_states, n_states))
        self.R = np.zeros(n_states)
        self.T = [] # Terminal states
        self.s0 = 0 

    def act(self, current_state, P): 
        if isspmatrix_csr(P):
            next_state = np.random.choice(P[current_state].indices, p=P[current_state].data) # Using sparse matrix
        else:
            next_state = np.random.choice(self.n_states, p=P[current_state])
        reward = self.R[next_state]
        terminal = next_state in self.T
        return next_state, reward, terminal
    
    def power_iteration(self, lmbda = 1, epsilon = 1e-10):
        Z = np.ones(self.n_states)
        V_diff = np.arange(self.n_states)
        n_steps = 0
        
        nonterminal_states = [i for i in range(self.n_states) if i not in self.T]
        G = np.diag(np.exp(self.R[nonterminal_states] / lmbda))
        ZT = np.exp(self.R[self.T] / lmbda)

        while max(V_diff) - min(V_diff) > epsilon:
            TZ = G @ self.P0 @ Z
            TZ = np.concatenate((TZ, ZT))
            V_diff = self.Z_to_V(TZ) - self.Z_to_V(Z)
            Z = TZ
            n_steps += 1

        return Z, n_steps
    
    def compute_Pu(self, Z, sparse = True):
        if sparse:
            P0 = csr_matrix(self.P0)
            Pu = P0.multiply(Z) # Element-wise multiplication of P0 and Z
            # Normalize each row of the matrix
            row_sums = Pu.sum(axis=1)
            Pu = Pu.multiply(csr_matrix(1.0 / row_sums))
        else:
            Pu = self.P0 * Z  # Element-wise multiplication of P0 and Z
            # Normalize each row of the matrix
            row_sums = Pu.sum(axis=1, keepdims=True)
            Pu /= row_sums
        return Pu
    
    def Z_to_V(self, Z, lmbda = 1):
        V = lmbda * np.log(Z)
        return V
  
    #TODO: Check embeddings. Z LMDP must be equal to Z from V from embedded MDP from LMDP
    def embedding_to_MDP(self):
        mdp = MDP(self.n_states, self.n_nonterminal_states, self.n_actions)
        Z_opt, _ = self.power_iteration()
        Pu = self.compute_Pu(Z_opt)

        for i in range(mdp.n_states):
            mdp.R[i, :] = self.R[i]
            if not self.terminal(i):
                data = Pu[i].data
                for a in range(mdp.n_actions):
                    mdp.P[i, a, (Pu[i].indices)] = np.roll(data,a)

        return mdp
    
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

    def __init__(self, grid_size = 14, walls = []):
        self.grid_size = grid_size
        self.n_orientations = 4
        super().__init__(n_states = self.n_orientations*(grid_size*grid_size - len(walls)), n_terminal_states = self.n_orientations, n_actions = 3)
        self.actions = list(range(self.n_actions))
        self._create_environment(grid_size, walls)
        self.n_cells = int(self.n_states / self.n_orientations)
        #TODO: Finish adaptations from MDP. Compare Z Learning - Q Learning convergence. Check embbedding value functions.
        self.create_P0()
        self.reward_function()

    def _create_environment(self, grid_size, walls):
        self.env = OrderEnforcing(CustomEnv(size=grid_size+2, walls=walls, render_mode="rgb_array"))
        self.env.reset()

        self.states = [
            (x, y, o) for x in range(1, self.env.grid.height) for y in range(1, self.env.grid.width) for o in range(4)
            if self._is_valid_position(x, y)
        ]
        assert self.grid_size * self.grid_size - len(walls) == len(self.states) / self.n_orientations
        self.state_to_index = {state: index for index, state in enumerate(self.states)}
        self.T = [
            self.state_to_index[s] for s in self.states if self.terminal(self.state_to_index[s])
        ]
        self.S = [
            self.state_to_index[s] for s in self.states if not self.terminal(self.state_to_index[s])
        ]

    def create_P0(self):
        for state in self.S: #range(self.n_states): 
            for action in self.actions:
                next_state = self.state_step(self.states[state], action)
                self.P0[state][self.state_to_index[next_state]] += 1/self.n_actions

    def reward_function(self):
        for state in self.S:
            #pos = self.env.grid.get(state[0], state[1])
            #at_goal = pos is not None and pos.type == "goal"
            self.R[state] = -1.0

    def _is_valid_position(self, x: int, y: int) -> bool:
        """Testing whether a coordinate is a valid location."""
        return (
            0 < x < self.env.width and
            0 < y < self.env.height and
            (self.env.grid.get(x, y) is None or self.env.grid.get(x, y).can_overlap())
        )

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
        state = self.s0
        self.env.reset(**kwargs)
        if state != self.state_to_index[tuple(np.array((*self.env.unwrapped.agent_start_pos, self.env.unwrapped.agent_start_dir), dtype=np.int32))]:
            self.env.unwrapped.agent_pos = self.states[state][:2]
            self.env.unwrapped.agent_dir = self.states[state][2]
        assert self.state_to_index[tuple(self.observation())] == state
        return state
    
    def step(self, state, P):
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
        image = self.env.render()
        plt.imshow(image)
        plt.show()
    
    def terminal(self, x: int) -> bool:
        state = self.states[x]
        pos = self.env.grid.get(state[0], state[1])
        at_goal = pos is not None and pos.type == "goal"
        return at_goal
    
    def embedding_to_MDP(self):
        mdp = super().embedding_to_MDP()
        mdp_minigrid = Minigrid_MDP(self.grid_size, walls = self.env.walls, env = self.env, P = mdp.P, R = mdp.R)
        return mdp_minigrid
    
    # Auxiliary Methods
    
    def print_attributes(self):
        print("Number of states: ", self.n_states)
        print("Number of actions: ", self.n_actions)
        print("Number of grid cells: ", self.n_cells)
    
    def print_environment(self):
        print("States: ", self.states)
        print("Actions: ", self.actions)
