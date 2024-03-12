import numpy as np
import matplotlib.pyplot as plt
from gym.wrappers import OrderEnforcing
from environments.grid import CustomEnv

class MDP:
    def __init__(self, n_states, n_terminal_states, n_actions):
        self.n_states = n_states
        self.n_nonterminal_states = n_states - n_terminal_states
        self.n_actions = n_actions
        self.P = np.zeros((self.n_nonterminal_states, n_actions, n_states))
        self.R = np.zeros((n_states, n_actions))
        self.T = [] # Terminal states
        self.s0 = 0 

    def act(self, current_state, action):
        next_state = np.random.choice(self.n_states, p=self.P[current_state, action]) 
        reward = self.R[current_state, action]
        terminal = next_state in self.T
        return next_state, reward, terminal
    
    def value_iteration(self, epsilon=1e-10, gamma = 0.95):
        Q = np.zeros((self.n_states, self.n_actions))
        V_diff = np.arange(self.n_states)
        n_steps = 0

        nonterminal_states = [i for i in range(self.n_states) if i not in self.T]
        R = self.R[nonterminal_states]
        P = gamma * self.P[nonterminal_states]
        QT = self.R[self.T]

        while max(V_diff) - min(V_diff) > epsilon:
            TQ = R + P @ Q.max(axis=1)
            TQ = np.concatenate((TQ, QT))
            V_diff = TQ.max(axis=1) - Q.max(axis=1)
            Q = TQ
            n_steps += 1

        greedy_policy = np.argmax(Q, axis=1)

        return Q, greedy_policy, n_steps
    
    def shortest_path_length(self, optimal_policy, s=0):
        done = s in self.T
        n_steps = 0
        while not done:
            s, _, done = self.act(s, optimal_policy[s])
            n_steps += 1
        return n_steps
    
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

    def __init__(self, grid_size = 14, walls = [], env = None, P = None, R = None):
        self.grid_size = grid_size
        self.n_orientations = 4
        super().__init__(n_states = self.n_orientations*(grid_size*grid_size - len(walls)), n_terminal_states = self.n_orientations, n_actions = 3)
        self.actions = list(range(self.n_actions))
        self._create_environment(grid_size, walls, env)
        self.n_cells = int(self.n_states / self.n_orientations)

        if P is None:
            self._create_P()
        else:
            self.P = P

        if R is None: 
            self._reward_function()
        else: 
            self.R = R

    def _create_environment(self, grid_size, walls, env):
        if env is None: 
            self.env = OrderEnforcing(CustomEnv(size=grid_size+2, walls=walls, render_mode="rgb_array"))
        else: 
            self.env = env
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

    def _create_P(self): #index and np.array can be changed
        for state in self.S:
            for action in self.actions:
                next_state = self.state_step(self.states[state], action)
                self.P[state][action] = np.array([1.0 if s == next_state else 0.0 for s in self.states]) # Assuming deterministic transitions

    def _reward_function(self):
        for state in self.S:
            for action in self.actions:
                # next_state = self.state_step(state, action)
                # pos = self.env.grid.get(next_state[0], next_state[1])
                self.R[state][action] =  -1.0

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
    
    def step(self, state, action): # To interact with the environment. Used only for environment rendering. Removable?
        next_state, reward, done = self.act(state, action)
        for a in self.actions:
            if self.state_step(self.states[state], a) == self.states[next_state]:
                self.env.step(a)
        return next_state, reward, done

    def observation(self): 
        """Transform observation."""
        obs = (*self.env.agent_pos, self.env.agent_dir)
        return np.array(obs, dtype=np.int32)
    
    def render(self): #To simplify the rendering code
        image = self.env.render()
        plt.imshow(image)
        plt.show()
    
    def terminal(self, x: int) -> bool:
        state = self.states[x]
        pos = self.env.grid.get(state[0], state[1])
        at_goal = pos is not None and pos.type == "goal"
        return at_goal
    
    # Auxiliary Methods
    
    def print_attributes(self):
        print("Number of states: ", self.n_states)
        print("Number of actions: ", self.n_actions)
        print("Number of grid cells: ", self.n_cells)
    
    def print_environment(self):
        print("States: ", self.states)
        print("Actions: ", self.actions)
