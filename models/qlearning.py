import numpy as np

class QLearning:
    """
    Implements Q-learning algorithm with epsilon-greedy exploration

    If learning_rate is None; alpha(x,a) = 1/max(1, N(s,a))**alpha
    """
    def __init__(self, mdp, gamma=0.95, learning_rate=0.25, epsilon=1.0, epsilon_decay=0.9,
                 epsilon_min=0.05, learning_rate_decay = 0.99999, learning_rate_min = 0.0005, seed=42):
        self.mdp = mdp
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.learning_rate_min = learning_rate_min
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.Q = np.zeros((mdp.n_states, mdp.n_actions))
        self.Nsa = np.zeros((mdp.n_states, mdp.n_actions))
        self.state = self.mdp.s0
        self.RS = np.random.RandomState(seed)
        self.episode_end = False

    def get_delta(self, r, x, a, y):
        """
        :param r: reward
        :param x: current state
        :param a: current action
        :param y: next state
        :return:
        """
        max_q_y_a = self.Q[y, :].max()
        q_x_a = self.Q[x, a]

        return r + self.gamma * max_q_y_a - q_x_a

    def get_action(self, state):
        if self.RS.uniform(0, 1) < self.epsilon:
            # Explore
            return np.random.choice(self.mdp.n_actions)
        else:
            # Exploit
            return self.Q[state, :].argmax()
        
    def step(self):

        # Take mdp action
        a  = self.get_action(self.state)
        next_state, r, self.episode_end = self.mdp.act(self.state, a)

        # Get Delta Update
        delta = self.get_delta(r, self.state, a, next_state)

        # Update Q
        self.Q[self.state, a] = self.Q[self.state, a] + self.learning_rate * delta

        # Keep track of state-action visits
        self.Nsa[self.state, a] += 1
        self.state = next_state

        if self.episode_end:
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
            self.learning_rate = max(self.learning_rate * self.learning_rate_decay, self.learning_rate_min)
            self.state = self.mdp.s0

def Qlearning_training(qlearning, opt_lengths, n_steps=int(5e5)):
    tt = 0
    l0 = 0
    s0 = 0
    lengths = []
    throughputs = np.zeros(n_steps)

    Q_est = np.zeros((qlearning.mdp.n_states, qlearning.mdp.n_actions))
    while tt < n_steps:
        qlearning.step()
        # Store estimate of Q*
        Q_est = qlearning.Q
        tt +=1

        if qlearning.episode_end:
            lengths.append((tt-l0)/opt_lengths[s0])
            #throughputs[l0:tt] = 1/(tt-l0)
            throughputs[l0:tt] = opt_lengths[s0]/(tt-l0)
            l0 = tt
            s0 = qlearning.state

        if tt % 10000 == 0:
            print("Step: ", tt)
    
    if l0 != tt: throughputs[l0:tt] = throughputs[l0-1]

    # Compute greedy policy (with estimated Q)
    greedy_policy = np.argmax(qlearning.Q, axis=1)

    return Q_est, greedy_policy, lengths, throughputs

def print_Qlearning(Q_opt, Q_est, mdp):

    for state in range(mdp.n_states):
        print("state:", mdp.states[state])
        print("true: ", Q_opt[state, :])
        print("est: ", Q_est[state, :])
        print("----------------------------")
