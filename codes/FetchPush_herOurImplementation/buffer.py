import numpy as np

class ReplayBuffer:
    def __init__(self, max_size, input_dims, n_actions, n_goals): #n_action means component of the action space since it is continuos
        self.mem_size = max_size #as we exceed, we'll overwrite eralieest memory with new one
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_dims))
        self.new_state_memory = np.zeros((self.mem_size, *input_dims))
        self.action_memory = np.zeros((self.mem_size, 4))
        self.reward_memory = np.zeros(self.mem_size) #array of floating point numbers
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool) #true or false
        self.goal_memory = np.zeros((self.mem_size, 3))
    def store_transition(self, state, action, reward, state_, done, goal):
        index = self.mem_cntr % self.mem_size #modulo of the division 7 % 2 gives 1

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done #
        self.goal_memory[index] = goal
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size) 
#used to know how much memory we've filled up
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]
        goals = self.goal_memory[batch]

        return states, states_, actions, rewards, dones, goals
