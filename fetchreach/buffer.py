import numpy as np

class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions): #n_action means component of the action space since it is continuos
        self.mem_size = max_size #as we exceed, we'll overwrite eralieest memory with new one
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size) #array of floating point numbers
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool) #true or false

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size #modulo of the division 7 % 2 gives 1

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done #

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

        return states, actions, rewards, states_, dones


np.random.choice(7,1, replace= False)
