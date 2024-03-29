import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork

class Agent:
    def __init__(self, input_dims, n_goals, n_actions, alpha=0.001, beta=0.002, env=None,
            gamma=0.99,  max_size=1000000, tau=0.005, 
            fc1=400, fc2=300, batch_size=64, noise=0.1):
    #alpha-->learning rate actor net, beta = lr critic network
    #gamma discount factor, tau for soft update target nets 
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_goals, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.n_goals = n_goals
        self.noise = noise
        self.max_action = env.action_space.high[0]
        self.min_action = env.action_space.low[0]
        
        self.actor = ActorNetwork(n_actions=n_actions, name='actor')
        self.critic = CriticNetwork( name='critic')
        self.target_actor = ActorNetwork(n_actions=n_actions, name='target_actor')
        self.target_critic = CriticNetwork(name='target_critic')

        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic.compile(optimizer=Adam(learning_rate=beta))
        self.target_actor.compile(optimizer=Adam(learning_rate=alpha)) #weights are soft updated, just a tf need
        self.target_critic.compile(optimizer=Adam(learning_rate=beta)) #weights ////////

        self.update_network_parameters(tau=1)

    def update_network_parameters(self, tau=None):
        if tau is None: #hard copy 
            tau = self.tau # that is equal 1

        weights = [] #for tager actor net
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_actor.set_weights(weights)

        weights = [] #for target critic net
        targets = self.target_critic.weights
        for i, weight in enumerate(self.critic.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_critic.set_weights(weights)

    def remember(self, state, action, reward, new_state, done, goal):
        self.memory.store_transition(state, action, reward, new_state, done, goal)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_weights(self.actor.checkpoint_file)
        self.target_actor.save_weights(self.target_actor.checkpoint_file)
        self.critic.save_weights(self.critic.checkpoint_file)
        self.target_critic.save_weights(self.target_critic.checkpoint_file)

    def load_models(self):
        print('... loading models ...')
        self.actor.load_weights(self.actor.checkpoint_file)
        self.target_actor.load_weights(self.target_actor.checkpoint_file)
        self.critic.load_weights(self.critic.checkpoint_file)
        self.target_critic.load_weights(self.target_critic.checkpoint_file)

    def choose_action(self, state, goal, evaluate=False):
        chosen_state_goal = np.concatenate((state,goal), axis=0)
        chosen_state_goal = tf.convert_to_tensor(np.array(chosen_state_goal)[np.newaxis,:], dtype=tf.float32)
        chosen_actions = self.actor(chosen_state_goal)
        if not evaluate: #this if adds noise to the actions, to be added always when training
            chosen_actions += tf.random.normal(shape=[self.n_actions],
                    mean=0.0, stddev=self.noise)
        # note that if the environment has an action > 1, we have to multiply by
        # max action at some point
        chosen_actions = tf.clip_by_value(chosen_actions, self.min_action, self.max_action)

        return chosen_actions[0]

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, new_state, action, reward, done, goal = \
                self.memory.sample_buffer(self.batch_size)
        
        states_goal = tf.convert_to_tensor(np.concatenate((state, goal), axis=1) ,dtype=tf.float32)
        states_goal_ = tf.convert_to_tensor(np.concatenate((new_state, goal), axis=1) ,dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)
        dones = tf.convert_to_tensor(done, dtype= tf.bool)
        
        
        with tf.GradientTape() as tape: #critic update
            
            target_actions = self.target_actor(states_goal_)
            critic_value_ = tf.squeeze(self.target_critic(
                                states_goal_, target_actions), 1)
            critic_value = tf.squeeze(self.critic(states_goal, actions), 1)
            target = reward + self.gamma*critic_value_*(1-done)
            critic_loss = keras.losses.MSE(target, critic_value)

        critic_network_gradient = tape.gradient(critic_loss,
                                            self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(
            critic_network_gradient, self.critic.trainable_variables))

        with tf.GradientTape() as tape: #actor update
            new_policy_actions = self.actor(states_goal)
            actor_loss = -self.critic(states_goal, new_policy_actions) #gradient ascent
            actor_loss = tf.math.reduce_mean(actor_loss)

        actor_network_gradient = tape.gradient(actor_loss, 
                                    self.actor.trainable_variables)#
        #gradient of actorloss wrt trainable variables
        self.actor.optimizer.apply_gradients(zip(
            actor_network_gradient, self.actor.trainable_variables))

        self.update_network_parameters()
