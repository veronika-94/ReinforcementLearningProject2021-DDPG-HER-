import os
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
import numpy as np

class CriticNetwork(keras.Model):
    def __init__(self, fc1_dims=512, fc2_dims=512,
            name='critic', chkpt_dir='./models'): #mkdir temp
        super(CriticNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, 
                    self.model_name+'_ddpg.h5') 

        self.fc1 = Dense(self.fc1_dims, activation='relu') #outputs fc1_dims
        self.fc2 = Dense(self.fc2_dims, activation='relu') #outputs fc2_dims
        self.q = Dense(1, activation=None)#outputs q value

    def call(self, state_goal, action): #forward propagation
        action_value = self.fc1(tf.concat([state_goal, action], axis=1))
        action_value = self.fc2(action_value)

        q = self.q(action_value)

        return q

class ActorNetwork(keras.Model):
    def __init__(self, n_actions, fc1_dims=512, fc2_dims=512, name='actor',
            chkpt_dir='./models'):
        super(ActorNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, 
                    self.model_name+'_ddpg.h5')

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.mu = Dense(self.n_actions, activation='tanh')

    def call(self, state_goal):
        prob = self.fc1(state_goal)
        prob = self.fc2(prob)

        mu = self.mu(prob)

        return mu
