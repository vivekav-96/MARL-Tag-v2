import os
import random

import numpy as np

from keras import Sequential
from keras.engine.saving import model_from_json
from keras.layers import Dense
from keras.optimizers import SGD

from multiagent.environment import MultiAgentEnv
from multiagent.policy import Policy

from bin.misc.experience import Experience


class DQNPolicy(Policy):
    env: MultiAgentEnv
    EPSILON_VALUE = 0.25

    def __init__(self, env, scenario, agent_index):
        super().__init__()
        self.env = env
        self.scenario = scenario
        self.agent_index = agent_index
        self.obs_space = env.observation_space[agent_index].shape[0]
        self.action_space = env.action_space[agent_index].n
        self.network_path = 'dqn_networks/{0}/{1}/network_obs{2}_act{3}/'.format(self.scenario, self.agent_index,
                                                                                 self.obs_space, self.action_space)
        self.network = self.load_network()
        self.memory = []

    def add_memory(self, experience):
        """
        :type experience: Experience
        """
        self.memory.append(experience)

    def clear_memory(self):
        self.memory.clear()

    def adapt(self):
        """
        Method to train DQN network
        :return: Training accuracy
        """
        batch = random.sample(self.memory, min(32, len(self.memory)))
        x = [sample.state for sample in batch]
        y = []
        for sample in batch:
            target = sample.action
            target[np.argmax(sample.action)] += sample.reward
            y.append(target)
        self.network.fit(np.asarray(x), np.asarray(y), verbose=0)

    def action(self, obs):
        r = random.random()
        if r < DQNPolicy.EPSILON_VALUE:
            return random.randint(0, self.action_space-1)
        obs = np.asarray(obs)
        obs = obs.reshape((1,) + obs.shape)
        res = self.network.predict([obs])
        action = np.argmax(res[0])
        return action

    def save_network(self):
        if not os.path.exists(self.network_path):
            os.makedirs(self.network_path)
        nw_json = self.network.to_json()
        with open(self.network_path + 'model.json', 'w') as json_file:
            json_file.write(nw_json)
        self.network.save_weights(self.network_path + 'model.h5')

    def load_network(self):
        """
        :return: Load and return the network if present else create and return one.
        """
        if os.path.exists(self.network_path):
            model_json = self.network_path + 'model.json'
            model_weights = self.network_path + 'model.h5'

            if os.path.exists(model_json) and os.path.exists(model_weights):
                # load json and create model
                with open(model_json, 'r') as json_file:
                    loaded_model_json = json_file.read()
                network = model_from_json(loaded_model_json)
                network.load_weights(model_weights)
            else:
                network = self.create_dqn_network()
        else:
            network = self.create_dqn_network()
        network.compile(loss='mean_squared_error', optimizer='adam')
        return network

    def create_dqn_network(self):
        """
        :return: DQN Agent's network

        Build the network and return it.
        """
        print('creating model ', self.network_path)
        model = Sequential()
        model.add(Dense(50, input_dim=self.obs_space, activation='relu'))
        model.add(Dense(100, activation='relu'))
        # model.add(Dense(self.obs_space * 3))
        # model.add(Dense(self.obs_space * 2))
        model.add(Dense(self.action_space, activation='linear'))
        return model
