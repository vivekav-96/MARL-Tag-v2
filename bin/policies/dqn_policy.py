import os
import random

import numpy as np

from keras import Sequential
from keras.engine.saving import model_from_json
from keras.layers import Dense

from multiagent.environment import MultiAgentEnv
from multiagent.policy import Policy

from bin.misc.experience import Experience


class DQNPolicy(Policy):
    env: MultiAgentEnv
    EPSILON_VALUE = 0.25
    REPLACE_TARG_NW_ITERS = 400

    def __init__(self, env, scenario, agent_index):
        super().__init__()
        self.gamma = 0.9
        self.learning_step = 0
        self.env = env
        self.scenario = scenario
        self.agent_index = agent_index
        self.obs_space = env.observation_space[agent_index].shape[0]
        self.action_space = env.action_space[agent_index].n
        self.network_path = 'dqn_networks/{0}/{1}/network_obs{2}_act{3}/'.format(self.scenario, self.agent_index,
                                                                                 self.obs_space, self.action_space)
        self.eval_network = self.load_network()
        self.target_network = self.load_network()
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
        self.learning_step += 1
        if self.learning_step % DQNPolicy.REPLACE_TARG_NW_ITERS == 0:
            self.update_target_weights()

        batch = random.sample(self.memory, min(32, len(self.memory)))

        states = np.asarray([sample.state for sample in batch])
        states_next = np.asarray([sample.next_state for sample in batch])
        done = np.asarray([sample.done for sample in batch])
        actions = np.asarray([sample.action for sample in batch])
        rewards = np.asarray([sample.reward for sample in batch])
        not_done = np.logical_not(done)
        rows = np.arange(done.shape[0])

        eval_next = self.eval_network.predict(states_next)
        target_next = self.target_network.predict(states_next)
        discounted_rewards = self.gamma * target_next[rows, np.argmax(eval_next, axis=1)]

        y = self.eval_network.predict(states)
        y[rows, np.argmax(actions, axis=1)] = rewards
        y[not_done, np.argmax(actions[not_done], axis=1)] += discounted_rewards[not_done]

        fit_result = self.eval_network.fit(np.asarray(states), np.asarray(y), verbose=0)
        return fit_result.history['loss'][0]

    def update_target_weights(self):
        self.target_network.set_weights(self.eval_network.get_weights())

    def action(self, obs):
        r = random.random()
        if r < DQNPolicy.EPSILON_VALUE:
            return random.randint(0, self.action_space-1)
        obs = np.asarray(obs)
        obs = obs.reshape((1,) + obs.shape)
        res = self.eval_network.predict([obs])
        action = np.argmax(res[0])
        return action

    def save_network(self):
        if not os.path.exists(self.network_path):
            os.makedirs(self.network_path)
        nw_json = self.eval_network.to_json()
        with open(self.network_path + 'model.json', 'w') as json_file:
            json_file.write(nw_json)
        self.eval_network.save_weights(self.network_path + 'model.h5')

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
        model.add(Dense(150, activation='linear'))
        model.add(Dense(75, activation='linear'))
        model.add(Dense(self.action_space, activation='linear'))
        return model
