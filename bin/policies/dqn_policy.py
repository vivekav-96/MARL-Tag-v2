import os

from keras import Sequential
from keras.engine.saving import model_from_json
from keras.layers import Dense

from multiagent.environment import MultiAgentEnv
from multiagent.policy import Policy


class DQNPolicy(Policy):
    env: MultiAgentEnv

    def __init__(self, env, scenario, agent_index):
        super().__init__()
        self.env = env
        self.agent_index = agent_index
        self.obs_space = env.observation_space[agent_index].shape[0]
        self.action_space = env.action_space[agent_index].n
        self.network_path = 'dqn_networks/{0}/network_obs{1}_act{2}/'.format(self.agent_index,
                                                                             self.obs_space, self.action_space)
        self.network = self.load_network()
        print('Action Space', self.action_space)
        print('Observation Space', self.obs_space)

    def action(self, obs):
        return [0, 0, 1, 0, 0]

    def train_network(self):
        pass

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
        network.compile(loss='mean_squared_error', optimizer='sgd')
        return network

    def create_dqn_network(self):
        """
        :return: DQN Agent's network

        Build the network and return it.
        """
        print('creating model ', self.network_path)
        model = Sequential()
        model.add(Dense(self.obs_space))
        model.add(Dense(self.obs_space * 2))
        model.add(Dense(self.obs_space * 3))
        model.add(Dense(self.obs_space * 2))
        model.add(Dense(self.action_space, activation='tanh'))
        return model
