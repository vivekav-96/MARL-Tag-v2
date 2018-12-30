#!/usr/bin/env python
import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))

import argparse
import time
import multiagent.scenarios as scenarios
from multiagent.environment import MultiAgentEnv
from bin.policies.dqn_policy import DQNPolicy

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-s', '--scenario', default='simple_tag.py', help='Path of the scenario Python script.')
    args = parser.parse_args()

    # load scenario from script
    scenario = scenarios.load(args.scenario).Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None,
                        shared_viewer=True)
    # render call to create viewer window (necessary only for interactive policies)
    env.render(mode='rgb_array')
    # create interactive policies for each agent
    policies = [DQNPolicy(env, args.scenario, i) for i in range(env.n)]
    # execution loop
    obs_n = env.reset()
    qwe = 0
    while qwe < 5:
        qwe += 1
        print('-----> ',qwe)
        # query for action from each agent's policy
        act_n = []
        for i, policy in enumerate(policies):
            act_n.append(policy.action(obs_n[i]))
        # step environment
        obs_n, reward_n, done_n, _ = env.step(act_n)
        env.render(mode='rgb_array')
        time.sleep(1)
        # display rewards
        # for agent in env.world.agents:
        #    print(agent.name + " reward: %0.3f" % env._get_reward(agent))

    for p in policies:
        p.save_network()
