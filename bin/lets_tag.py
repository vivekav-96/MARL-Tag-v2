#!/usr/bin/env python
import os
import sys

from bin.misc.experience import Experience

sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np
import multiagent.scenarios as scenarios
from multiagent.environment import MultiAgentEnv
from bin.policies.dqn_policy import DQNPolicy

SCENARIO = 'simple_marl_tag.py'

RUNNER_SPEED = 0.3
CHASER_SPEED = 0.25
CHECKPOINT_ITERATIONS = 50
GAME_ITERATION_LIMIT = 500


def is_collision(agent1, agent2):
    delta_pos = agent1.state.p_pos - agent2.state.p_pos
    dist = np.sqrt(np.sum(np.square(delta_pos)))
    dist_min = agent1.size + agent2.size
    return True if dist < dist_min else False


def game_end_callback(agent, world):
    if iterations > GAME_ITERATION_LIMIT:
        return True

    for a in world.agents:
        if a == agent:
            continue
        else:
            if agent.adversary:
                if not a.adversary:
                    if is_collision(agent, a):
                        return True
            else:
                if a.adversary:
                    if is_collision(agent, a):
                        return True
    return False


def save_policy_networks(policies):
    for p in policies:
        p.save_network()


def show_game_over_dialog():
    import pyglet

    window = pyglet.window.Window(width=250, height=125, caption='Game Over')
    label = pyglet.text.Label('Runner Has Been Captured !' if iterations < GAME_ITERATION_LIMIT
                              else 'Runner Escaped !',
                              font_name='Times New Roman',
                              font_size=20,
                              x=window.width // 2, y=window.height // 2, width=window.width // 2,
                              anchor_x='center', anchor_y='center', multiline=True)

    @window.event
    def on_draw():
        window.clear()
        label.draw()

    pyglet.app.run()


if __name__ == '__main__':

    # load scenario from script
    scenario = scenarios.load(SCENARIO).Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None,
                        done_callback=game_end_callback,
                        shared_viewer=True)
    env.seed(312)

    agents = env.agents
    policies = [DQNPolicy(env, SCENARIO, i) for i in range(env.n)]

    experiences = []

    # execution loop
    state_n = env.reset()
    iterations = 0
    while True:
        iterations += 1
        # query for action from each agent's policy
        act_n = []
        for policy in policies:
            action = policy.action(state_n[policy.agent_index])
            action_one_hot = np.zeros(policy.action_space)
            action_one_hot[action] = RUNNER_SPEED if agents[policy.agent_index] else CHASER_SPEED
            act_n.append(action_one_hot)

        # step environment
        next_state_n, reward_n, done_n, _ = env.step(act_n)

        for i, p in enumerate(policies):
            print('{} got reward {}'.format(agents[i].name, reward_n[i]))
            p.add_memory(Experience(state_n[i], act_n[i], reward_n[i], next_state_n[i], done_n[i]))
        print('------------------------------------------------------------------------------------------------------')

        for p in policies:
            p.adapt()

        if any(done_n):
            save_policy_networks(policies)
            show_game_over_dialog()
            break

        state_n = next_state_n

        env.render(mode='rgb_array')

        # CheckPointing : Save networks every 100 iterations
        if iterations % CHECKPOINT_ITERATIONS == 0:
            save_policy_networks(policies)

    save_policy_networks(policies)
