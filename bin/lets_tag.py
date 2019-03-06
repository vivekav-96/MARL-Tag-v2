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
TRAIN_FOR_EPISODES = 50
TRAINING_MODE = True
BENCHMARK_DATA = True

RUNNER_CAPTURED_MSG = 'Runner Has Been Captured !'
RUNNER_ESCAPED_MSG = 'Runner Escaped !'


def is_collision(agent1, agent2):
    delta_pos = agent1.state.p_pos - agent2.state.p_pos
    dist = np.sqrt(np.sum(np.square(delta_pos)))
    dist_min = agent1.size + agent2.size
    return True if dist < dist_min else False


def game_end_callback(agent, world):
    global iterations
    if iterations >= GAME_ITERATION_LIMIT:
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
    label = pyglet.text.Label(RUNNER_CAPTURED_MSG if iterations < GAME_ITERATION_LIMIT else RUNNER_ESCAPED_MSG,
                              font_name='Times New Roman',
                              font_size=20,
                              x=window.width // 2, y=window.height // 2, width=window.width // 2,
                              anchor_x='center', anchor_y='center', multiline=True)

    @window.event
    def on_draw():
        window.clear()
        label.draw()

    pyglet.app.run()


def start_a_game():
    # Clear memory of each agent before a game.
    for p in policies:
        p.clear_memory()

    state_n = env.reset()

    global iterations
    iterations = 0

    # execution loop
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

        if not TRAINING_MODE:
            print('-------------------------------------------------------------------------------------------')
            print('Episode : {}, Iteration : {}'.format(episode, iterations))
        for i, p in enumerate(policies):
            p.add_memory(Experience(state_n[i], act_n[i], reward_n[i], next_state_n[i], done_n[i]))
            loss = p.adapt()
            if not TRAINING_MODE:
                print('{0} got reward {1}. Trained with loss {2:.4f}'.format(agents[i].name, reward_n[i], loss))

        if any(done_n):
            save_policy_networks(policies)
            if not TRAINING_MODE:
                show_game_over_dialog()
            break

        state_n = next_state_n

        env.render(mode='rgb_array')

        # CheckPointing : Save networks every N iterations
        if iterations % CHECKPOINT_ITERATIONS == 0:
            save_policy_networks(policies)

    save_policy_networks(policies)
    return iterations


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
    policies = [DQNPolicy(env, SCENARIO, i, agents[i].adversary) for i in range(env.n)]

    episode = 0
    iterations = 0
    if TRAINING_MODE:
        while episode < TRAIN_FOR_EPISODES:
            episode += 1
            start_a_game()
            msg = RUNNER_CAPTURED_MSG if iterations < GAME_ITERATION_LIMIT else RUNNER_ESCAPED_MSG
            print('Episode {} over. {}'.format(episode, msg))
    else:
        start_a_game()

    if BENCHMARK_DATA:
        for p in policies:
            p.benchmark()
