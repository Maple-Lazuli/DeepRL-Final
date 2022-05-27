"""
Original Author: Ada Lazuli
Source modified from: https://github.com/Maple-Lazuli/Deep-RL/blob/main/src/train_agents.py
"""

import os.path
import os
import gym
import random
import torch
from collections import deque
from unityagents import UnityEnvironment
import numpy as np

import ddpg_agent_shared_memory as ddpg


def print_env_info(env):
    """
    Prints the properties of the environment, such as the action space, number of agents, and the state size.
    :param env: The environment for the learning task
    :return: None
    """
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
    print('The state for the first agent looks like:', states[0])


def main():
    """
    Instantiates the environment and agents, then trains the agents on the environment.
    :return: None
    """
    # Instantiate the environment and print metrics
    env = UnityEnvironment(file_name='../data/Tennis_Linux/Tennis.x86_64')
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]
    num_agents = len(env_info.agents)
    action_size = brain.vector_action_space_size

    agent_groups = [
        create_agent_group("ddpg1", ddpg.Agent, ddpg.ReplayBuffer, action_size, num_agents, buffer_size=int(1e5),
                           batch_size=512, modified_layers=True, weight_decay=0)
    ]
    epochs = 0
    while agent_group_below_min_score(agent_groups, 2.0):
        if epochs % 100 == 0:
            print(f"Starting epoch {epochs}")
        epochs += 1
        for agent_group in agent_groups:
            agent_group = train_episode(env, agent_group)
        write_group_performance(agent_groups)

        if epochs % 200 == 0:
            for agent_group in agent_groups:
                save_agent_group(agent_group)
                print(f"saved: {agent_group['name']}")



def agent_group_below_min_score(agent_groups, minimum):
    """
    Determine if the learning objective has been met.
    :param agent_groups: a list of agent groups, with each group consisting of 20 agents.
    :param minimum: The minimum score over the last 100 episodes the agents need to achieve.
    :return: Boolean
    """
    for agent_group in agent_groups:
        if len(agent_group['scores_deque']) < 100:
            continue
        elif np.mean(np.array(agent_group['scores_deque'])) <= minimum:
            continue
        else:
            print(
                f"{agent_group['name']} has scored above the minimum {minimum} with {np.mean(np.array(agent_group['scores_deque']))}")
            return False
    save_agent_group(agent_groups[0])
    return True


def create_agent_group(name, agent_constructor_fn, memory_constructor_fn, action_size, num_agents, buffer_size=int(1e6),
                       batch_size=64, gamma=0.99, tau=1e-3, lr_actor=1e-4, lr_critic=1e-3, weight_decay=0.0,
                       modified_layers=True):
    """
    Creates a group of 20 agents with shared memory and the specified hyperparameters from the function call
    """

    memory = [memory_constructor_fn(action_size, buffer_size=buffer_size, batch_size=batch_size, random_seed=2),
              memory_constructor_fn(action_size, buffer_size=buffer_size, batch_size=batch_size, random_seed=2)]

    agents = [agent_constructor_fn(state_size=24, action_size=action_size, random_seed=2, memory=memory[i], step_slot=i,
                                   buffer_size=buffer_size,
                                   batch_size=batch_size, gamma=gamma, tau=tau, lr_actor=lr_actor, lr_critic=lr_critic,
                                   weight_decay=0,
                                   modified_layers=modified_layers)
              for i in
              range(0, num_agents)]
    scores_deque = deque(maxlen=100)
    scores = []
    group_dict = {
        "name": name,
        "agents": agents,
        "scores": scores,
        "scores_deque": scores_deque,
        "scores_deque_trend":[],
        "episodes": 0
    }

    return group_dict


def train_episode(env, agent_group):
    """
    Train an agent group on the enviornment
    :param env: The environment to train the group in.
    :param agent_group: The agent group to train
    :return: Returns a reference to the agent group.
    """
    brain_name = env.brain_names[0]
    env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
    # reset the noise of agents in the group
    agents = agent_group['agents']

    [agent.reset() for agent in agents]
    scores_episode = np.zeros(len(agents))
    states = env_info.vector_observations
    steps = 0
    while True:
        steps += 1
        # give each agent a state and collect the actions
        actions = [agent.act(states[idx], add_noise=False) for idx, agent in enumerate(agents)]
        # take the actions chosen by the agents
        env_info = env.step(actions)[brain_name]
        # capture the new states following the actions
        next_states = env_info.vector_observations
        rewards = env_info.rewards
        dones = env_info.local_done
        [agent.step(states[idx], actions[idx], rewards[idx], next_states[idx], dones[idx]) for
         idx, agent in enumerate(agents)]
        states = next_states
        scores_episode += rewards
        if np.any(dones):  # exit loop if episode finished
            break


    # scores is a list containing all the episode scores
    agent_group["scores"].append(np.max(scores_episode))
    # scores deque contains the last 100 episode scores
    agent_group["scores_deque"].append(np.max(scores_episode))
    agent_group["scores_deque_trend"].append(np.mean(agent_group["scores_deque"]))
    agent_group['episodes'] += 1
    return agent_group


def write_group_performance(agent_groups):
    """
    Write the performance of the agent groups to a CSV in the reports directory.
    :param agent_groups: A list of agent groups to write the metrics for
    :return: None
    """
    num_episodes = 0
    header = ""
    for agent_group in agent_groups:
        header += f"{agent_group['name']}_score, {agent_group['name']}_deque_score_mean,{agent_group['name']}_deque_score_trend"
        num_episodes = max(num_episodes, agent_group['episodes'])
    header = header[:-1] + "\n"
    lines = [header]

    for episode in range(0, num_episodes):
        line = ""
        for agent_group in agent_groups:
            line += f"{agent_group['scores'][episode]},{np.mean(agent_group['scores_deque'])},{agent_group['scores_deque_trend'][episode]}"
        line = line[:-1] + "\n"
        lines.append(line)

    with open("../reports/data_dump/training.csv", "w") as file_out:
        for line in lines:
            file_out.write(line)


def save_agent_group(agent_group):
    """
    Save the checkpoints for an agent group in the saved_models directory
    :param agent_group: The agent group to save the weights for
    :return: None
    """
    save_dir = f"../saved_models/{agent_group['name']}"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    agents = agent_group['agents']

    [torch.save(agent.actor_local.state_dict(), save_dir + '/checkpoint_actor_' + str(idx) + '.pth') for idx, agent in
     enumerate(agents)]
    [torch.save(agent.critic_local.state_dict(), save_dir + '/checkpoint_critic_' + str(idx) + '.pth') for idx, agent in
     enumerate(agents)]


if __name__ == "__main__":
    main()
