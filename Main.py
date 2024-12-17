"""
    Name:
    Surname:
    Student ID:
"""

import os.path

import numpy as np

import rl_agents
from Environment import Environment
import time


GRID_DIR = "grid_worlds/"


if __name__ == "__main__":
    file_name = input("Enter file name: ")

    assert os.path.exists(os.path.join(GRID_DIR, file_name)), "Invalid File"

    env = Environment(os.path.join(GRID_DIR, file_name))

    # TODO: Type your parameters
    # Define the parameters for the QLearningAgent
    discount_rate = 0.9  # Discount factor for future rewards
    epsilon = 1.0  # Starting epsilon for epsilon-greedy policy (full exploration)
    epsilon_decay = 0.995  # Decay rate for epsilon (gradual decrease in exploration)
    epsilon_min = 0.1  # Minimum value for epsilon (prevents no exploration)
    alpha = 0.1  # Learning rate
    max_episode = 1000 # Maximum number of episodes to train the agent
    seed = 20

    # Initialize the QLearningAgent with the environment and hyperparameters
    agent = rl_agents.QLearningAgent(env,
                                     discount_rate=discount_rate,
                                     epsilon=epsilon,
                                     epsilon_decay=epsilon_decay,
                                     epsilon_min=epsilon_min,
                                     alpha=alpha,
                                     max_episode=max_episode,
                                     seed=seed)

    actions = ["UP", "LEFT", "DOWN", "RIGHT"]

    print("*" * 50)
    print()

    env.reset()

    start_time = time.time_ns()

    agent.train()

    end_time = time.time_ns()

    path, score = agent.validate()

    print("Actions:", [actions[i] for i in path])
    print("Score:", score)
    print("Elapsed Time (ms):", (end_time - start_time) * 1e-3)

    print("*" * 50)
