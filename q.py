import gym
import numpy as np
from tqdm import trange


def q_learning(env, random_init: bool = True, alpha: float = 0.5, gamma: float = 0.95,
               epsilon: float = 0.1, num_episodes: int = 500):
    """ Given some environment perform Q-learning
    :param env: Unwrapped OpenAI gym environment
    :param random_init: Whether to use random values or zeroes for
    the q-table initialization
    :param alpha: the learning rate
    :param gamma: the discount factor of
    :param epsilon: the percentage of time to choose a random action
    :param num_episodes: the number of episodes to run the training loop
    :param terminal_states: a # terminal states by 2 vector containing the state row number
    in the first column and and the value to set the expected reward for that row
    :return: 2-D numpy array containing the Q-table
    """
    np.random.seed(42)
    env.seed(42)

    num_states = env.observation_space.n  # gets number of states
    num_actions = env.action_space.n     # gets number of actions

    if random_init:
        q_table = np.random.random(
            (num_states, num_actions))  # Random initialization
    else:
        # Zero initialization
        q_table = np.zeros((num_states, num_actions))

    for _ in trange(num_episodes):
        done = False
        observation = env.reset()

        while not done:
            if np.random.rand() > epsilon:
                action = np.argmax(q_table[observation])  # Use Q-value
            else:
                action = env.action_space.sample()       # Christopher Columbus

            # Get new observation, take the action
            new_observation, reward, done, _ = env.step(action)

            # Update the Q-table
            reward + gamma * np.max(q_table[new_observation])
            q_table[observation, action] = (1 - alpha) * q_table[observation, action]\
                + alpha * (reward + gamma * np.max(q_table[new_observation]))

            observation = new_observation


    return q_table
