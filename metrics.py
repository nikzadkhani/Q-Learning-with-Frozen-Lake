import gym
import numpy as np
from tqdm import trange


def win_rate(Q_table: np.array, env, num_episodes: int):
    """ Calculate the win rate of a given policy in a given number of episodes and
    :param Q_table: a 2-D numpy array with size num_states by num_actions
    :param env: unwrapped gym environment
    :param num_episodes: the number of episodes to calculate the win rate over
    :return: a float with the calculated win rate
    """
    m, n = Q_table.shape
    assert(m == env.observation_space.n)
    assert(n == env.action_space.n)

    num_wins = 0
    num_time_outs = 0

    for _ in trange(num_episodes):
        current_state = env.reset()
        done = False
        num_steps = 0
        while not done:
            action = np.argmax(Q_table[current_state, :])
            next_state, reward, done, _ = env.step(action)
            num_wins += reward
            current_state = next_state
            num_steps += 1
    print(num_time_outs)
    return num_wins/num_episodes


def test_q_table(q_table, num_trials=5000):
  num_wins = 0
  num_losses = 0

  env = gym.make('FrozenLake-v0')
  for _ in range(num_trials):
    done = False
    observation = env.reset()
    while not done:
      action = np.argmax(q_table[observation])
      observation, _, done, _ = env.step(action)

    if observation == 15:
      num_wins += 1
    else:
      num_losses += 1

  print("Win Rate", num_wins/num_trials, "Loss Rate", num_losses/num_trials)
