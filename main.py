import gym
import numpy as np
from q import q_learning
from metrics import test_q_table

NUM_TRAIN_EPISODES = 50_000
NUM_TEST_EPISODES = 1_000


def main():
    env = gym.make('FrozenLake-v0')

    print("Started Training")
    q_table = q_learning(env, random_init=False, num_episodes=NUM_TRAIN_EPISODES)
    print("Finished Training")

    print("\nQ_table:\n", q_table, "\n")
    test_q_table(q_table)


if __name__ == '__main__':
    main()
