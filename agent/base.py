import numpy as np
import pandas as pd

class Agent:
    """
    Wrap each env with a game-playing agent
    """
    def __init__(self, env, action_space):
        self._env = env
        self.envs_through_time = []
        self.action_space = action_space

    @property
    def env(self):
        return self._env

    def set_level(self, path_to_level):
        self.envs_through_time.append(path_to_level)
        self._env.unwrapped._setLevel(path_to_level)
        self._env.reset()

    def evaluate(self):
        done = False
        rewards = []
        self._env.reset()
        while not done: 
            action = self.get_action()
            state, reward, done, info = self._env.step(action)
            rewards.append(reward)
        return rewards

    def update(self):
        pass

    def get_action(self):
        # randomly for now
        return np.random.choice(self.action_space)

def simulate(model, level, n_episodes=5):
    model.set_level(level)

    # track information
    total_rewards = []
    for _ in range(n_episodes):
        episode_reward = model.evaluate()
        total_rewards.append(sum(episode_reward))

    return total_rewards