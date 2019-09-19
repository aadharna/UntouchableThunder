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
        pass

    def update(self):
        pass

    def get_action(self):
        # randomly for now
        return np.random.choice(self.action_space)

# will move this when I figure out where it goes
def generate_level_from_rules(rules):
    pass

def simulate(model, rules, n_episodes=5):
    level = generate_level_from_rules(rules) #returns path to txt/npy file
    model.set_level(level)

    # track information
    total_rewards = []
    total_reward = 0
    for _ in range(n_episodes):
        model.env.reset()
        done = False
        while not done:
            action = model.get_action()
            obs, reward, done, info = model.env.step(action)
            total_reward += reward
        total_rewards.append(total_reward)

    return total_rewards