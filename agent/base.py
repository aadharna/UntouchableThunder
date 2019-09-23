import numpy as np
import pandas as pd
from generator.levels.base import Generator

class Agent:
    """
    Wrap each env with a game-playing agent
    """
    def __init__(self, GG, max_steps=250):
        """Wrap environment with a game-playing agent
        
        :param GG: GridGame Class (contains gym_gvgai env and a level generator.
        
        """

        self._env = GG
        self.envs_through_time = []
        self.action_space = GG.env.env.action_space.n
        self.max_steps = max_steps

    @property
    def env(self):
        return self._env

    def evaluate(self):
        """Run self agent on current generator level. 
        """
        print("evaluating agent")
        done = False
        rewards = []
        self._env.reset()
        step = 0
        while not done:
            action = self.get_action()
            state, reward, done, info = self._env.step(action)
            print(f"step: {step}, action: {action}, done: {done}, reward: {reward}")
            # state is a grid world here since we're using GridGame class
            step += 1
            rewards.append(reward)
        return rewards

    def update(self):
        pass

    def get_action(self):
        # randomly for now
        return np.random.choice(self.action_space)
    
    def fitness(self):
        """run this agent through the current generator env once and store result into 
            agent.env._fit
        """
        return self.env.fitness(self)

def simulate(model, level, max_steps=250, n_episodes=5):
    """Run this agent on this level n_episodes times and get reward for each run
    """
    # use property env rather than private _env
    model.env.set_level(level)

    # track information
    total_rewards = []
    for _ in range(n_episodes):
        episode_reward = model.evaluate()
        total_rewards.append(episode_reward)

    return total_rewards