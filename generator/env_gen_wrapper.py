import os
import gym
import gym_gvgai

import numpy as np

from generator.levels.base import Generator
from generator.levels.base import _initialize

class GridGame(gym.Wrapper):
    def __init__(self, game, play_length, path, lvl_name, mechanics=[], id=0):
        """Returns Grid instead of pixels
        Sets the reward
        Generates new level on reset
        --------
        """
        self.id = id
        self.name = game
        self.dir_path = path # gvgai.dir path + envs/games/zelda_v0
        self.lvl_path = os.path.join(path, lvl_name)
        self.mechanics = mechanics
        
        self.generator = Generator(tile_world=_initialize(os.path.join(path, lvl_name)), 
                                   path=path,
                                   mechanics=self.mechanics,
                                   generation=0)
        
        self.env = gym_gvgai.make('gvgai-{}-lvl0-v0'.format(game))
        gym.Wrapper.__init__(self, self.env)

        self.depth = None # gets set in self.reset()
        # env must exist to reset
        self.reset()

        self.steps = 0
        self.score = 0
        self.play_length = play_length
        
    def swap_generator(self, child):
        self.generator = child

    def reset(self):
        """reset gym simulation with whatever level the Generator currently holds
        """
        self.steps = 0
        self.score = 0
        # save file currently in generator to disk
        f = self.generator.to_file(self.id)
        # reset to just saved file
        state = self.set_level(f)
        return state
    
    def step(self, action):
        im, reward, done, info = self.env.step(action)
        if(self.steps >= self.play_length):
            done = True
            reward += 1 #reward the agent for living the same amount as for winning
            print(f"lived until end of env: +{reward}")
        
        #reward = self.get_reward(done, info["winner"], r) #extra r parameter
        
        state = info['grid'] # need to understand how Philip changes this below
        # state = self.get_state(info['grid'])
        
        self.steps += 1
        self.score += reward
        return state, reward, done, {'pic':im}
    
    def set_level(self, path_to_level):
        self.env.unwrapped._setLevel(path_to_level)
        self.env.reset()
        _, _, _, info = self.env.step(0) # do nothing
        state = info['grid']
        if self.depth is None:
            self.depth = state.shape[0] # for zelda shape is (13, 9, 13).
                                        #  The matrix shape is 9 x 13. So we want to extract the
                                        #  first element.
        return state
    
    def fitness(self, agent):
        """Score agent by having it try to complete the level.
        :param agent: NN-agent
        :return:
        """
        print("running agent")
        score = np.sum(agent.evaluate())
        self._fit = score / 100
    