import os
import gym
import gvgai

import numpy as np

from generator.levels.base import Generator
from generator.levels.base import _initialize

class GridGame(gym.Wrapper):
    # static variable. Increment when new GG objs are created
    # and use value as part of unique id.
    env_count = 0
    def __init__(self,
                 game,
                 play_length,
                 path='/home/aadharna/miniconda3/envs/thesis/lib/python3.7/site-packages/GVGAI_GYM/gym_gvgai' + '/envs/games/zelda_v0/',
                 lvl_name='zelda_lvl0.txt',
                 mechanics=[],
                 locations={},
                 gen_id=0,
                 images=False):

        """Returns Grid instead of pixels
        Sets the reward
        Generates new level on reset
        --------
        """
        self.name = game
        self.dir_path = path # gvgai.dir path + envs/games/zelda_v0
        self.lvl_path = os.path.join(path, lvl_name)
        self.mechanics = mechanics

        self.generator = Generator(tile_world=_initialize(self.lvl_path),
                                   path=path,
                                   mechanics=self.mechanics,
                                   generation=gen_id,
                                   locations=locations)
        self.pics = images
        
        # self.env = gym_gvgai.make('gvgai-{}-lvl0-v0'.format(game))
        self.env = gym.make(f'gvgai-{game}-custom-v0',
                            level_data=str(self.generator),
                            pixel_observations=images,
                            tile_observations=True)

        # update static count of number of all envs
        self.id = GridGame.env_count
        GridGame.env_count += 1
        gym.Wrapper.__init__(self, self.env)

        self.depth = None # gets set in self.reset()
        # env must exist to reset
        self.steps = 0
        self.score = 0
        self.play_length = play_length

        self.reset()



    def reset(self):
        """reset gym simulation with whatever level the Generator currently holds
        """
        self.steps = 0
        self.score = 0
        # save file currently in generator to disk
        s = str(self.generator)
        self.env.reset(environment_id=f'{self.name}-custom', level_data=s)
        if self.pics:
            (pix, state), _, _, _ = self.env.step(0)
        else:
            state, _, _, _ = self.env.step(0)
        state = np.transpose(state, (2, 0, 1))
        if self.depth is None:
            self.depth = state.shape[0]
        return state

    def step(self, action):
        im = None
        if self.pics:
            (im, tile), reward, done, info = self.env.step(action)
        else:
            tile, reward, done, info = self.env.step(action)
        if done:
            print(f"solved env with sc: {self.score + reward}")
        if self.steps >= self.play_length:
            done = True
        state = np.transpose(tile, (2, 0, 1))
        
        self.steps += 1
        self.score += reward
        return state, reward, done, {'pic': im}

    def mutate(self, mutationRate):
        new_map = self.generator.mutate(mutationRate)

        childGG = GridGame(game=self.game,
                             play_length=self.play_length,
                             path=self.dir_path,
                             lvl_name=self.lvl_path,
                             gen_id=self.generator.generation + 1,
                             mechanics=self.mechanics,
                             locations=new_map)

        return childGG


    def fitness(self, agent):
        """Score agent by having it try to complete the level.
        :param agent: (NN-)agent
        :return:
        """
        print("running agent")
        score = np.sum(agent.evaluate())
        self._fit = score / 100

