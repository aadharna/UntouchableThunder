import os
from copy import deepcopy
from pprint import pprint

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
                 images=False,
                 shape=(9, 13)):

        """Returns Grid instead of pixels
        Sets the reward
        Generates new level on reset
        --------
        """
        self.game = game
        self.dir_path = path # gvgai.dir path + envs/games/zelda_v0
        self.lvl_path = os.path.join(path, lvl_name)
        self.mechanics = mechanics
        
        # if we do not have parsed location data on the sprites, read in a level and use that
        if not bool(locations):
            #set up first level, read it in from disk.
            lvl = _initialize(self.lvl_path)
            self.lvl_shape = lvl.shape
            self.generator = Generator(tile_world=lvl,
                                       shape=lvl.shape,
                                       path=path,
                                       mechanics=self.mechanics,
                                       generation=gen_id,
                                       locations=locations)
        
        # this condition will be used 99% of the time.
        else:
            # use generated lvl contained within locations dict.
            self.shape = shape
            self.generator = Generator(tile_world=None,
                                       shape=shape,
                                       path=path,
                                       mechanics=self.mechanics,
                                       generation=gen_id,
                                       locations=locations)
        # save to disk, can comment out if that lets me multithread.
        self.generator.to_file(GridGame.env_count, self.game)
        
        # return picture states?
        self.pics = images
        
        # self.env = gym_gvgai.make('gvgai-{}-lvl0-v0'.format(game))
        self.env = gym.make(f'gvgai-{game}-custom-v0',
                            level_data=str(self.generator),
                            pixel_observations=images,
                            tile_observations=True)

        # update static count of number of all envs
        self.id = GridGame.env_count
        GridGame.env_count += 1

        self.depth = None # gets set in self.reset()
        # env must exist to reset
        self.steps = 0
        self.score = 0
        self.play_length = play_length

        self.reset()
        
        self.orientation = np.eye(4, dtype=int)
        self.prev_move = 4
        self.rotating_actions = [1, 2, 3, 4]



    def reset(self):
        """reset gym simulation with whatever level the Generator currently holds
        """
        self.steps = 0
        self.score = 0
        self.prev_move = 4

        # save file currently in generator to disk
        s = str(self.generator)
        self.env.reset(environment_id=f'{self.game}-custom', level_data=s)
        if self.pics:
            (pix, state), _, _, _ = self.env.step(0)
        else:
            state, _, _, _ = self.env.step(0)
        state = np.transpose(state, (2, 0, 1))
        if self.depth is None:
            self.depth = state.shape[0]
        # print(state)
        return state

    def step(self, action):
        im = None
        if self.pics:
            (im, tile), reward, done, info = self.env.step(action)
        else:
            tile, reward, done, info = self.env.step(action)
        if done:
            pass
            #print(f"finished env with sc: {self.score + reward} \nenv: {self.game}_id:{self.id}_g:{self.generator.generation}")
        if self.steps >= self.play_length:
            done = True
        state = np.transpose(tile, (2, 0, 1))
        
        self.steps += 1
        self.score += reward
        
        # update orientation
        if action != self.prev_move and action in self.rotating_actions:
            self.prev_move = action
        
        return state, reward, done, {'pic': im}

    def mutate(self, mutationRate):
        new_map, shape = self.generator.mutate(mutationRate)

        childGG = GridGame(game=self.game,
                             play_length=self.play_length,
                             path=self.dir_path,
                             lvl_name=f"{self.game}_id:{self.id}_g:{self.generator.generation+1}.txt",
                             gen_id=self.generator.generation + 1,
                             mechanics=self.mechanics,
                             images=self.pics,
                             locations=new_map,
                             shape=shape)

        return childGG


    def fitness(self, agent):
        """Score THIS agent by having it try to complete THIS level.
        
        This function allows you to easily test any agent on an env that it's not paired with.
        (or it can be used to test an agent with it's own paired env if called through the agent's fitness fn).
        
        To test a non-paired agent with this env, call this function directly: PAIR.env.fitness(new_agent)
        
        :param agent: (NN-)agent
        :return:
        """
        # print(f"testing env {self.id} on agent {agent.id}")
        return np.sum(agent.evaluate(self))

    
    # To be able to pickle the GridGame
    def __getstate__(self):
        dictionary = {}
        for k, v in self.__dict__.items():
            # skip the gvgai env. But DO NOT DELETE IT.
            if not k == 'env':
                dictionary[k] = v
                
        dictionary['lvl_data'] = str(self.generator)
            
        return dictionary
    
    def __setstate__(self, d):
        self.__dict__ = d
        
        #since we skipped the gvgai env, we need to create a new one. 
        self.__dict__['env'] = gym.make(f'gvgai-{self.game}-custom-v0',
                                        level_data=self.lvl_data,
                                        pixel_observations=self.pics,
                                        tile_observations=True)
                
        
