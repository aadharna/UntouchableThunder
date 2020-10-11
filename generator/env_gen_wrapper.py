import os
import gym
import gvgai #needs to be here for gvgai envs
import numpy as np

from generator.levels.EvolutionaryGenerator import EvolutionaryGenerator
from generator.levels.IlluminatingGenerator import IlluminatingGenerator
from generator.levels.base import _initialize

from utils.loader import load_from_yaml

class GridGame(gym.Wrapper):
    # static variable. Increment when new GG objs are created
    # and use value as part of unique id.
    env_count = 0

    def __init__(self,
                 game,
                 play_length,
                 prefix='..',
                 args_file='./args.yml',
                 path='./levels',
                 lvl_name='start.txt',
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
        self.args_file = args_file
        self.prefix = prefix
        self.game = game
        self.dir_path = path # gvgai.dir path + envs/games/zelda_v0
        self.lvl_name = lvl_name
        self.lvl_path = os.path.join(path, lvl_name)
        self.mechanics = mechanics

        self.args = load_from_yaml(args_file)

        if self.args.generatorType  == "evolutionary":
        
            # if we do not have parsed location data on the sprites, read in a level and use that
            if not bool(locations):
                #set up first level, read it in from disk.
                lvl = _initialize(self.lvl_path, d=shape[0])
                self.lvl_shape = lvl.shape
                self.generator = EvolutionaryGenerator(game=self.game,
                                                       args_file=args_file,
                                                       tile_world=lvl,
                                                       shape=lvl.shape,
                                                       path=path,
                                                       mechanics=self.mechanics,
                                                       generation=gen_id,
                                                       locations=locations)

            # this condition will be used 99% of the time.
            else:
                # use generated lvl contained within locations dict.
                self.lvl_shape = shape
                self.generator = EvolutionaryGenerator(game=self.game,
                                                       args_file=args_file,
                                                       tile_world=None,
                                                       shape=shape,
                                                       path=path,
                                                       mechanics=self.mechanics,
                                                       generation=gen_id,
                                                       locations=locations)
            # save to disk, can comment out if that lets me multithread.
            # self.generator.to_file(GridGame.env_count, self.game)

        elif self.args.generatorType == "illuminating":
            self.generator = IlluminatingGenerator(shape=shape,
                                                   args_file=args_file,
                                                   path=path,
                                                   generation=gen_id,
                                                   run_folder=prefix)


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
        self.done = False
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

        if self.pics:
            (pix, state) = self.env.reset(environment_id=f'{self.game}-custom', level_data=s)
        else:
            state = self.env.reset(environment_id=f'{self.game}-custom', level_data=s)

        if self.depth is None:
            self.depth = state.shape[2] #shape is (9, 13, 13) Going to be reshaped to (13, 9, 13). 
        # print(state)
        return np.transpose(state, (2, 0, 1))

    def step(self, action):
        im = None
        if self.pics:
            (im, tile), reward, done, info = self.env.step(action)
        else:
            tile, reward, done, info = self.env.step(action)

        if self.steps >= self.play_length:
            done = True
            
        state = np.transpose(tile, (2, 0, 1))
        
        self.steps += 1
        reward -= 1e-4       # punish just randomly moving around
                             # This should add some gradient signal.
        self.score += reward
        self.done = info['winner']

        if self.args.no_score:
            if self.done == 3:
                reward = 1 - (self.steps / self.args.game_len)
            elif self.done == 2:
                reward = (self.steps / self.args.game_len) - 1
            else:
                reward = 0

        # update orientation
        if action != self.prev_move and action in self.rotating_actions:
            self.prev_move = action
        
        return state, reward, done, {'pic': im, 'won': info['winner']}

    def mutate(self, mutationRate):
        new_map, shape = self.generator.mutate(mutationRate)
        childGG = GridGame(game=self.game,
                             play_length=self.play_length,
                             args_file=self.args_file,
                             path=self.dir_path,
                             lvl_name=f"{self.game}_id:{self.id}_g:{self.generator.generation+1}.txt",
                             gen_id=self.generator.generation + 1,
                             mechanics=self.mechanics,
                             images=self.pics,
                             locations=new_map,
                             shape=shape)
        return childGG


    def fitness(self, agent, rl=False):
        """Score THIS agent by having it try to complete THIS level.
        
        This function allows you to easily test any agent on an env that it's not paired with.
        (or it can be used to test an agent with it's own paired env if called through the agent's fitness fn).
        
        To test a non-paired agent with this env, call this function directly: PAIR.env.fitness(new_agent)
        
        :param agent: (NN-)agent
        :return:
        """
        # print(f"testing env {self.id} on agent {agent.id}")
        return np.sum(agent.evaluate(self, rl=rl))

    
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
        # NOTE: THIS IS SUPER-FUCKING-EXPENSIVE
        self.__dict__['env'] = gym.make(f'gvgai-{self.game}-custom-v0',
                                        level_data=self.lvl_data,
                                        pixel_observations=self.pics,
                                        tile_observations=True)
        
        
    # for use in vec_envs
    def make(self):
        def _make():
            return GridGame(game=self.game,
                             play_length=self.play_length,
                             path=os.path.join(self.dir_path),
                             lvl_name=f"{self.game}_id:{self.id}_g:{self.generator.generation+1}.txt",
                             gen_id=self.generator.generation + 1,
                             mechanics=self.mechanics,
                             images=self.pics,
                             locations=self.generator.locations,
                             shape=self.lvl_shape)
        return _make
    
    def close(self):
        self.env.stop()
                
    
