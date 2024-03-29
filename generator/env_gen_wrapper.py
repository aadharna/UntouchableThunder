import os
import gym
from gym.spaces import MultiDiscrete, Discrete

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
                 game=None,
                 play_length=None,
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
        self.args = load_from_yaml(args_file)
        self.prefix = prefix
        
        self.game = game if game is not None else self.args.game 
        self.dir_path = path # gvgai.dir path + envs/games/zelda_v0
        self.lvl_name = lvl_name
        self.lvl_path = os.path.join(path, lvl_name)
        self.mechanics = mechanics if not bool(mechanics) else self.args.mechanics
        
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

        if self.args.engine == 'GDY':
            from griddly import gd
            from griddly import GymWrapperFactory
            wrapper = GymWrapperFactory()

            if self.pics:
                self.observer = gd.ObserverType.SPRITE_2D
            else:
                self.observer = gd.ObserverType.VECTOR
            print(self.observer)
            try:
                wrapper.build_gym_from_yaml(
                    f'{game}-custom',
                    os.path.join(self.dir_path, f'{self.game}.yaml'),
                    level=0,
                    global_observer_type=self.observer,
                    player_observer_type=self.observer
                )
            except gym.error.Error:
                pass

        else:
            raise ValueError("gvgai is not supported anymore. Please use Griddly.")


        self.env = gym.make(f"{self.args.engine}-{game}-custom-v0")
        _ = self.env.reset()
        # count the number of distinct discrete actions
        # subtract the zero sub-action from each unique action
        # add back in a zero action at the end
        # THIS ASSUMES ACTION SPACES ARE DISCRETE
        actionType = self.env.action_space
        if type(actionType) == MultiDiscrete:
            self.nActions = 1
            for d in actionType.nvec:
                self.nActions += (d - 1)
        elif type(actionType) == Discrete:
            self.nActions = actionType.n
        else:
            raise ValueError(f"Unsupported action type in game: {self.game}. "
                             f"Only Discrete and MultiDiscrete are supported")

        self.env.enable_history(True)

        # update static count of number of all envs
        self.id = GridGame.env_count
        GridGame.env_count += 1

        self.depth = None # gets set in self.reset()
        # env must exist to reset
        self.win = False
        self.steps = 0
        self.score = 0
        self.play_length = play_length if play_length is not None else self.args.game_len

        self.info_list = []
        self.reset()
        
        self.orientation = np.eye(4, dtype=int)
        self.prev_move = 4
        self.rotating_actions = [1, 2, 3, 4]


    def reset(self):
        """reset gym simulation with whatever level the Generator currently holds
        """
        self.steps = 0
        self.score = 0
        self.win = False
        self.prev_move = 4
        self.info_list.clear()
        # import time
        # time.sleep(0.5)

        # save file currently in generator to disk
        badgen = True
        while badgen:
            try:
                s = str(self.generator)
                # this is a global observation
                state = self.env.reset(level_string=s, global_observations=True)
                self.env.enable_history(True)
                badgen = False
            except ValueError as e:
                if self.args.generatorType == "illuminating":
                    continue
                else:
                    raise e

        # after you have a state, get the conv-depth
        if self.depth is None:
            self.depth = state['player'].shape[0]

        return state['player']

    def step(self, action):

        state, reward, done, info = self.env.step(action)
        # self.env.render(observer='global')
        if self.steps >= self.play_length:
            done = True
        
        self.steps += 1
        self.score += reward
        if "PlayerResults" in info:
            self.win = info['PlayerResults']['1']
            # print(f"set win to: {self.win}")

        if self.args.no_score:
            if self.win == 'Win':
                reward = 1 - (self.steps / self.play_length)
            elif self.win == 'Lose':
                reward = (self.steps / self.play_length) - 1
            else:
                reward = 0

        info['pic'] = state if self.pics else None
        info['won'] = self.win
        self.info_list.append(info)

        # update orientation
        # if action != self.prev_move and action in self.rotating_actions:
        #     self.prev_move = action

        return state, reward, done, info

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

        if self.args.engine == 'GDY':
            from griddly import gd
            from griddly import GymWrapperFactory
            wrapper = GymWrapperFactory()

            if self.pics:
                self.observer = gd.ObserverType.SPRITE_2D
            else:
                self.observer = gd.ObserverType.VECTOR
            # print(self.observer)
            try:
                wrapper.build_gym_from_yaml(
                    f'{self.game}-custom',
                    os.path.join(self.dir_path, f'{self.game}.yaml'),
                    level=0,
                    global_observer_type=self.observer,
                    player_observer_type=self.observer
                )
            except gym.error.Error:
                pass

        else:
            raise ValueError("gvgai is not supported anymore. Please use Griddly.")

        self.__dict__['env'] = gym.make(f'{self.args.engine}-{self.game}-custom-v0')
        
        
    # for use in vec_envs
    def make(self):
        def _make():
            return GridGame(game=self.game,
                             play_length=self.play_length,
                             path=self.dir_path,
                             lvl_name=self.lvl_name,
                             gen_id=self.generator.generation,
                             mechanics=self.mechanics,
                             images=self.pics,
                             locations=self.generator.locations,
                             shape=self.lvl_shape)
        return _make
    
    def close(self):
        self.env.close()
                
    
