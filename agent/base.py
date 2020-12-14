import os
import numpy as np
from utils.utils import add_noise

import torch
from torch.autograd import Variable
from torch.distributions import Categorical

class Agent:
    """
    Wrap each env with a game-playing agent
    """
    agent_count = 0
    def __init__(self, GG, time_stamp, prefix='.', master=True):
        """Wrap environment with a game-playing agent
        
        :param GG: GridGame Class (contains gym_gvgai env and a level generator)
        
        """
        
        self.unique_run_id = time_stamp
        self.prefix = prefix

        self._env = GG
        self.depth = GG.depth
        self.envs_through_time = []
        self.action_space = GG.env.action_space.n
        self.max_steps = GG.play_length
        # total reward of agent playing env
        self.max_achieved_score = 0
        self.score = 0
        self.noisy = False
        self.vis = None
        self.images = []        
        self.id = Agent.agent_count
        
        
        if master:
            Agent.agent_count += 1
            
            run_folder = f'{prefix}/results_{self.unique_run_id}/'
            
            if not os.path.exists(run_folder):
                os.mkdir(run_folder)
                
            agent_folder = os.path.join(run_folder, str(self.id))
            if not os.path.exists(agent_folder):
                os.mkdir(agent_folder)
            with open(f'{agent_folder}/lvl{self._env.id}.txt', 'w+') as fname:
                fname.write(str(self._env.generator))
        

    @property
    def env(self):
        return self._env

    
    def evaluate(self, env, rl=False):
        """Run self agent on current generator level. 
        """
        self.images = []
        # print("evaluating agent")
        done = False
        rewards = []
        state = add_noise(env.reset()) if self.noisy else env.reset()
        while not done:
            c = env.orientation[env.prev_move - 1]
            state = torch.DoubleTensor(np.array([state]))
            c = torch.DoubleTensor(np.array([c]))
            
            action, nlogpob, ent = self.get_action(state, c) if not rl else self.rl_get_action(state, c)

            # todo: FIGURE OUT HOW TO MAKE IT SO THAT I DON'T HAVE TO INT(ACTION).
            state, reward, done, info = env.step(int(action))
            if self.noisy:
                state = add_noise(state)
            # print(f"action: {action}, done: {done}, reward: {reward}")
            # state is a grid world here since we're using GridGame class
            rewards.append(reward)
            if self.vis:
                self.images.append(info['pic'])
                self.vis(env.env, action, image=info['pic'])

        self.won = info['won']

        # self.update_score(np.sum(rewards))
        # print("evaluated")
        # print(len(rewards))
        # if the user wants to do another noisy trial,
        # let them request it again.
        self.noisy = False
        return rewards

#     def update_score(self, potential_score):
#         """If new score is greater than previous score, update max_achieved_score

#         :param potential_score: total score of current agent playing env
#         :return:
#         """
#         if potential_score > self.max_achieved_score:
#             self.max_achieved_score = potential_score

    def mutate(self, mutationRate):
        childGG = self.env.mutate(mutationRate)
        return Agent(childGG)

    def get_action(self, state, c):
        # randomly for now
        probs = Categorical(probs=torch.Tensor([1/self.action_space for _ in range(self.action_space)]))
        action = probs.sample()
        return action, -probs.log_prob(action), probs.entropy()

    def rl_get_action(self, state, c):
        return self.get_action(state, c)
    
    def fitness(self, noisy=False, fn=None, rl=False):
        """run this agent through the current generator env once and store result into 
        """
        self.noisy = noisy
        self.vis = fn
        return self.env.fitness(self, rl=rl)

    def reset(self):
        return self.env.reset()
    
    def step(self, x):
        return self.env.step(x)
    
    def __str__(self):
        return str(self.env.generator)
    

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