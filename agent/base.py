import numpy as np
from utils.utils import add_noise

class Agent:
    """
    Wrap each env with a game-playing agent
    """
    agent_count = 0
    def __init__(self, GG, max_steps=250):
        """Wrap environment with a game-playing agent
        
        :param GG: GridGame Class (contains gym_gvgai env and a level generator)
        
        """

        self._env = GG
        self.envs_through_time = []
        self.action_space = GG.env.env.action_space.n
        self.max_steps = max_steps
        # total reward of agent playing env
        self.max_achieved_score = 0
        self.noisy = False
        self.vis = None
        self.images = []        
        self.id = Agent.agent_count
        Agent.agent_count += 1
        

    @property
    def env(self):
        return self._env

    @property
    def name(self):
        return f'{self.env.game}_{self.env.id}_{self.env.generator.generation}'

    def evaluate(self, env):
        """Run self agent on current generator level. 
        """
        # print("evaluating agent")
        done = False
        rewards = []
        state = add_noise(env.reset()) if self.noisy else env.reset()
        while not done:
            action = self.get_action(state)
            state, reward, done, info = env.step(action)
            if self.noisy:
                state = add_noise(state)
            # print(f"action: {action}, done: {done}, reward: {reward}")
            # state is a grid world here since we're using GridGame class
            rewards.append(reward)
            if self.vis:
                self.images.append(info['pic'])
                self.vis(env.env, action, image=info['pic'])

        self.update_score(np.sum(rewards))
        # print("evaluated")
        # print(len(rewards))
        # if the user wants to do another noisy trial,
        # let them request it again.
        self.noisy = False
        return rewards

    def update_score(self, potential_score):
        """If new score is greater than previous score, update max_achieved_score

        :param potential_score: total score of current agent playing env
        :return:
        """
        if potential_score > self.max_achieved_score:
            self.max_achieved_score = potential_score

    def mutate(self, mutationRate):
        childGG = self.env.mutate(mutationRate)
        return Agent(childGG)

    def get_action(self, state):
        # randomly for now
        return np.random.choice(self.action_space)
    
    def fitness(self, noisy=False, fn=None):
        """run this agent through the current generator env once and store result into 
        """
        self.noisy = noisy
        self.vis = fn
        return self.env.fitness(self)

    def reset(self):
        return self.env.reset()
    
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