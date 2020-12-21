import os

class BaseAgent:
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
        self.action_space = GG.env.action_space.n
        self.max_steps = GG.play_length
        # total reward of agent playing env
        self.max_achieved_score = 0
        self.score = 0
        self.noisy = False
        self.vis = None
        self.images = []        
        self.id = BaseAgent.agent_count
        
        
        if master:
            BaseAgent.agent_count += 1
            
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

    def evaluate(self, env, **kwargs):
        raise NotImplementedError

    def mutate(self, mutationRate):
        raise NotImplementedError

    def get_action(self, state, c):
        raise NotImplementedError

    def rl_get_action(self, state, c):
        raise NotImplementedError
    
    def fitness(self, noisy=False, fn=None, rl=False):
        """run this agent through the current generator env once and store result into 
        """
        raise NotImplementedError

    def reset(self):
        raise self.env.reset()
    
    def step(self, x):
        raise self.env.step(x)
    
    def __str__(self):
        raise str(self.env.generator)
