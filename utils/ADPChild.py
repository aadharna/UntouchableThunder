import os
from agent.NNagent import NNagent
from generator.env_gen_wrapper import GridGame
from optimization.Optimizer import PyTorchObjective
from utils.ADPTASK_ENUM import ADPTASK

from optimization.runners import run_TJ_DE, run_ppo
# from devo import jDE, DE, CoDE

from utils.utils import save_obj, load_obj

class ADPChild:
    def __init__(self, child_id):
        self.root = os.path.join('.', 'communication')
        self.subfolders = {
            'sent_by_parent': 'outgoing',
            'send_to_parent': 'incoming',
            'alive_signals': 'alive',
            'busy_signals': 'busy'
        }
        self.createFolders()

        self.id = child_id

        self.alive = os.path.join(self.root,
                                  self.subfolders['alive_signals'],
                                  f'{self.id}.txt')

        self.busy = os.path.join(self.root,
                                  self.subfolders['busy_signals'],
                                  f'{self.id}.txt')

        self.placeChildFlag(self.alive)
        print(f"child {self.id} alive signal sent")
        # self.placeChildFlag(self.busy)

        self.pair = \
            NNagent(
                GridGame(game='zelda',
                        play_length=250,
                        path='./levels',
                        lvl_name='start.txt',
                        mechanics=['+', 'g'],
                        # monsters, key, door, wall
                        )
               )

        self.game_length = self.pair.env.play_length

    ########### END CONSTRUCTOR #############




    def hasTask(self):
        path = os.path.join(self.root,
                            self.subfolders['sent_by_parent'],
                            f'child{self.id}') + '.pkl'
        if os.path.exists(path):
            print(f"child{self.id} has a task")
            return True
        return False

    # create folders if parent has failed to do so.
    def createFolders(self):
        for f in self.subfolders.keys():
            path = os.path.join(self.root, self.subfolders[f])
            if not os.path.exists(path):
                os.mkdir(path)

    def placeChildFlag(self, path):
        with open(path, 'w+') as f:
            pass

    def doTask(self, nn, lvl, task_id, chromosome_id, rl,
               algo='jDE',
               ngames=1000,
               popsize=100):
        """

        :param nn: PyTorch nn state_dict
        :param lvl: flat lvl string
        :param task_id: EVALUATE the NN or OPTIMIZE it
        :param chromosome_id: id of NN-GG pair
        :param rl: use RL?
        :return:
        """

        # update network and env to execute on task
        self.pair.env.generator.update_from_lvl_string(lvl)
        self.pair.nn.load_state_dict(nn)

        if task_id == ADPTASK.EVALUATE:
            score = self.pair.evaluate(rl=rl)
            return {
                'chromosome_id': chromosome_id,
                'score': score,
            }

        elif task_id == ADPTASK.OPTIMIZE:
            # run optimization here
            if rl:
                # optimizes in place
                run_ppo(policy_agent       = self.pair,
                        env_fn             = self.pair.env.make,
                        path               = './runs',
                        n_concurrent_games = 1,
                        frames             = ngames * self.game_length)
            else:
                objective = PyTorchObjective(agent=self.pair, popsize=popsize)
                # run_TJ_DE(_de=self.algo,
                #           pair=objective,
                #           n=ngames)
                # objective.update_nn(objective.best_individual)
            score = self.pair.evaluate(rl=rl)
            return {
                'score': score,
                'chromosome_id': chromosome_id,
                'nn': self.pair.nn.state_dict()
            }
        else:
            raise ValueError('unspecified task requested')

    def parseRecievedTask(self):
        path = os.path.join(self.root,
                            self.subfolders['sent_by_parent'])

        task_params = load_obj(path, f'child{self.id}.pkl')
        os.remove(os.path.join(path, f'child{self.id}') + '.pkl')

        lvls = task_params['lvls']
        nns = task_params['nns']
        chromosome_ids = task_params['chromosome_ids']
        kwargs = task_params['kwargs']
        task_ids = task_params['task_ids']

        answers = {}

        for i in range(len(nns)):
            nn = nns[i]
            lvl = lvls[i]
            task_id = task_ids[i]
            chromosome_id = chromosome_ids[i]

            # key word args
            rl = False
            ngames = 1000
            popsize = 100
            algo='jDE'
            if hasattr(kwargs, 'rl'):
                rl = kwargs['rl'][i]

            if hasattr(kwargs, 'ngames'):
                ngames = kwargs['ngames'][i]

            if not rl and task_id == ADPTASK.OPTIMIZE:
                if hasattr(kwargs, 'algo'):
                    algo = kwargs['algo'][i]
                if hasattr(kwargs, 'popsize'):
                    popsize = kwargs['popsize'][i]


            answers[chromosome_id] = self.doTask(nn, lvl, task_id, chromosome_id, rl,
                                                 algo=algo,
                                                 ngames=ngames,
                                                 popsize=popsize)

        # execute the asked for task
        import pdb
        pdb.set_trace()
        return answers

    def returnAnswer(self, answer):
        path = os.path.join(self.root,
                            self.subfolders['send_to_parent'])
        save_obj(answer, path, f'answer{self.id}')

    def recieveTaskAndReturnAnswer(self):
        self.placeChildFlag(self.busy)
        answer = self.parseRecievedTask()
        self.returnAnswer(answer)
        os.remove(self.busy)

    def __del__(self):
        os.remove(self.alive)
        if os.path.exists(self.busy):
            os.remove(self.busy)
        self.pair.env.close()

