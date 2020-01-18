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
        self.chromosome_id = None
        self.alive = os.path.join(self.root,
                                  self.subfolders['alive_signals'],
                                  f'{self.id}.txt')

        self.busy = os.path.join(self.root,
                                  self.subfolders['busy_signals'],
                                  f'{self.id}.txt')

        self.placeChildFlag(self.alive)
        print(f"child {self.id} alive signal sent")
        # self.placeChildFlag(self.busy)

        self.args = {}

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

    def doTask(self, taskID):
        """
        Execute the asked for task

        :param taskID: EVALUATE the NN or OPTIMIZE it
        :return: result dict
        """
        rl = False
        if hasattr(self.args, 'rl'):
            rl = self.args['rl']

        if taskID == ADPTASK.EVALUATE:
            score = self.pair.evaluate(rl=rl)
            return {
                'chromosome_id': self.chromosome_id,
                'score': score,
            }

        elif taskID == ADPTASK.OPTIMIZE:
            # run optimization here
            if rl:
                # optimizes in place
                run_ppo(policy_agent       = self.pair,
                        env_fn             = self.pair.env.make,
                        path               = './runs',
                        n_concurrent_games = 1,
                        frames             = 100000)
            else:
                objective = PyTorchObjective(agent=self.pair)
                # run_TJ_DE(_de=devo.jDE,
                #           pair=objective,
                #           n=10000)
                # objective.update_nn(objective.best_individual)
            score = self.pair.evaluate(rl=rl)
            return {
                'score': score,
                'chromosome_id': self.chromosome_id,
                'nn': self.pair.nn.state_dict()
            }
        else:
            raise ValueError('unspecified task requested')

    def parseRecievedTask(self):
        path = os.path.join(self.root,
                            self.subfolders['sent_by_parent'])

        task_params = load_obj(path, f'child{self.id}.pkl')
        os.remove(os.path.join(path, f'child{self.id}') + '.pkl')

        # update network and env to execute on task
        self.pair.env.generator.update_from_lvl_string(task_params['lvl'])
        self.pair.nn.load_state_dict(task_params['nn'])
        # save id of PAIR to pass back
        self.chromosome_id = task_params['chromosome_id']
        # get additional arguments passed in
        self.args.update(task_params['kwargs'])

        # execute the asked for task
        return self.doTask(task_params['task_id'])

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
        os.remove(self.busy)
        self.pair.env.close()

