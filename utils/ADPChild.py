import os
from agent.NNagent import NNagent
from generator.env_gen_wrapper import GridGame
from utils.ADPTASK_ENUM import ADPTASK

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

        self.mark_availability(self.alive)
        self.mark_availability(self.busy)

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

    def createFolders(self):
        for f in self.subfolders.keys():
            path = os.path.join(self.root, self.subfolders[f])
            if not os.path.exists(path):
                os.mkdir(path)

    def mark_availability(self, path):
        with open(path, 'w+') as f:
            pass

    def doTask(self, taskID):
        if taskID == ADPTASK.EVALUATE:
            rl = False
            if hasattr(self.args, 'rl'):
                rl = True
            score = self.pair.evaluate(rl=rl)
            return {
                'chromosome_id': self.chromosome_id,
                'score': score,
            }
        elif taskID == ADPTASK.OPTIMIZE:
            # run optimization here

            return {
                'chromosome_id': self.chromosome_id,
                'nn': self.pair.nn.state_dict()
            }
        else:
            raise ValueError('unspecified task requested')

    def parseRecievedTask(self):
        path = os.path.join(self.root,
                            self.subfolders['sent_by_parent'])

        task_params = load_obj(path, f'child{self.id}')
        self.pair.env.generator.update_from_lvl_string(task_params['lvl'])
        self.pair.nn.load_state_dict(task_params['nn'])
        self.chromosome_id = task_params['chromosome_id']
        self.args.update(task_params['kwargs'])

        return self.doTask(task_params['task_id'])

    def returnAnswer(self, answer):
        path = os.path.join(self.root,
                            self.subfolders['send_to_parent'])
        save_obj(answer, path, f'answer{self.id}')

    def recieveTaskAndReturnAnswer(self):
        self.mark_availability(self.busy)
        answer = self.parseRecievedTask()
        self.returnAnswer(answer)
        os.remove(self.busy)

    def __del__(self):
        self.pair.env.close()
        os.remove(self.busy)
        os.remove(self.alive)