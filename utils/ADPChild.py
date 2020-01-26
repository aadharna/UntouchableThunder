import os
from torch import save as torch_save

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
            'available_signals': 'available'
        }
        self.createFolders()

        self.id = child_id

        self.alive = os.path.join(self.root,
                                  self.subfolders['alive_signals'],
                                  f'{self.id}.txt')

        self.available = os.path.join(self.root,
                                      self.subfolders['available_signals'],
                                      f'{self.id}.txt')

        # this pair is for useage by children. It does not count as a POET pair.
        self.pair = \
            NNagent(
                GridGame(game='zelda',
                        play_length=250,
                        path='./levels',
                        lvl_name='start.txt',
                        mechanics=['+', 'g'],
                        # monsters, key, door, wall
                        ),
                master=False,
               )

        self.game_length = self.pair.env.play_length

        self.placeChildFlag(self.available)
        self.placeChildFlag(self.alive)
        print(f"child {self.id} alive signal sent")
        
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

    def doTask(self, nn, lvl, task_id, chromosome_id, env_id, rl, poet_loop_counter,
               noisy=False,
               algo='jDE',
               ngames=1000,
               popsize=100):
        """

        :param nn: PyTorch nn state_dict
        :param lvl: flat lvl string
        :param task_id: EVALUATE the NN or OPTIMIZE it
        :param chromosome_id: id of NN-GG pair
        :param rl: use RL?
        :param poet_loop_counter: poet number loop
        :return:
        """
        
        # update network and env to execute on task
        self.pair.env.generator.update_from_lvl_string(lvl)
        self.pair.nn.load_state_dict(nn)

        if task_id == ADPTASK.EVALUATE:
            self.pair.noisy = noisy
            score = self.pair.evaluate(rl=rl)
            return {
                'chromosome_id': chromosome_id,
                'env_id': env_id,
                'score': score,
            }

        elif task_id == ADPTASK.OPTIMIZE:
            # run optimization here
            if rl:
                # optimizes in place
                run_ppo(policy_agent       = self.pair,
                        env_fn             = self.pair.env.make,
                        path               = './runs',
                        pair_id            = chromosome_id,
                        outer_poet_loop_count= poet_loop_counter,
                        n_concurrent_games = 1,
                        frames             = ngames * self.game_length)
            else:
                objective = PyTorchObjective(agent=self.pair, popsize=popsize)
                run_TJ_DE(opt_name          = algo,
                          pair              = objective,
                          n                 = ngames,
                          pair_id           = chromosome_id,
                          poet_loop_counter = poet_loop_counter)
                objective.update_nn(objective.best_individual)
            
            # get score of optimized weights
            score = self.pair.evaluate(rl=rl)
            state_dict = self.pair.nn.state_dict()
            
            # save best weights
            torch_save(state_dict, f"./results/{chromosome_id}/final_weights_poet{poet_loop_counter}.pt")
            return {
                'score': score,
                'chromosome_id': chromosome_id,
                'env_id': env_id,
                'nn': state_dict
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
        env_ids = task_params['env_ids']
        kwargs = task_params['kwargs']
        task_id = task_params['task_id']
        poet_loop_counter = task_params['poet'] # int
        
        answers = {}
        if bool(nns):
            for i in range(len(nns)):
                nn = nns[i]
                lvl = lvls[i]
                chromosome_id = chromosome_ids[i]
                env_id = env_ids[i]

                # key word args
                rl = False
                ngames = 1000
                popsize = 100
                algo = 'jDE'
                noisy = False

                if 'rl' in kwargs:
                    rl = kwargs['rl']

                if 'noisy' in kwargs:
                    noisy = kwargs['noisy']

                if 'ngames' in kwargs:
                    ngames = kwargs['ngames']

                if not rl and task_id == ADPTASK.OPTIMIZE:
                    if 'algo' in kwargs:
                        algo = kwargs['algo']
                    if 'popsize' in kwargs:
                        popsize = kwargs['popsize']

                answer = self.doTask(nn, lvl, task_id, chromosome_id, env_id, rl, poet_loop_counter,
                                     noisy=noisy,
                                     algo=algo,
                                     ngames=ngames,
                                     popsize=popsize)

                if chromosome_id not in answers:
                    answers[chromosome_id] = []

                answers[chromosome_id].append(answer)

        # execute the asked for task
        # import pdb
        # pdb.set_trace()
        return answers

    def returnAnswer(self, answer):
        path = os.path.join(self.root,
                            self.subfolders['send_to_parent'])
        save_obj(answer, path, f'answer{self.id}')

    def recieveTaskAndReturnAnswer(self):
        answer = self.parseRecievedTask()
        self.returnAnswer(answer)
        self.placeChildFlag(self.available)
        print('waiting')

#     def __del__(self):
#         if os.path.exists(self.available):
#             os.remove(self.available)
#         self.pair.env.close()

