import os
import time
from torch import save as torch_save

from agent.OrientationConditionedNNagent import OrientationConditionedNNagent
from generator.env_gen_wrapper import GridGame
from optimization.Optimizer import PyTorchObjective
from utils.ADPTASK_ENUM import ADPTASK

from optimization.runners import run_TJ_DE, run_ppo, run_CoDE, run_DE
# from devo import jDE, DE, CoDE

from utils.utils import save_obj, load_obj
from utils.loader import load_from_yaml
import logging
#logging.basicConfig(filename='example.log',level=logging.DEBUG)

#from memory_profiler import profile

class ADPChild:
    def __init__(self, child_id,
                 args_file='./args.yml',
                 game='dzelda',
                 length=250,
                 lvl_dir='./levels',
                 init_lvl='start.txt',
                 prefix='.'):

        self.prefix = prefix
        self.root = os.path.join(prefix, 'communication')
        self.subfolders = {
            'sent_by_parent': 'outgoing',
            'send_to_parent': 'incoming',
            'alive_signals': 'alive',
            'available_signals': 'available'
        }
        # self.createFolders()

        self.id = child_id
        self.alive = os.path.join(self.root,
                                  self.subfolders['alive_signals'],
                                  f'{self.id}.txt')

        self.available = os.path.join(self.root,
                                      self.subfolders['available_signals'],
                                      f'{self.id}.txt')

        # this pair is for useage by children. It does not count as a POET pair.
        self.pair = \
            OrientationConditionedNNagent(time_stamp=None,
                                          prefix=prefix,
                                          GG=GridGame(game=game,
                                args_file=args_file,
                                prefix=prefix,
                                play_length=length,
                                path=lvl_dir,
                                lvl_name=init_lvl,
                                mechanics=['+', 'g'],  # doesn't matter.
                                # monsters, key, door, wall
                                ),
                                          master=False,  # won't use time_stamp or prefix
                                          )
        

        self.args = load_from_yaml(args_file)

        self.game_length = self.pair.env.play_length

        self.placeChildFlag(self.available)
        self.placeChildFlag(self.alive)
        #logging.debug(f"child {self.id} alive signal sent")
        
    ########### END CONSTRUCTOR #############




    def hasTask(self):
        path = os.path.join(self.root,
                            self.subfolders['sent_by_parent'],
                            f'child{self.id}') + '.pkl'
        if os.path.exists(path):
            # print(f"child{self.id} has a task")
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
    
    #@profile
    def doTask(self, run_id, nn, lvl, task_id, chromosome_id, env_id, rl, poet_loop_counter,
               noisy=False,
               algo='CoDE',
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
                'won': self.pair.env.win == 'Win',
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
                        path               = f'{self.pair.prefix}/runs',
                        pair_id            = chromosome_id,
                        outer_poet_loop_count= poet_loop_counter,
                        n_concurrent_games = 1,
                        frames             = ngames * self.game_length)
            else:
                objective = PyTorchObjective(agent=self.pair, popsize=popsize)
                ans = run_DE(AE_pair=objective,
                                results_prefix=self.prefix,
                                unique_run_id=run_id, 
                                pair_id=chromosome_id, 
                                poet_loop_counter=poet_loop_counter,
                                generation_max=ngames // popsize,
                                scaling_factor=0.6,
                                crossover_rate=0.4,
                                lower_bound=-5,
                                upper_bound=5)
                objective.update_nn(ans)
                del objective
            
            # get score of optimized weights
            score = self.pair.evaluate(rl=rl)
            state_dict = self.pair.nn.state_dict()

            # save best weights
            #destination = f"{self.pair.prefix}/results_{run_id}/{chromosome_id}/final_weights_poet{poet_loop_counter}.pt"
            #torch_save(state_dict, destination)
            # did the agent WIN the game?
            if self.pair.env.win == 'Win':
                path = os.path.join(f'{self.prefix}', 
                                    f'{chromosome_id}',
                                    f'winning_weights_poet{poet_loop_counter}.pt')
                torch_save(state_dict, path)
            return {
                'won': self.pair.env.win == 'Win',
                'score': score,
                'chromosome_id': chromosome_id,
                'env_id': env_id,
                'nn': state_dict
            }
        else:
            raise ValueError('unspecified task requested')
    
    #@profile
    def parseRecievedTask(self, task_params):
        """
        THIS is MAIN. When the child recieves a task, it enters here!
        :return:
        """
        
        # have some communcation go to java each time so that the server does more than just sleep
        # self.pair.env.reset()
        # print(task_params)

        
        lvls = task_params['lvls']
        nns = task_params['nns']
        chromosome_ids = task_params['chromosome_ids']
        env_ids = task_params['env_ids']
        diffs = task_params['diff']
        kwargs = task_params['kwargs']
        task_id = task_params['task_id']
        poet_loop_counter = task_params['poet'] # int
        run_id = task_params['run_id']

        answers = {}
        if bool(nns):
            for i in range(len(nns)):
                nn = nns[i]
                lvl = lvls[i]
                chromosome_id = chromosome_ids[i]
                diff = diffs[i]

                # update base path to the id of the pair being used
                self.pair.env.generator._path = os.path.join(self.prefix, str(chromosome_id))
                self.pair.env.generator.env_id = chromosome_id
                self.pair.env.generator.diff = diff

                env_id = env_ids[i]

                # key word args
                rl = False
                ngames = 100000
                popsize = 1000
                algo = 'CoDE'
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

                answer = self.doTask(run_id, nn, lvl, task_id, chromosome_id, env_id, rl, poet_loop_counter,
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
        
        path = os.path.join(self.root,
                            self.subfolders['sent_by_parent'])

        task_params = load_obj(path, f'child{self.id}.pkl')
        os.remove(os.path.join(path, f'child{self.id}') + '.pkl')
        
        if "resend" in task_params:
            print("asking for work to be resent")
            self.placeChildFlag(os.path.join(self.root, self.subfolders['send_to_parent'], f'resend{self.id}.txt'))
            time.sleep(10)
            self.placeChildFlag(self.available)
            return
        
        try:
            answer = self.parseRecievedTask(task_params)
        except ConnectionResetError as e:
            # die gracefully here.
            print(f"{self.id} died")
            os.remove(self.alive)
            self.placeChildFlag(os.path.join(self.root, self.subfolders['send_to_parent'], f'dead{self.id}.txt'))
            return

        self.returnAnswer(answer)
        del answer
        self.placeChildFlag(self.available)
        print('waiting')

#     def __del__(self):
#         if os.path.exists(self.available):
#             os.remove(self.available)
#         self.pair.env.close()
