from utils.ADPParent import ADPParent
from utils.ADPTASK_ENUM import ADPTASK

from agent.NNagent import NNagent
from generator.env_gen_wrapper import GridGame

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("n", type=int, default=1)
args = parser.parse_args()


parent = ADPParent(args.n)


pair = NNagent(GridGame(game='zelda',
                        play_length=250,
                        path='./levels',
                        lvl_name='start.txt',
                        mechanics=['+', 'g'],
                        # monsters, key, door, wall
                        )
               )

parent.registerChildren()


parent.createChildTask(pair.nn, pair.env, ADPTASK.EVALUATE)




pair.env.close()