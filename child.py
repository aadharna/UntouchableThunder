import os
import time
from utils.ADPChild import ADPChild
from utils.loader import load_from_yaml

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--id", type=int, help='set child id')
parser.add_argument("--args_file", type=str, default='./args.yml', help='path to args file')

line_args = parser.parse_args()
file_args = load_from_yaml(line_args.args_file)

print(line_args)
print(file_args)

if __name__ == "__main__":

    child = ADPChild(line_args.id,
                     game=file_args.game,
                     length=file_args.game_len,
                     lvl_dir=file_args.lvl_dir,
                     prefix=file_args.result_prefix)
    done = False
    while not done:
        try:
            while not child.hasTask():
                # if you're waiting for a task and 
                # your alive flag is removed by the parent
                # kill yourself. 
                if not os.path.exists(child.alive):
                    done = True
                    break
                    
                time.sleep(5)
            
            if not done:
                print("found task")
                child.recieveTaskAndReturnAnswer()
            
            

        except KeyboardInterrupt as e:
            print(e)
            child.pair.env.close()
            import sys
            sys.exit(0)

        # done = True

    child.pair.env.close()
