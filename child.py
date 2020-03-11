import os
import time
from utils.ADPChild import ADPChild
from utils.loader import load_from_yaml

import logging

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--id", type=int, help='set child id')
parser.add_argument("--exp_name", type=str, help='exp name')
parser.add_argument("--args_file", type=str, default='./args.yml', help='path to args file')

line_args = parser.parse_args()
file_args = load_from_yaml(line_args.args_file)

print(line_args)
print(file_args)

# logging.basicConfig(filename=f'example.log',level=logging.DEBUG)


if __name__ == "__main__":

    child = ADPChild(line_args.id,
                     game=file_args.game,
                     args_file=line_args.args_file,
                     length=file_args.game_len,
                     lvl_dir=file_args.lvl_dir,
                     init_lvl=f"{file_args.game}_{file_args.init_lvl}",
                     prefix=f"{file_args.result_prefix}/results_{line_args.exp_name}")

    #logging.debug(f"child {child.id} alive signal sent")
    
    start = time.time()

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
                
                current_time = time.time()
                if (current_time - start) // 3600 >= 10:
                    print("close to dying")
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
    #check for id.txt.done files. If so, do NOT launch sbatch command.
    # placeChildFlag(os.path.join(self.root, self.subfolders['send_to_parent'], f'dead{self.id}.txt') 
    if os.path.exists(child.alive):
        os.remove(child.alive)
    if os.path.exists(os.path.join(child.root, child.subfolders['alive_signals'], f'{child.id}.txt.done')):
        print("task completely finished")

    else:
        print(f"refreshing worker {child.id}")
        os.system(f'sbatch run-workers.sbatch {line_args.exp_name} {line_args.args_file} {line_args.id}')


    child.pair.env.close()
