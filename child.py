import os
import time
import subprocess
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


time.sleep(10)

if __name__ == "__main__":
    print("starting up")
    child = ADPChild(line_args.id,
                     game=file_args.game,
                     args_file=line_args.args_file,
                     length=file_args.game_len,
                     lvl_dir=file_args.lvl_dir,
                     init_lvl=f"{file_args.game}_{file_args.init_lvl}",
                     prefix=os.path.join(file_args.result_prefix, f"results_{line_args.exp_name}"))

    #logging.debug(f"child {child.id} alive signal sent")
    
    start = time.time()
    print(start)
    done = False
    while not done:
        try:
            while not child.hasTask():
                # if you're waiting for a task and 
                # your alive flag is removed by the parent
                # kill yourself. 
                if not os.path.exists(child.alive) or os.path.exists(child.alive + '.cycle'):
                    done = True
                    break
                
                time.sleep(5)
            
            if not done:
                # print("found task")
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
    if os.path.exists(child.alive + '.done'):
        print("task completely finished")

    else:
        if os.path.exists(child.alive + '.cycle'):
            os.remove(child.alive + '.cycle')
        print(f"refreshing worker {child.id}")
        import platform
        if platform.system() == "Linux":
            os.system(f'bash refreshWorker.sh {line_args.exp_name} {line_args.args_file} {line_args.id}')
        elif platform.system() == "Windows":
            # p = subprocess.Popen(f'pwsh -w hidden -file refreshWorker.ps1 -i {line_args.id} -expname {line_args.exp_name} -fname {line_args.args_file}')
            # print(p.pid)
            pass
        print("stuff")


    child.pair.env.close()
