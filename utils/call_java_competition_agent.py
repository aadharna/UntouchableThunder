from subprocess import *

def _jarWrapper(*args):
    process = Popen(['java', '-jar']+list(args), stdout=PIPE, stderr=PIPE)
    ret = []
    # keep the script holding the processes
    while process.poll() is None:
        line = process.stdout.readline()
        
        if len(line) == 0: #don't add in blanks ''
            continue
        ret.append(line[:-1])

    result = ret[-1] #forget about any warning that happened, and just extract the JSON
    return eval(result) #turn JSON string into dict


def runJavaAgent(jar, vgdl, lvl, agent, length):
    
    assert type(jar) == str
    assert type(vgdl) == str
    assert type(lvl) == str
    assert type(agent) == str
    assert type(length) == str
    
    args = [jar, vgdl, lvl, agent, length]
    # print(args)
    
    result = _jarWrapper(*args)
    
    # boolean. Did the agent win the level
    return float(result['win']) == 1


import os
from tqdm import tqdm

def _jarLimitedMCTS(*args):
    process = Popen(['java', '-jar']+list(args), stdout=PIPE, stderr=PIPE)
    ret = []
    # keep the script holding the processes
    while process.poll() is None:
        pass
    return


def limited_mcts(jar, vgdl, lvl, path, treesize, length):
    for d in tqdm(sorted(os.listdir('.'))): 
        if os.path.isdir(d): 
            try: 
                _net = int(d) 
            except ValueError as e: 
                continue 
            for f in os.listdir(f'./{_net}'): 
                if f'lvl' in f:
                    lvl_id = f.split(".")[0].split('l')[-1]
                    args = [jar, vgdl, f"{_net}/{f}", f"mcts{treesize}/{_net}.json", treesize, length]
                    result = _jarLimitedMCTS(*args)
                    break


