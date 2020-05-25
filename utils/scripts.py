import pandas as pd; import os 


def make_transfer_graph():

    c = {} 
    for d in os.listdir('.'): 
         if os.path.isdir(d): 
            try: 
                _net = int(d) 
            except ValueError as e: 
                continue 
            for i in range(10, 5010, 10): 
                j = i - 1 
                if j not in c: 
                    c[j] = 0 
                for f in os.listdir(f'./{_net}'): 
                    if f'poet{j}_network' in f: 
                        c[j] += 1
    
    df = pd.DataFrame.from_dict(c, orient='index') 
    df.to_csv("transfers_per_attempt.csv") 
    
    return

def concat_opt_gens(lvl, exp):
    dfs = []  
    for f in os.listdir('.'): 
        if f[-3:] == 'csv': 
            dfs.append(pd.read_csv(f).drop('Unnamed: 0', axis=1))
    df = pd.concat(dfs, axis=1)
    
    df.to_csv(f'scores{lvl}_{exp}.csv')
    
    return

def countEvents(event): 
    eventCounter = 0 
    dirs = os.listdir('.')
    for d in dirs: 
        if os.path.isdir(d): 
            files = os.listdir(os.path.join('.', d))
            for f in files: 
                if event in f: 
                    eventCounter += 1 
                    if event == 'win':
                        break 
    return eventCounter


import os
import shutil
import numpy as np

def get_parent(directory):

    child = [directory, None]
    parent = 0
    files = os.listdir(f'./{directory}')
    par_id = -1
    for f in files:
        if "parent" in f:
            par_id = int(f.split(".")[0].split('_')[-1])
    for f in os.listdir(f'./{par_id}'):
        if 'lvl' in f:
            child = f
    return par_id, child                                                                                 


def get_tree(seedparent, lvl, folder):
    
    parent = seedparent
    x = [(parent, f'lvl{lvl}.txt')]
    while not parent == 0:
        parent, lvl = get_parent(parent)
        x.append((parent, lvl))                                                                              
    q = []
    for i, (netid, lvl) in enumerate(reversed(x)):
        try:
            q.append((np.random.choice(sorted([f for f in os.listdir(f'./{netid}') if '.pt' in f], key=lambda z: int(z.split( ".")[0].split('t')[-1]))), netid, lvl))
        except ValueError as e:
            q.append((None, netid, lvl))

    for i, (net, f, lvl) in enumerate(q):
        if net is not None:
            shutil.copy(f'./{f}/{net}', f'../../{folder}/{i}_{net}')
        shutil.copy(f'./{f}/{lvl}', f'../../{folder}/{i}_{lvl}')


def createMCTSCurve(treeSize): 
    """Given a folder of results created by `call_java_comp_agent: limited_mcts` search through that folder, load the results of the treesize experiment ran, and save that to a csv file. 
    result 0 = lose, 1 = win"""
    results = {}
    for lvl in sorted(os.listdir(f"./mcts{treeSize}"), key=lambda x: int(x.split('.')[0])):
        lvl_id = int(lvl.split(".")[0])
        results[lvl_id] = json.load(open(f"mcts{treeSize}/{lvl}"))
    df = pd.DataFrame.from_dict(results, orient='index')
    df.to_csv(f"mcts{treeSize}_results.csv")


