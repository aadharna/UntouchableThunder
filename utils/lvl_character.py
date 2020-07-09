import numpy as np
from scipy.spatial.distance import euclidean
from generator.levels.base import Generator, _initialize
from utils.AStar import astar

def getLvlCharacterization(lvlpath, seedLvls, args_file):
    tile = _initialize(lvlpath)
    gen = Generator(tile, shape=tile.shape, path=seedLvls, args_file=args_file)
    
    booleanMap = gen.tile_world
    valid = ~np.logical_xor((booleanMap != 'w'), (booleanMap != '3'))
    numericMap = np.zeros(shape=booleanMap.shape)
    numericMap[valid] = 0
    numericMap[~valid] = 1
    
    
    keyPath = astar(numericMap, 
                    gen.locations['A'][0], 
                    gen.locations['+'][np.argmin([euclidean(gen.locations['A'], k) for k in gen.locations['+']])])
    
    
    doorPath = []
    prev = keyPath[-1]
    for each_exit in gen.locations['g']:
        doorPath += astar(numericMap, 
                          prev, 
                          each_exit)
        prev = each_exit
        
    
    gamePath = keyPath + doorPath[1:]
    
    featureSet = [len(gen.locations['g']), 
                  len(gen.locations['+']), 
                  len(gen.locations['3']), 
                  len(keyPath), 
                  len(doorPath)]
    
    return featureSet


