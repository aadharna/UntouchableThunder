import numpy as np
from scipy.spatial.distance import euclidean
from generator.levels.base import _initialize
from generator.levels.EvolutionaryGenerator import EvolutionaryGenerator
from utils.AStar import astar

def getSolarfoxLvlCharacterization(lvlpath, seedLvls, args_file):
    tile = _initialize(lvlpath)
    gen = EvolutionaryGenerator(tile, shape=tile.shape, path=seedLvls, args_file=args_file)
    
    booleanMap = gen.tile_world
    
    valid = ~np.logical_xor((booleanMap != 'w'),
                            (booleanMap != '.'))
    
    numericMap = np.zeros(shape=booleanMap.shape)
    numericMap[valid] = 0
    numericMap[~valid] = 1
    
    
    doorPath = []
    prev = gen.locations['A'][0]
    keys = gen.locations['b'] + gen.locations['p']
    for each_exit in keys:
        doorPath += astar(numericMap, 
                          prev, 
                          each_exit)
        prev = each_exit
    
    featureSet = [len(gen.locations['b']), 
                  len(gen.locations['p']), 
                  len(gen.locations['w']), 
                  len(doorPath)]
    
    return featureSet


