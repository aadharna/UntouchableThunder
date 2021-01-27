#!/usr/bin/env python
# coding: utf-8

import math
import functools
import numpy as np


class Cell:
    def __init__(self):
        self.walls = [True, True, True, True]
        self.marked = False
        
    def unlockDirection(self, direction):
        if direction['x'] == -1:
            self.walls[0] = False
        if direction['x'] == 1:
            self.walls[1] = False
        if direction['y'] == -1:
            self.walls[2] = False
        if direction['y'] == 1:
            self.walls[3] = False
            
        return
    
    def getWall(self, direction):
        
        if direction['x'] == -1:
            return self.walls[0]
        if direction['x'] == 1:
            return self.walls[1]
        if direction['y'] == -1:
            return self.walls[2]
        if direction['y'] == 1:
            return self.walls[3]
            
        return True

class Maze:
    
    def __init__(self):
        pass
    
    def generate(self, width, height):
        maze = []
        for y in range(height):
            maze.append([])
            for x in range(width):
                maze[y].append(Cell())
        
        #########        
        
        start = { 'x': math.floor(np.random.rand() * width), 'y': math.floor(np.random.rand() * height) }
                
        s = sorted([start], key=lambda x: 2 * np.random.rand() - 1)
        
        while len(s) > 0:
            current = s.pop(0)
            
            if not maze[current['y']][current['x']].marked:
                surrounding = []
                for x in [-1, 0, 1]:
                    for y in [-1, 0, 1]:
                        if (x == 0 or y == 0) and not (x == 0 and y == 0):
                            newPos = {'x':current['x'] + x,
                                      'y':current['y'] + y}
                            if (newPos['x'] >= 0 and 
                                newPos['y'] >= 0 and 
                                newPos['x'] <= (width - 1) and 
                                newPos['y'] <= (height - 1)):
                                
                                if maze[newPos['y']][newPos['x']].marked:
                                    surrounding.append({'x': x, 'y':y})
                sorted_surrounding = sorted(surrounding, key=lambda x: np.random.rand() - 0.5)
                if len(sorted_surrounding) > 0:
                    maze[current['y']][current['x']].unlockDirection(surrounding[0])
                    maze[current['y'] + sorted_surrounding[0]['y']][current['x'] + surrounding[0]['x']].unlockDirection(
                        { x: -1 * sorted_surrounding[0]['x'], y: -1 * sorted_surrounding[0]['y'] }
                    )
                    maze[current['y']][current['x']].marked = True
                    for x in [-1, 0, 1]:
                        for y in [-1, 0, 1]:
                            if ((x == 0 or y == 0) and not (x == 0 and y == 0)):
                                newPos = { 'x': current['x'] + x, 'y': current['y'] + y }
                                if (newPos['x'] >= 0 and newPos['y'] >= 0 and newPos['x'] <= width - 1 and newPos['y'] <= height - 1):
                                    s.append(newPos)
        
        
        result = []
        for y in range(height):
            result.append([])
            for x in range(width):
                result[y].append(1)
        
        
        for y in range(len(result)):
            for x in range(len(result[y])):
                if (y % 2 == 1 and x % 2 == 1):
                    pos = { 'x':math.floor(x / 2), 
                            'y': math.floor(y / 2) }
                    
                    result[y][x] = 0
                    if (not maze[pos['y']][pos['x']].getWall({ 'x': -1, 'y': 0 })):
                        result[y][x - 1] = 0
                    if (not maze[pos['y']][pos['x']].getWall({ 'x': 1, 'y': 0 })):
                        result[y][x + 1] = 0
                    if (not maze[pos['y']][pos['x']].getWall({ 'x': 0, 'y': -1 })):
                        result[y - 1][x] = 0
                    if (not maze[pos['y']][pos['x']].getWall({ 'x': 0, 'y': 1 })):
                        result[y + 1][x] = 0
        
            
        return result


class Zelda:
    def __init__(self, maze, charMap):
        self._maze = maze
        self._charMap = charMap
    
    def getAllPossibleLocations(self, _map):
        possibleLocations = []
        for x in range(len(_map[0])):
            for y in range(len(_map)):
                if (_map[y][x] == 0):
                    possibleLocations.append({ 'x': x, 'y': y })

        return possibleLocations
    
    def getAllSeparatorWalls(self, _map):
        possibleLocations = []
        for x in range(len(_map[0]) - 1):
            for y in range(len(_map) - 1):
                if (_map[y][x] == 1 and ((_map[y - 1][x] == 0 and _map[y + 1][x] == 0) or 
                                         (_map[y][x - 1] == 0 and _map[y][x + 1] == 0))):
                    possibleLocations.append({ 'x': x, 'y': y })

        return possibleLocations
    
    def distance(self, p1, p2):
        return np.abs(p1['x'] - p2['x']) + np.abs(p1['y'] - p2['y'])
    
    def getDifficultyParameters(self, diff, maxWidth, maxHeight):
        width = maxWidth
        height = maxHeight
        openess = (1 - diff) * 0.7 + 0.2 * np.random.rand() + 0.1
        enemies = diff * 0.4 + 0.3 * np.random.rand()
        distanceToGoal = diff * 0.6 + 0.3 * np.random.rand()
        return [width, height, openess, enemies, distanceToGoal]
    
    def adjustParameters(self, width, height, openess, enemies, distanceToGoal):
        parameters = [openess, enemies, distanceToGoal]
        parameters[0] = math.floor(openess * (width - 1) * (height - 1))
        parameters[1] = math.floor(enemies * 0.05 * width * height)
        parameters[2] = distanceToGoal + 1
        return [max(width, 4), max(height, 4)] + parameters
    
    def createPath(self, _map, a, b):
        "Greedily create path from agent a to b."
        start = a        
        while not start == b:
            
        
            dists = []
            for y in [-1, 0, 1]:
                for x in [-1, 0, 1]:
                    if x == 0 and y == 0:
                        continue

                    dists.append(({'x':start['x'] + x, 'y':start['y'] + y}, 
                                  self.distance({'x':np.clip(start['x'] + x, 0, len(_map[0])),
                                                 'y':np.clip(start['y'] + y, 0, len(_map))}, 
                                                b))
                                )

            s = sorted(dists, key=lambda x:x[1])[0]
            _map[s[0]['y']][s[0]['x']] = 0
            start = s[0]

        return
    
    def generate(self, width, height, openess, enemies, distanceToGoal):
        _map = self._maze.generate(width, height)
        walls = sorted(self.getAllSeparatorWalls(_map), key=lambda x: np.random.rand() - 0.5)
                
        for i in range(len(walls)):
            if not openess:
                break
            _map[walls[i]['y']][walls[i]['x']] = 0
            openess -= 1
        
        locations = sorted(self.getAllPossibleLocations(_map), key=lambda x: np.random.rand() - 0.5)
        
        avatar = locations.pop(0)
        _map[avatar['y']][avatar['x']] = 2
        
        def avatarCompare(a, b):
            return self.distance(avatar, b) - distanceToGoal * self.distance(avatar, a) + min(width, height) * (2 * np.random.rand() - 1)
        
        locations = sorted(locations, key=functools.cmp_to_key(avatarCompare))
        
        exit = locations.pop(0)
        _map[exit['y']][exit['x']] = 3
        
        def exitCompare(a, b):
            return (self.distance(avatar, b) - 
                    distanceToGoal * 
                    self.distance(avatar, a) + 
                    self.distance(exit, b) - 
                    self.distance(exit, a) + 
                    min(width, height) * 
                    (2 * np.random.rand() - 1))
            
        locations = sorted(locations, key=functools.cmp_to_key(exitCompare))
        
        key = locations.pop(0)
        _map[key['y']][key['x']] = 4
        
        def keyCompare(a, b):
            return (self.distance(avatar, b) - 
                    self.distance(avatar, a) + 
                    min(width, height) * 
                    (2 * np.random.rand() - 1))
        
        locations = sorted(locations, key=functools.cmp_to_key(keyCompare))
        
        for l in locations:
            if not enemies:
                break
            _map[l['y']][l['x']] = 5
            enemies -= 1
        
        self.createPath(_map, avatar, key)
        self.createPath(_map, key, exit)
        _map[key['y']][key['x']] = 4
        _map[exit['y']][exit['x']] = 3
        _map[avatar['y']][avatar['x']] = 2
        
        results = ""
        for y in range(len(_map)):
            for x in range(len(_map[y])):
                if _map[y][x] == 0:
                    results += self._charMap["EMPTY"]
                elif _map[y][x] == 1:
                    results += self._charMap["WALL"]
                elif _map[y][x] == 2:
                    results += self._charMap["AVATAR"]
                elif _map[y][x] == 3:
                    results += self._charMap["EXIT"]
                elif _map[y][x] == 4:
                    results += self._charMap["KEY"]
                elif _map[y][x] == 5:
                    results += self._charMap["MONSTER"]

            results += "\n"

        return results   
    
    def newLevel(self, diff, width, height):
        return self.generate(*self.adjustParameters(*self.getDifficultyParameters(diff, width, height)))
    


charMap = {
    "EMPTY":'.',
    "WALL":'w',
    "AVATAR":'A',
    "EXIT":'g',
    "KEY":'+',
    "MONSTER":'3'
}


# In[86]:


# if __name__ == "__main__":
#
#     m = Maze()
#     Z = Zelda(m, charMap)
#     print(Z.newLevel(0.02, 13, 9))


# In[ ]:




