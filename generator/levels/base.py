import numpy as np
from copy import deepcopy
import os
from gym_gvgai import dir

class Generator:
    def __init__(self, tile_world, path=dir+'/envs/games/zelda_v0/', mechanics=[], generation=0, locations={}):
        """

        :param tile_world: 2d numpy array of map
        :param path: gym_gvgai.dir
        :param mechanics: list of sprites you would like to be able to mutate into
        :param generation: int 
        """
        
        self._length = tile_world.shape[0]
        self._height = tile_world.shape[1]
        self.BOUNDARY = {'w': [(x, y)
                               for x in range(self._length)
                               for y in range(self._height)
                               if (x == 0 or
                                   y == 0 or
                                   x == (self._length - 1) or
                                   y == (self._height - 1))]
                         }

        self._tile_world = tile_world

        self.mechanics = mechanics
        #make folder in levels folder
        self.base_path = path
        if not os.path.exists(os.path.join(self.base_path, 'levels')):
            os.mkdir(os.path.join(self.base_path, 'levels'))
        self._path = os.path.join(self.base_path, 'levels')

        self.generation = generation
        self.locations = locations if bool(locations) else self._parse_tile_world(tile_world)

        self.chars = np.unique(np.unique(self.tile_world).tolist() + self.mechanics)
        self.chars = list(set(self.chars) - {'A'}) # do not place more agents


    def _parse_tile_world(self, tile_world):
        locations = {}
        # comb through world, extract positions for each element currently in world
        for i in range(len(tile_world)):
            for j in range(len(tile_world[i])):
                c = tile_world[i][j]
                if c not in locations:
                    locations[c] = []
                    locations[c].append((i, j))
                else:
                    locations[c].append((i, j))
        # add in user-specified sprites as empty lists.
        for char in self.mechanics:
            if char not in locations:
                locations[char] = []

        # separate out mutable walls vs non-mutable walls
        mutable_walls = list(set(locations['w']) - set(self.BOUNDARY['w']))
        locations['w'] = mutable_walls

        return locations

    @property
    def tile_world(self):
        # numpy array
        npa = np.array([['0'] * self._height] * self._length, dtype=str)
        for k in self.locations.keys():
            for pos in self.locations[k]:
                npa[pos[0]][pos[1]] = k
        for pos in self.BOUNDARY['w']:
            npa[pos[0]][pos[1]] = 'w'
        return npa

    def cleanup(self):
        """remove generated/saved files.
        SHOULD ONLY BE CALLED ONCE (which actually tells me that this shouldn't live in the Generator class).

        :return:
        """
        for fname in os.listdir(self._path):
            os.remove(os.path.join(self._path, fname))
        os.rmdir(self._path)

    def to_file(self, env_id, game='zelda'):
        """Save environment currently in generator to a file.

        :param env_id: env_id passed in from above (GridGame class)
        :param game: name of game being mutated
        :return: path to newly created level file
        """
        path = os.path.join(self._path, f"{game}_id:{env_id}_g:{self.generation}.txt")
        with open(path, 'w+') as fname:
            fname.write(str(self))
            self.path_to_file = path
            # np.save(f"{path.split('.')[0]}.npy", self.tile_world)
        return path

    def mutate(self, mutationRate):
        """randomly edit parts of the level!
        :param mutationRate: e.g. 0.2
        :return: dict of location data for the entire level
        """
        locations = deepcopy(self.locations)
        def find_place_for_sprite():
            """find an empty location in the matrix for the sprite that is empty.

            :return: new (x, y) location
            """
            conflicting = True
            new_location = (0, 0)
            while conflicting:
                new_location = (np.random.randint(0, self._length),   # [, )  in, ex
                                np.random.randint(0, self._height))
                print(f"potential location {new_location}")

                # don't overwrite Agent, goal, or key
                if not (new_location in locations['A'] or
                        (new_location in locations['g'] and len(locations['g']) == 1) or
                        (new_location in locations['+'] and len(locations['+']) == 1) or
                         new_location in self.BOUNDARY['w']):
                    conflicting = False

            return new_location

        # if we manage to mutate:
        if np.random.rand() < mutationRate:
            choices = np.arange(1, 4)
            mutationType = np.random.choice(choices, p=[0.2, 0.4, 0.4])  # [, )  in, ex
            print(mutationType)
            # 1 -- remove sprite from scene               .... 20% chance
            # 2 -- spawn new sprite into the scene        .... 40% chance
            # 3 -- change location of sprite in the scene .... 40% chance
            if mutationType == 1:
                skip = False
                somethingToRemove = False
                # choose a random sprite that has multiple instances of itself to remove
                while not somethingToRemove:
                    sprite = np.random.choice(list(locations))
                    print(f"removing {sprite}?")
                    # do not remove agent, cannot remove floor
                    if sprite in ['A', '.']:
                        # pick a new mutation
                        mutationType = np.random.choice(choices, p=[0, 0.5, 0.5])
                        print(f"new mutation {mutationType}")
                        skip = True
                        break

                    # do not remove goal or key if there are only one of them
                    #  do not attempt to remove sprite when there are none of that type
                    elif len(locations[sprite]) <= 1:
                        if sprite in ['g', '+'] or len(locations[sprite]) == 0:
                            mutationType = np.random.choice(choices, p=[0, 0.5, 0.5])
                            print(f"new mutation {mutationType}")
                            skip = True
                            break
                    # else we have found something meaningful we can remove
                    else:
                        somethingToRemove = True
                # choose location index in list of chosen sprite
                if not skip:
                    ind = np.random.choice(len(locations[sprite]))
                    v = deepcopy(locations[sprite][ind])
                    print(f"removed {v}")
                    self.locations['.'].append(v)
                    self.locations[sprite].pop(ind)

            # spawn a new sprite into the scene
            if mutationType == 2:
                # choose a random sprite
                spawned = False
                while not spawned:
                    sprite = np.random.choice(list(locations))
                    if sprite == 'A':
                        continue
                    spawned = True
                print(f"spawning {sprite}?")
                new_location = find_place_for_sprite()

                # remove from whoever already has that new_location
                for k in locations.keys():
                    if new_location in locations[k]:
                        rm_ind = locations[k].index(new_location)
                        locations[k].pop(rm_ind)
                        break

                # add new sprite to the level
                locations[sprite].append(new_location)

            # move an existing sprite in the scene
            if mutationType == 3:
                moved = False
                while not moved:
                    # choose a random viable sprite
                    sprite = np.random.choice(list(self.locations))
                    if len(list(locations[sprite])) == 0 or sprite == '.':
                        continue
                    moved = True

                print(f"moving {sprite}?")
                # choose location index in list of chosen sprite
                ind = np.random.choice(len(locations[sprite]))
                # where the sprite was previously
                old = locations[sprite][ind]
                print(f'from {old}')
                # new location for sprite
                new_location = find_place_for_sprite()

                # remove whoever already has that new_location
                # e.g. wall, floor
                for k in locations.keys():
                    if new_location in locations[k]:
                        rm_ind = locations[k].index(new_location)
                        locations[k].pop(rm_ind)
                        break

                # move sprite to new location
                locations[sprite].append(new_location)
                # fill previous spot with blank floor.
                locations['.'].append(old)
                locations[sprite].pop(ind)

        # remove anything that was in the boundary wall's spot.
        for k in locations.keys():
            for i, p in enumerate(locations[k]):
                if p in self.BOUNDARY['w']:
                    locations[k].pop(i)

        return locations

    # def crossOver(self, parent):
    #     """Edit levels via crossover rather than mutation
    #     :param self: parent level A
    #     :param parent: parent level B
    #     :return: child level generator
    #     """
    #
    #     child = Generator(tile_world= self.tile_world,
    #                       mechanics = self.mechanics,
    #                       generation= self.generation + 1)
    #
    #     for i in range(len(child._tile_world)):
    #         for j in range(len(child._tile_world[i])):
    #             if np.random.choice([0, 1]):
    #                 child._tile_world[i][j] = self._tile_world[i][j]
    #             else:
    #                 child._tile_world[i][j] = parent._tile_world[i][j]
    #
    #     return child

    def __str__(self):
        stringrep = ""
        tile_world = self.tile_world
        for i in range(len(tile_world)):
            for j in range(len(tile_world[i])):
                stringrep += tile_world[i][j]
                if j == (len(tile_world[i]) - 1):
                    stringrep += '\n'
        return stringrep


def _initialize(path, d=9):
    """build numpy array of level from txt file

    :param path: path to txt file representation of level
    :return:
    """
    f = open(path, 'r')
    f = f.readlines()
    rep = []
    for l in f:
        rep.append(l[:-1])  # cut off '\n'
    mat = []
    for r in rep:
        for s in r:
            mat.append(s)
    npa = np.array(mat).reshape((d, -1))  # make into numpy array 9x13
    return npa