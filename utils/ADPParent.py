import os
import time
import shutil
import numpy as np
from utils.utils import save_obj, load_obj


class ADPParent:
    """
    ADAPTIVE DYNAMIC PROCESS PARENT CLASS
    This will launch out commands into the void
    for APDChild processes to pick up and execute.
    """
    def __init__(self, prefix='.'):

        self.root = os.path.join(prefix, 'communication')
        if not os.path.exists(self.root):
            os.mkdir(self.root)
        
        self.subfolders = {
            'send_to_child': 'outgoing',
            'sent_by_child': 'incoming',
            'alive_signals': 'alive',
            'available_signals': 'available'
        }
        self.createFolders()
        self.resetFolders()

    def placeChildFlag(self, path):
        with open(path, 'w+') as f:
            pass

    def createFolders(self):
        for f in self.subfolders.keys():
            path = os.path.join(self.root, self.subfolders[f])
            if not os.path.exists(path):
                os.mkdir(path)

    def resetFolders(self):
        for (_, dirs, _) in os.walk(self.root):
            for d in dirs:
                path  = os.path.join(self.root, d)
                files = os.listdir(path)
                for f in files:
                    os.remove(os.path.join(path, f))

    def checkChildResponseStatus(self, allChildren, send_again=[]):        
        response_folder = os.path.join(self.root,
                                       self.subfolders['sent_by_child'])
        
        available_signals = os.listdir(os.path.join(self.root,
                                   self.subfolders['available_signals']))
        
        childTaskComplete = [False]*len(allChildren)
        deadChildren = []
        
        
        # In order to assign work from child i to child j AFTER the initial distribution of work,
        #  we need to know which children have finished their current work. Therefore, this check
        #  needs to happen first as will potentially be used in the later check.
        for i, c in enumerate(allChildren):
            if os.path.exists(os.path.join(response_folder, f'answer{c}') + '.pkl') and \
               f'{c}.txt' in available_signals:
                childTaskComplete[i] = True
        
        finishedChildren = np.array(allChildren)[childTaskComplete]
        
        for i, c in enumerate(allChildren):
            if os.path.exists(os.path.join(response_folder, f'resend{c}') + '.txt') and \
                f'{c}.txt' in available_signals:
                send_again.append((c, c))
                os.remove(os.path.join(response_folder, f'resend{c}') + '.txt')

            if os.path.exists(os.path.join(response_folder, f'dead{c}') + '.txt') and finishedChildren.size > 0:
                send_again.append((c, self.selectAvailableChild(finishedChildren)))
                deadChildren.append(c)
                os.remove(os.path.join(response_folder, f'dead{c}') + '.txt')
                
        # python passes by reference. Therefore this is editing POET's "which children are alive" list.
        for c in deadChildren:
            allChildren.remove(c)
        
        return np.all(childTaskComplete)

    def selectAvailableChild(self, availableChildren):
        return np.random.choice(availableChildren)

    def isChildAvailable(self, children):
        availableChildren = []
        alive_signals = [int(c.split('.')[0]) for c in os.listdir(
            os.path.join(self.root, self.subfolders['alive_signals'])
        )]

        available_signals = [int(c.split('.')[0]) for c in os.listdir(
            os.path.join(self.root, self.subfolders['available_signals'])
        )]

        for child in children:
            # if child is alive and available
            if child in alive_signals and child in available_signals:
                availableChildren.append(child)

        return availableChildren

    def readChildAnswer(self, response_file):
        folder = os.path.join(self.root,
                              self.subfolders['sent_by_child'])
        answer = load_obj(folder, response_file)
        # remove answer from folder
        os.remove(os.path.join(folder, response_file))
        return answer

    def pickupChildren(self):
        """
        Picks up all of the alive children.
        :return:
        """
        alive_signals = os.listdir(
            os.path.join(self.root, self.subfolders['alive_signals'])
        )
        children = []
        for child in alive_signals:
            id = child.split('.')[0] #child names are 1.txt, etc
            children.append(int(id))

        return sorted(children)

    def createChildTask(self, run_id, work_dict, worker_id, task_id, poet_loop_counter, **kwargs):
        """

        :param work_dict: dict of nns, envs, nn_ids, env_ids
        :param worker_id:  child id (int)
        :param task_id: ADP TASK TYPE
        :param poet_loop_counter: poet number loop
        :return:
        """

        work = {
            'run_id': run_id,
            'nns': work_dict['nn'],
            'lvls': work_dict['env'],
            'task_id': task_id,
            'chromosome_ids': work_dict['nn_id'],
            'env_ids': work_dict['env_id'],
            'diff': work_dict['diff'],
            'poet': poet_loop_counter,
            'kwargs': kwargs
        }

        save_obj(work,
                 os.path.join(self.root, self.subfolders['send_to_child']),
                 f'child{worker_id}')

        available = os.path.join(self.root,
                                 self.subfolders['available_signals'],
                                 f'{worker_id}.txt')
        
        if not os.path.exists(available):
            time.sleep(10)
        
        if os.path.exists(available):
            os.remove(available)
