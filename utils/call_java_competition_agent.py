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
    print(args)
    
    result = _jarWrapper(*args)
    
    # boolean. Did the agent win the level
    return float(result['win']) == 1