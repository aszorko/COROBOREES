# -*- coding: utf-8 -*-
"""


@author: alexansz
"""

import functools
import evoplot
import numpy as np
import UnityInterfaceBrain
import matsuoka_quad
import matsuoka_hex
import os
import time
import sys

def evalall(pop,workers):
    n = len(pop)
    fitnesses = [[] for i in range(n)]
    ind_tups = list(zip(range(n),pop))
    for ind in ind_tups:
         workers.addtask(ind)
    
    #wait for completion
    workers.join()
    
    print('Retrieving fitnesses')
    for i in range(n):
         newval = workers.outqueue.get()
         fitnesses[newval[0]] = newval[1]
         workers.outqueue.task_done()
    return fitnesses

def dummy(env,ind):
    pass

if __name__ == "__main__":

    port = 9200
    n_processes = int(sys.argv[1])
    if sys.argv[2]=="short":
        datadir = r'./paper2_data/'
        files = os.listdir(datadir)
        bodytype = 'shortquad'
        n_cpg = 23
        inpaths = [datadir + file for file in files if 'brain' in file and 'short' in file and 'final' not in file]
    elif sys.argv[2]=="normal":
        datadir = r'./paper2_data/'
        files = os.listdir(datadir)
        n_cpg = 23
        bodytype = 'ODquad'
        inpaths = [datadir + file for file in files if 'brain' in file and 'short' not in file and 'final' not in file]
    elif sys.argv[2]=="hex":
        datadir = r'./hexdata/'
        files = os.listdir(datadir)
        bodytype = 'AIRLhex'
        n_cpg = 28
        inpaths = [datadir + file for file in files if 'brain' in file and 'final' not in file]
    else:
        raise ValueError("invalid body type")
        
    datadir = r'./hexdata/'
    files = os.listdir(datadir)
    
        
    n_brain = 6
    numiter = 5
    kwargs = {
    'ratios': [0.618,0.786,1,1.272,1.618],
    'skipevery': 4,
    'sdev':0.02,
    'seed': 111,
    'combined': False,
    'numiter': numiter
    }

    unitypath = UnityInterfaceBrain.getpath('Linux',bodytype)

    for k, path in enumerate(inpaths):
        print(path)
        outpath = '.'.join(path.split('.')[:-1]) + '_final.txt'        
        data,inds,scores,header = evoplot.main(path,[],getheader=True)
        for line in header:
            if 'body parameters' in line:
                cpgstr = line.split(':')[1].replace('[','').replace(']','').split(', ')
                cpginds =  [int(num) for num in cpgstr]
            if 'Base period' in line:
                baseperiod = float(line.split(':')[1])
            
        print(cpginds)
        print(baseperiod)
        
        if 'hex' in bodytype:
            cpg = matsuoka_hex.array2param(cpginds[:n_cpg])
        else:
            cpg = matsuoka_quad.array2param(cpginds[:n_cpg])
        
        function = functools.partial(UnityInterfaceBrain.run_brain_array,n_brain,cpg,cpginds[n_cpg:],baseperiod,bodytype,**kwargs)
        workers = UnityInterfaceBrain.WorkerPool(function,unitypath,nb_workers=n_processes,port=port+(k%2)*n_processes)
        fitnesses = evalall(inds,workers)
        
        fitarray = np.array(fitnesses)
        n_fits = len(fitnesses[0]) // numiter
        n_pop = len(fitnesses)
        newfits =  np.zeros([n_pop,n_fits])
        for i,fits in enumerate(fitarray):
            for j in range(n_fits):
                newfits[i,j] = np.median(fits[j*numiter:(j+1)*numiter])
    
        #period matching score is every second element 
        sortinds = np.argsort(np.mean(newfits[:,1::2],axis=1))
        #sortinds = np.argsort(np.mean(newfits,axis=1))
        
        
        with open(outpath,'w') as outfile:    
            for i in range(len(sortinds)):
                outfile.writelines(str(inds[sortinds[i]])+'\n')
                outfile.writelines(str(tuple(newfits[sortinds[i]]))+'\n')
        
        #close down workers
        for i in range(n_processes):
            workers.addtask(None)
        workers.join()
        workers.terminate()
        
        time.sleep(5)
        
        print(outpath)

    

