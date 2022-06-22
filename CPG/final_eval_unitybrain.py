# -*- coding: utf-8 -*-
"""
Runs all final brain populations several times and gets separate scores for height and entrainment
Uses multiprocessing
Usage: python final_eval_unitybrain.py [n_cpu] [bodytype]

@author: alexansz
"""

import functools
import evoplot
import numpy as np
import UnityInterfaceBrain
import matsuoka_quad
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



if __name__ == "__main__":

    n_processes = int(sys.argv[1])
    if sys.argv[2]=="normal":
        port = 9200
        short = False
    elif sys.argv[2]=="short":
        port = 9400
        short = True
    else:
        raise ValueError("invalid body type")
        
    datadir = r'./paper2_data/'
    files = os.listdir(datadir)
    
    if short==True:
       bodytype = 'shortquad'
       inpaths = [datadir + file for file in files if 'brain' in file and 'short' in file and 'final' not in file]
    else:
       bodytype = 'ODquad'
       inpaths = [datadir + file for file in files if 'brain' in file and 'short' not in file and 'final' not in file]
        
    n_brain = 6    
    n_cpg = 23 #number of parameters in CPG generator
    numiter = 5
    kwargs = {
    'skipevery': 4,
    'sdev':0.02,
    'seed': 111,
    'combined': False,
    'numiter': numiter,
    }

    unitypath = UnityInterfaceBrain.getpath('Linux',bodytype)

    for k, path in enumerate(inpaths):
        print(path)
        outpath = '.'.join(path.split('.')[:-1]) + '_final2.txt'        
        data,inds,scores,header = evoplot.main(path,[],getheader=True)
        for line in header:
            if 'body parameters' in line:
                cpgstr = line.split(':')[1].replace('[','').replace(']','').split(', ')
                cpginds =  [int(num) for num in cpgstr]
            if 'Base period' in line:
                baseperiod = float(line.split(':')[1])
            
        print(cpginds)
        print(baseperiod)
        
        
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

    

