# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 15:04:36 2022

To be run after nsga_optimize_body.py
Goes through the entire final evolved population and evaluates n times
Uses multiprocessing

Run and output to file: python final_eval_cpg.py [num processes] [nsga3 output file]

@author: alexansz
"""

import nsga_optimize_body
import evoplot
import numpy as np
import sys
import functools
import UnityInterfaceBrain

def evalall(pop,workers):
    n = len(pop)
    fitnesses = [0.0 for i in range(n)]
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
    port = int(sys.argv[2])
    bodytype = sys.argv[3]
    inpath = sys.argv[4]
    outpath = inpath.split('.')[0] + '_final.txt'
    
    data,inds,scores = evoplot.main(inpath,[8,9,10])

    if 'hex' in bodytype:
        n_cpg = 28
        n_body = 9
        kwargs = {"tilt":[-1,1,1],'k':15,'seed':111}
    else:
        n_cpg = 23
        n_body = 9
        kwargs = {'k':15,'seed':111}

    function = functools.partial(UnityInterfaceBrain.run_from_array,n_cpg,bodytype,**kwargs)
    expath = UnityInterfaceBrain.getpath('Linux',bodytype) 

    workers = UnityInterfaceBrain.WorkerPool(function,expath,nb_workers=n_processes,port=port)

    fitnesses = evalall(inds,workers)

    #close down workers
    for i in range(n_processes):
        workers.addtask(None)
    workers.join()
    workers.terminate()
    
    with open(outpath,'w') as outfile:    
        for i in range(len(inds)):
            outfile.writelines(str(inds[i])+'\n')
            outfile.writelines(str(tuple(fitnesses[i]))+'\n')



    

