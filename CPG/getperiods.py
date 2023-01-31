# -*- coding: utf-8 -*-
"""
Created on Mon May 30 16:34:48 2022

Iterate over a CPG's control parameters, get periods and interlimb correlations

@author: alexansz
"""

import numpy as np
import UnityInterfaceBrain
import evoplot
import pandas as pd
import functools
import os
import sys

def evalall(pop,bodytype,n_cpg,n_processes=16,numiter=5,dc=[0.5],tilt=[0],seed=111):
    
    path = UnityInterfaceBrain.getpath('Linux',bodytype)
    function = functools.partial(UnityInterfaceBrain.iterate,n_cpg,dc,tilt,bodytype,iterations=numiter,seed=seed)
    workers = UnityInterfaceBrain.WorkerPool(function,path,nb_workers=n_processes,port=9600)
    n = len(pop)
    outputs = [[] for i in range(n)]
    
    ind_tups = list(zip(range(n),pop))
    for ind in ind_tups:
         workers.addtask(ind)

    #wait for completion
    workers.join()

    #print('Retrieving fitnesses')
    for i in range(n):
         newval = workers.outqueue.get()
         outputs[newval[0]] = newval[1]
         workers.outqueue.task_done()
          
    #close down workers
    for i in range(n_processes):
        workers.addtask(None)
    workers.join()
    workers.terminate()
    
    return outputs

    
    
if __name__ == "__main__":
    
    #function 1: all CPGs, limited parameters; function 2: all CPGs with brains, full control parameter sweep
    
    n_processes = int(sys.argv[1])
    function = int(sys.argv[2])
    bodytype = sys.argv[3]   

    runs = list(range(1,6))
    
    if function==1:
        
        if bodytype == 'shortquad':
           filebase = 'paper2_data/nsga_unityshort_t4_g200_'
           outfile = 'paper2_data/cpg_short_alldata_5runs_forward.txt'
           f1tilt = [0.016]
           n_cpg = 23
        elif bodytype == 'ODquad':
           filebase = 'paper2_data/nsga_unity_t4_g200_'
           outfile = 'paper2_data/cpg_alldata_5runs_forward.txt'
           f1tilt = [0.016]
           n_cpg = 23
        elif bodytype == 'AIRLhex':
           filebase = 'hexdata/nsga_hex_lowangle_'
           outfile = 'hexdata/cpg_hex_alldata_5runs_forward.txt'
           f1tilt = [1.0]
           n_cpg = 28
        else:
           raise ValueError('Invalid body type')

        files = [filebase + str(i) + '_final.txt' for i in runs]
        
        allinds = []
        allscores = []
        allruns = []
        for i,file in enumerate(files):
            data,inds,scores = evoplot.main(file,[],startmode=2)
            scorearr = np.array(scores)
            inds_u,indices = np.unique(inds,axis=0,return_index=True)
            scorearr_u = scorearr[indices,:]
            allinds.append(inds_u)
            allscores.append(scorearr_u)
            allruns.append([runs[i] for x in range(len(inds_u))])

        allinds = np.concatenate(allinds)
        allscores = np.concatenate(allscores)
        allruns = np.concatenate(allruns)

        print(len(allinds),"unique individuals")
        outputs = evalall(allinds,bodytype,n_cpg,n_processes=n_processes,dc=[0.5,1],tilt=f1tilt)
        
        with open(outfile,'w') as f:
            for i,output in enumerate(outputs):
                newscores = np.concatenate([allscores[i],np.array(output).flatten(),[allruns[i]]])
                f.writelines(str(list(allinds[i])) + '\n')
                f.writelines(str(tuple(newscores)) + '\n')

    if function==2:
                
        dc_arr = np.arange(0, 1.01, 0.05)

        cpgs = []
        runnums = []
        cpgnums = []
        
        if 'hex' in bodytype:
           datadir = r'./hexdata/'
           tilt_arr = np.arange(-1, 1.05, 0.25)
           n_cpg = 28
        else:    
           datadir = r'./paper2_data/'
           tilt_arr = np.arange(-0.02, 0.022, 0.005)
           n_cpg = 23

        files = os.listdir(datadir) 
        
        if bodytype=='shortquad':
           outpath = datadir + 'unityshort_run'
           inpaths = [file for file in files if 'brain' in file and 'short' in file and 'final' not in file]
        elif bodytype=='ODquad':
           outpath = datadir + 'unity_run'
           inpaths = [file for file in files if 'brain' in file and 'short' not in file and 'final' not in file]
        else:
           outpath = datadir + 'unity_hex_run'
           inpaths = [file for file in files if 'brain' in file and 'final' not in file]
        for path in inpaths:
            #below will only work for single digit run and cpg numbers
            parts = path.split('_')
            run = int(parts[1][-1])   
            if run in runs:
                print(path)
                cpg = int(parts[2].split('.')[0][-1])
                data,inds,scores,header = evoplot.main(datadir + path,[],getheader=True)
                for line in header:
                    if 'body parameters' in line:
                        cpgstr = line.split(':')[1].replace('[','').replace(']','').split(', ')
                        cpginds =  [int(num) for num in cpgstr]
                        break
                cpgs.append(cpginds)
                runnums.append(run)
                cpgnums.append(cpg)
                        
        print('Running...')                
        outputs = evalall(cpgs,bodytype,n_cpg,n_processes=n_processes,dc=dc_arr,tilt=tilt_arr)
       
        for i,output in enumerate(outputs):
            run = runnums[i]
            cpgnum = cpgnums[i]
            outfilebase = outpath + str(run) + '_cpg' + str(cpgnum)
            outputvars = ['_dist', '_period', '_height', '_tilt', '_corrmax', '_corrind']
            for j,arr in enumerate(output):
                df = pd.DataFrame(arr)
                df.to_csv(outfilebase + outputvars[j] + '.txt', index=False)
        
