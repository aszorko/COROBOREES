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
import matsuoka_hex
import matsuoka_brain
import os
import sys




def evalone(n_brain,n_cpg,n_body,bodytype,dc_in,t_arr,amp_arr,asym,pattern,env,ind,numiter=5,nframes=1600,seed=None):

    cpg_inds = ind[:n_cpg]
    body_inds = ind[n_cpg:(n_cpg+n_body)]
    brain_inds = ind[(n_cpg+n_body):]
    
    if 'hex' in bodytype:
        cpg = matsuoka_hex.array2param(cpg_inds)
        m=6
    else:
        cpg = matsuoka_quad.array2param(cpg_inds)
        m=4
    
    brain,outw,decay,outbias = matsuoka_brain.array2brain(n_brain,m,brain_inds)
    
    UnityInterfaceBrain.evaluate(env,cpg,body_inds,bodytype,nframes=100)   
    
    if len(amp_arr)>1:
        dim2 = len(amp_arr)
    elif asym is not None and hasattr(asym[1], '__iter__'):
        dim2 = len(asym[1])
    else:
        dim2 = len(dc_in)

    allperiod = np.zeros([numiter,len(t_arr),dim2])
    allheight = np.zeros([numiter,len(t_arr),dim2])    
        
        

    for j in range(numiter):            
        seed1 = seed+j*len(t_arr)
        if len(amp_arr)>1:
            for i,amp in enumerate(amp_arr):
                (dist,height,period,zero_std) = UnityInterfaceBrain.run_with_input(env,cpg,body_inds,bodytype,1,brain,outw,decay,outbias,t_arr,amp,nframes,dc_in[0],0,seed=seed1,asym=asym,pattern=pattern)
                allperiod[j,:,i] = period
                allheight[j,:,i] = height
        elif asym is not None and hasattr(asym[1], '__iter__'):
            for i,asym_frac in enumerate(asym[1]):
                (dist,height,period,zero_std) = UnityInterfaceBrain.run_with_input(env,cpg,body_inds,bodytype,1,brain,outw,decay,outbias,t_arr,amp_arr[0],nframes,dc_in[0],0,seed=seed1,asym=(asym[0],asym_frac),pattern=pattern)
                allperiod[j,:,i] = period
                allheight[j,:,i] = height
        else:
            for i,dc in enumerate(dc_in):
                (dist,height,period,zero_std) = UnityInterfaceBrain.run_with_input(env,cpg,body_inds,bodytype,1,brain,outw,decay,outbias,t_arr,amp_arr[0],nframes,dc,0,seed=seed1,asym=asym,pattern=pattern)
                allperiod[j,:,i] = period
                allheight[j,:,i] = height
        
    return (np.squeeze(np.median(allperiod,0)),np.squeeze(np.median(allheight,0)))    


def evalall(pop,workers):
    n = len(pop)
    fitnesses = [() for i in range(n)]
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

    if len(sys.argv)<3:
        raise ValueError('Too few arguments. [bodytype] [process no.]')
    process = int(sys.argv[2])
    if sys.argv[1]=="normal":
        port = 9500 + 50*(process-1)
        bodytype = 'ODquad'
        datadir = r'./paper2_data/'
        files = os.listdir(datadir)
        suff = '_final3'
        inpaths = [datadir + file for file in files if 'brain' in file and 'short' not in file and suff in file]
    elif sys.argv[1]=="short":
        port = 9600 + 50*(process-1)
        bodytype = 'shortquad'
        datadir = r'./paper2_data/'
        files = os.listdir(datadir)
        suff = '_final3'
        inpaths = [datadir + file for file in files if 'brain' in file and 'short' in file and suff in file]
    elif sys.argv[1]=="hex":
        port = 9700 + 50*(process-1)
        bodytype = 'AIRLhex'
        datadir = r'./hexdata/'
        files = os.listdir(datadir)
        suff = '_final'
        inpaths = [datadir + file for file in files if 'brain' in file and suff in file]        
    else:
        raise ValueError("invalid body type")

        
    outdir = r'./paper3_data/'
    
    
        
    n_brain = 6    
    n_body = 9 #number of body parameters

    if 'hex' in bodytype:
        n_cpg = 28
        periodfactor = 1
        minheight = 1
        nframes = 1600
    else:
        n_cpg = 23 #number of parameters in CPG generator
        periodfactor = 0.0481 #paper2 adjustment from CPG time to Unity time
        minheight = 0.75
        nframes = 240

    kwargs = {
       'numiter': 5,
       'seed': 111,
       'nframes':nframes
    }

    

    unitypath = UnityInterfaceBrain.getpath('Linux',bodytype)

    inds = []
    #allbaseperiod = []
    allorigscore = []
    goodinpaths = []


    #get all cpg and brain combos, concatenate each into single list
    for path in inpaths:
        data,brains,bscores = evoplot.main(path,[],startmode=2)
        #for line in header:
        #    if 'body parameters' in line:
        #        cpgstr = line.split(':')[1].replace('[','').replace(']','').split(', ')
        #        currcpg = [int(num) for num in cpgstr]
        #    if 'Base period' in line:
        #        allbaseperiod.append(float(line.split(':')[1]))

        currcpg = []
        #baseperiod = None

        #find best brain ind    
        i=0
        currscore = np.nan
        currheight = np.nan
        currbrain = []
        while i<len(bscores):
           i += 1
           if np.sum(np.array(bscores[-i])[::2]<minheight)==0:
               currheight = np.mean(bscores[-i][::2])
               currscore = np.mean(bscores[-i][1::2])
               currbrain = brains[-i]
               #print(run,cpg,i,currheight,currscore)
               break
        if len(currbrain)==0:
            continue #good brain not found
        
        origpath = path.replace(suff,'')
        _,_,_,header = evoplot.main(origpath,[],getheader=True)

        for line in header:
            if 'body parameters' in line:
                cpgstr = line.split(':')[1].replace('[','').replace(']','').split(', ')
                currcpg = [int(num) for num in cpgstr]
            #if 'Base period' in line:
            #    baseperiod = float(line.split(':')[1])

        #if baseperiod==None:
        #    raise ValueError('missing base period')
        if len(currcpg)==0:
            raise ValueError('missing body parameters')

        allorigscore.append(currscore) 
        #allbaseperiod.append(baseperiod*periodfactor)
        inds.append(currcpg + currbrain)
        goodinpaths.append(path)

        print(path)

    #if len(allbaseperiod)<len(inpaths):
    #    raise ValueError('missing base period')
    #if len(inds)<len(inpaths):
    #    raise ValueError('missing individual')
       
    
    n_processes = len(goodinpaths)
    asym_metre = 2

    if process == 1:
       amp_arr = np.arange(0.1,1.1,0.1)
       dc_arr = [0.5]
       t_arr = 1/np.arange(1,3.1,0.2) #1 to 3 Hz
       asym = None
       pattern = None
       outprefix = '_impulse_'
    elif process == 2:
       amp_arr = [1.0]
       dc_arr = [0.5]
       t_arr = 1/np.arange(1,3.1,0.2) #1 to 3 Hz
       asym = (2,np.arange(0,0.95,0.1))
       pattern = None
       outprefix = '_asym_'
    elif process == 3:
       amp_arr = [1.0]
       dc_arr = np.arange(0,1.05,0.1)
       t_arr = np.array([0.5,1.0])
       asym = None
       pattern = None
       outprefix = '_iso120_'
    elif process == 4:
       amp_arr = np.arange(0.1,1.1,0.1)
       dc_arr = [0.5]
       t_arr = 1/np.arange(1,3.1,0.2) #1 to 3 Hz
       asym = (3,0.5)
       pattern = None
       outprefix = '_ternary_'
    elif process == 5:
       amp_arr = np.arange(0.1,1.1,0.1)
       dc_arr = [0.5]
       t_arr = 1/np.arange(1,3.1,0.2) #1 to 3 Hz
       asym = (4,0.5)
       pattern = None
       outprefix = '_quaternary_'
    elif process == 6:
       amp_arr = np.arange(0.1,1.1,0.1)
       dc_arr = [0.5]
       t_arr = 0.5/np.arange(1,3.1,0.2) #1 to 3 Hz
       asym = None
       pattern = [True,False,True,False,False,True,False,False]
       outprefix = '_quaternary2_'       

    function = functools.partial(evalone,n_brain,n_cpg,n_body,bodytype,dc_arr,t_arr,amp_arr,asym,pattern,**kwargs)
    workers = UnityInterfaceBrain.WorkerPool(function,unitypath,nb_workers=n_processes,port=port+(i%2)*n_processes,clargs = ['-bodytype',bodytype])

    alloutput = evalall(inds,workers)

    #close down workers
    for ii in range(n_processes):
       workers.addtask(None)
    workers.join()
    workers.terminate()
    
    
    outnames = ['period','height']
    
    #export data
    for kk,path in enumerate(goodinpaths):
        arr = np.array(alloutput[kk])
        outpath = path.replace(datadir,outdir).replace(suff,outprefix+'info')
        with open(outpath,'w') as outfile:
            outfile.writelines('period ratios:' + str(t_arr)+'\n')
            outfile.writelines('amplitude:' + str(amp_arr)+'\n')
            outfile.writelines('asymmetry:' + str(asym)+'\n')
            outfile.writelines('dc:' + str(dc_arr)+'\n')
        for jj,name in enumerate(outnames):
           outpath = path.replace(datadir,outdir).replace(suff,outprefix+name)
           with open(outpath,'w') as outfile:
               if len(arr[jj].shape)==1:
                   outfile.writelines(str(list(arr[jj,:])).replace('[','').replace(']','')+'\n')
               else:
                   for i in range(len(arr[jj])):
                       outfile.writelines(str(list(arr[jj,i,:])).replace('[','').replace(']','')+'\n')



