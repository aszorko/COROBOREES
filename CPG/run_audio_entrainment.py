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
import time
import sys


dc_arr = np.arange(0,1.1,0.1)

def evalone(n_brain,n_cpg,n_body,bodytype,dc_in,env,ind,numiter=5,nframes=1600,seed=None):
    #hack: dc_in is a single float

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
    
    allperiod = np.zeros([numiter,1])
    allcorr = np.zeros([numiter,1])
    allheight = np.zeros([numiter,1])    
    
    dtUnity,dt,t0 = UnityInterfaceBrain.gettimestep(bodytype,True)
    maxperiod = 0.5 * 8.5 / 2 / t0

    for j in range(numiter):            
        if seed==None:
            seed1 = None
            seed2 = None
        else:
            seed1 = seed+2*j
            seed2 = seed+2*j+1
        cpg.reset(seed1)
        brain.reset(seed2)
        (pardist, perpdist, heightmean, tiltmean, period, corr, corrind) = UnityInterfaceBrain.evaluate(env,cpg,body_inds,bodytype,dc_in=[dc_in,dc_in],nframes=nframes,brain=brain,outw=outw,outbias=outbias,decay=decay,interactive=True,sound=True,getperiod=True,maxperiod=maxperiod)
        allperiod[j,0] = period
        allheight[j,0] = heightmean[0]
        allcorr[j,0] = corr
            
            
        
    return (np.median(allperiod),np.median(allheight),np.median(allcorr),np.min(allperiod),np.max(allperiod))    


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

    
    if sys.argv[1]=="normal":
        port = 9200
        bodytype = 'ODquad'
        datadir = r'./paper2_data/'
        files = os.listdir(datadir)
        suff = '_final3'
        inpaths = [datadir + file for file in files if 'brain' in file and 'short' not in file and suff in file]
    elif sys.argv[1]=="short":
        port = 9300
        bodytype = 'shortquad'
        datadir = r'./paper2_data/'
        files = os.listdir(datadir)
        suff = '_final3'
        inpaths = [datadir + file for file in files if 'brain' in file and 'short' in file and suff in file]
    elif sys.argv[1]=="hex":
        port = 9400
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
    else:
        n_cpg = 23 #number of parameters in CPG generator
        periodfactor = 0.0481 #paper2 adjustment from CPG time to Unity time
        minheight = 0.75

    kwargs = {
       'numiter': 5,
       'seed': 111
    }

    

    unitypath = UnityInterfaceBrain.getpath('LinuxInt',bodytype)

    inds = []
    allbaseperiod = []
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
        baseperiod = None

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
            if 'Base period' in line:
                baseperiod = float(line.split(':')[1])

        if baseperiod==None:
            raise ValueError('missing base period')
        if len(currcpg)==0:
            raise ValueError('missing body parameters')

        allorigscore.append(currscore) 
        allbaseperiod.append(baseperiod)
        inds.append(currcpg + currbrain)
        goodinpaths.append(path)

        print(path)

    #if len(allbaseperiod)<len(inpaths):
    #    raise ValueError('missing base period')
    #if len(inds)<len(inpaths):
    #    raise ValueError('missing individual')
    
    n_clips = 16

    #each of these will become a 3d array [loop,ind,dc]
    allperiod = np.zeros([n_clips,len(goodinpaths),len(dc_arr)])
    allheight = np.zeros([n_clips,len(goodinpaths),len(dc_arr)])
    allcorr = np.zeros([n_clips,len(goodinpaths),len(dc_arr)])
    allminperiod = np.zeros([n_clips,len(goodinpaths),len(dc_arr)])
    allmaxperiod = np.zeros([n_clips,len(goodinpaths),len(dc_arr)])
    
    
    n_processes = 10 #len(goodinpaths)

    #main loop through all audio stimuli    
    for k in range(n_clips):
        for i,dc in enumerate(dc_arr):                
           function = functools.partial(evalone,n_brain,n_cpg,n_body,bodytype,dc,**kwargs)
           if k==0:
              workers = UnityInterfaceBrain.WorkerPool(function,unitypath,nb_workers=n_processes,port=port+(i%2)*n_processes,clargs = ['-bodytype',bodytype])
           else:
              workers = UnityInterfaceBrain.WorkerPool(function,unitypath,nb_workers=n_processes,port=port+(i%2)*n_processes,clargs = ['-bodytype',bodytype,'-clip',str(k+1)])
           fitnesses = evalall(inds,workers)
        
           currperiod = []
           currheight = []
           currcorr = []
           currmin = []
           currmax = []

           for fitness in fitnesses:
              currperiod.append(fitness[0])
              currheight.append(fitness[1])
              currcorr.append(fitness[2])
              currmin.append(fitness[3])
              currmax.append(fitness[4])
        
           allperiod[k,:,i] = currperiod
           allheight[k,:,i] = currheight
           allcorr[k,:,i] = currcorr
           allminperiod[k,:,i] = currmin
           allmaxperiod[k,:,i] = currmax

           #close down workers
           for ii in range(n_processes):
              workers.addtask(None)
           workers.join()
           workers.terminate()
        
           time.sleep(5)
    
        
    alloutput = (np.squeeze(allperiod),np.squeeze(allheight),np.squeeze(allcorr),np.squeeze(allminperiod),np.squeeze(allmaxperiod))
    outnames = ['period','height','corr','minperiod','maxperiod']
    
    #export data
    for kk,path in enumerate(goodinpaths):
        outpath = path.replace(datadir,outdir).replace(suff,'_audio2_info')
        with open(outpath,'w') as outfile:
            outfile.writelines('cpg:' + str(inds[kk][:(n_cpg+n_body)])+'\n')
            outfile.writelines('brain:' + str(inds[kk][(n_cpg+n_body):])+'\n')
            outfile.writelines('base period:' + str(allbaseperiod[kk]*periodfactor)+'\n')
            outfile.writelines('original score:' + str(allorigscore[kk])+'\n')
        for jj,name in enumerate(outnames):
           outpath = path.replace(datadir,outdir).replace(suff,'_audio2_'+name)
           with open(outpath,'w') as outfile:    
               arr = np.array(alloutput[jj])
               for i in range(len(arr)):
                   outfile.writelines(str(list(arr[i,kk,:])).replace('[','').replace(']','')+'\n')



