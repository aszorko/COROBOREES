# -*- coding: utf-8 -*-
"""
Runs robots against outputs from all other robots in a set
Uses multiprocessing

@author: alexansz
"""

import functools
import evoplot
import numpy as np
import UnityInterfaceBrain
import ControllerFuncs
import SimBodies
import MathUtils
import matsuoka_quad
import matsuoka_hex
import matsuoka_brain
import hebb
import os
import time


outputfiles = ['mean_input','free_height','free_period','free_autocorr','free_target_diff','sync_height','sync_period','sync_autocorr','sync_target_diff','sync_free_diff','learn_height','learn_period','learn_autocorr','learn_target_diff','learn_free_diff','learn_sync_diff','feedback_target_diff','mean_feedback']

def evalstudent(inds,n_brain,n_cpg,n_body,bodytype,nframes,dc,tilt,env,target,**kwargs):

    index_in = target[0]
    teacher_bodytype = target[1]
    z_in = target[2]
    period_in = target[3]
    
    alloutputs = []
    
    #test all agents against target    
    for i,ind in enumerate(inds):
        
        cpg_inds = ind[:n_cpg]
        body_inds = ind[n_cpg:(n_cpg+n_body)]
        brain_inds = ind[(n_cpg+n_body):]
        
        if 'hex' in bodytype:
            body = SimBodies.HexBody(bodytype,body_inds)
            cpg = matsuoka_hex.array2param(cpg_inds)
            m=6
        else:
            body = SimBodies.QuadBody(bodytype,body_inds)
            cpg = matsuoka_quad.array2param(cpg_inds)
            m=4
        
        brain,outw,decay,outbias = matsuoka_brain.array2brain(n_brain,m,brain_inds)
        
        controller = ControllerFuncs.Controller(body,cpg,brain,outw,decay,outbias,bodytype)
        
        if i==0:
            #warm up
            hebb.run(env,controller,300)
            
        if i==index_in and bodytype==teacher_bodytype:
            alloutputs.append([-1 for j in range(len(outputfiles))])
            continue

        controller.cpg.reset(111)
        controller.brain.reset(222)
    
        outputs = hebb.main(env,controller,bodytype,z_in,period_in,nframes,dc,tilt,seed=333+len(inds)*2*index_in+2*i,**kwargs)
        alloutputs.append(list(outputs))
        
    return alloutputs    


def runteacher(n_brain,n_cpg,n_body,bodytype,nframes,dc,tilt,env,ind):

    dt_unity = 0.015
    stepsperframe = 2
    
    cpg_inds = ind[:n_cpg]
    body_inds = ind[n_cpg:(n_cpg+n_body)]
    brain_inds = ind[(n_cpg+n_body):]
    
    if 'hex' in bodytype:
        body = SimBodies.HexBody(bodytype,body_inds)
        cpg = matsuoka_hex.array2param(cpg_inds)
        m=6
    else:
        body = SimBodies.QuadBody(bodytype,body_inds)
        cpg = matsuoka_quad.array2param(cpg_inds)
        m=4
    
    brain,outw,decay,outbias = matsuoka_brain.array2brain(n_brain,m,brain_inds)
    
    controller = ControllerFuncs.Controller(body,cpg,brain,outw,decay,outbias,bodytype)
    
    #warm up
    hebb.run(env,controller,300)

    controller.cpg.reset(111)
    controller.brain.reset(222)

    _,_,allx,allsensors,_,_,_,_ = hebb.run(env,controller,nframes,dc=dc,tilt=tilt)
    
    period,_ = MathUtils.autocorr(allx,0.33,round(0.1/dt_unity),maxdelay=round(2.0/dt_unity))
    
    sensoramp = 50.0 #200.0 / m

    return (sensoramp*np.sum(1+allsensors,axis=0)/2,period*dt_unity)    



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




def getdata(bodytype):
    if bodytype =="ODquad":
        datadir = r'../CPG/paper2_data/'
        files = sorted(os.listdir(datadir))
        suff = '_final3'
        minheight = 0.75
        inpaths = [datadir + file for file in files if 'brain' in file and 'short' not in file and suff in file]
    elif bodytype =="shortquad":
        datadir = r'../CPG/paper2_data/'
        files = sorted(os.listdir(datadir))
        suff = '_final3'
        minheight = 0.75
        inpaths = [datadir + file for file in files if 'brain' in file and 'short' in file and suff in file]
    elif bodytype=="AIRLhex":
        datadir = r'../CPG/hexdata/'
        files = sorted(os.listdir(datadir))
        suff = '_final'
        minheight = 1.0
        inpaths = [datadir + file for file in files if 'brain' in file and suff in file]        
    else:
        raise ValueError("invalid body type")

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
        #currheight = np.nan
        currbrain = []
        while i<len(bscores):
           i += 1
           if np.sum(np.array(bscores[-i])[::2]<minheight)==0:
               #currheight = np.mean(bscores[-i][::2])
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
        
    return inds,goodinpaths

if __name__ == "__main__":


    outdir = r'./paper4_data/'
    outprefix = 'singlet_inhib'

    bodytypes = ['shortquad','AIRLhex']       
    
    port = 9600
    
    dc = 0.5
    n_brain = 6    
    n_body = 9 #number of body parameters

    kwargs = {}
     

    #alloutput = np.zeros([len(inds),len(inds),len(outputfiles)])    
    
    n_processes = 10 #len(inds)
    
    nframes = 4000
    

    teachers = []
    allindfiles = []
    allinds = []

    for bodytype in bodytypes:
       inds,indfiles = getdata(bodytype)
       allinds.append(inds)

       if 'hex' in bodytype:
          n_cpg = 28
          tilt = 1.0
          unitypath = r'./Unity/LinuxHebbHex.x86_64'
       else:
          n_cpg = 23 #number of parameters in CPG generator
          tilt = 0.015
          unitypath = r'./Unity/LinuxHebb.x86_64'

       #get outputs and periods from all individuals
       function = functools.partial(runteacher,n_brain,n_cpg,n_body,bodytype,nframes,dc,tilt,**kwargs)      
       workers = UnityInterfaceBrain.WorkerPool(function,unitypath,nb_workers=n_processes,port=port,clargs = ['-bodytype',bodytype])
       teachersout = evalall(inds,workers)
    
       #close down workers
       for ii in range(n_processes):
          workers.addtask(None)
       workers.join()
       workers.terminate()

       time.sleep(5)

       for i,out in enumerate(teachersout):
          teachers.append((i,bodytype,out[0],out[1]))
          allindfiles.append(indfiles[i])

    alloutput = []

    for i,bodytype in enumerate(bodytypes):
       if 'hex' in bodytype:
          n_cpg = 28
          tilt = 1.0
          unitypath = r'./Unity/LinuxHebbHex.x86_64'
       else:
          n_cpg = 23 #number of parameters in CPG generator
          tilt = 0.015
          unitypath = r'./Unity/LinuxHebb.x86_64'
        
       #make each individual a teacher
       function = functools.partial(evalstudent,allinds[i],n_brain,n_cpg,n_body,bodytype,nframes,dc,tilt,**kwargs)      
       workers = UnityInterfaceBrain.WorkerPool(function,unitypath,nb_workers=n_processes,port=port+n_processes,clargs = ['-bodytype',bodytype])
       output = evalall(teachers,workers)
       for out in output:
           alloutput.append(out)

       #close down workers
       for ii in range(n_processes):
          workers.addtask(None)
       workers.join()
       workers.terminate()
 
       time.sleep(5)
    
    outputarr = np.array(alloutput)
        
    outpath = outdir + outprefix + '_indlist.txt'
    with open(outpath,'w') as outfile:
        for i,filen in enumerate(allindfiles):
            outfile.writelines(filen+ ', ' + str(teachers[i][3]) + ', ' +  str(np.mean(teachers[i][2])) + '\n')
    #export data
    for jj,name in enumerate(outputfiles):
       outpath = outdir + outprefix + '_' + name + '.txt'
       with open(outpath,'w') as outfile:    
          arr = outputarr[:,:,jj].squeeze()
          for i in range(len(arr)):
              outfile.writelines(str(list(arr[i,:])).replace('[','').replace(']','')+'\n')



