# -*- coding: utf-8 -*-
"""
Created on Fri May  5 23:40:35 2023

Generates Figure 5a. Must be run from virtual environment

@author: alexansz
"""

import time
import hebb
import evoplot
import numpy as np
import SimBodies
import matsuoka_quad
import matsuoka_hex
import matsuoka_brain
import MathUtils
import ControllerFuncs
import UnityInterfaceBrain
from mlagents_envs.environment import UnityEnvironment


def getdata(inpaths):

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

        if 'hex' in path:
            minheight=1.0
        else:
            minheight=0.75

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
        
        origpath = '_'.join(path.split('_')[:-1]) + '.txt'
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
    
    indlist = []
    with open(r'./paper4_data/singlet_adaptthresh4_indlist.txt','r') as f:
       lines = f.readlines()
       for line in lines:
           path = line.split('\n')[0].split(',')[0].split('/')
           indlist.append(path[-2] + '/' + path[-1])
    
    inds,indfiles = getdata(indlist)           
    
    n_brain = 6    
    n_body = 9 #number of body parameters
    


        
    unitypath = r'../Hebb/My project.exe'
         
    dc = 0.5
    nframes = 2000
            
    studentindnum = 1

    ind = inds[studentindnum]
    
    studentfile = indfiles[studentindnum]

    if 'hex' in studentfile:
        bodytype = 'AIRLhex'
        basefile = studentfile.replace('unityhex','unity_hex').replace('final.txt','').replace('brain','cpg').split('/')[-1]
    else:
        bodytype = 'shortquad'
        basefile = studentfile.replace('final3.txt','').replace('brain','cpg').split('/')[-1]
    if 'hex' in bodytype:
        n_cpg = 28
        tilt = 1
    else:
        n_cpg = 23 #number of parameters in CPG generator
        tilt = 0.015
        


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
    
    
    startframe = round(nframes/2)
    allspeed = []
    allrot = []
    allperiod = []
    dc_arr = np.arange(0,1.01,0.05)
    corrstart=0.5
    dt_real,dt,t0 = UnityInterfaceBrain.gettimestep(bodytype,True)
    
    for dc in dc_arr:
       env = UnityEnvironment(file_name=unitypath, seed = 4, side_channels = [], worker_id=0, no_graphics=True, additional_args = ["-bodytype",bodytype])       
       hebb.run(env,controller,300)
       controller.cpg.reset(111)
       controller.brain.reset(222)       
       _,_,allcpgout,_,_,_,speed,rot,_,_ = hebb.run(env,controller,nframes,dc,tilt) 
       allspeed.append(np.mean(speed[startframe:]))
       allrot.append(np.mean(rot[startframe:]))
       period,_ = MathUtils.autocorr(allcpgout, corrstart, mindelay=round(0.1/dt_real))
       allperiod.append(period*dt_real) 
       env.close()
       time.sleep(1)

    with open('./paper4_data/' + basefile + 'speedrot.txt','w') as f:
        f.writelines(','.join([str(x) for x in dc_arr]) + '\n')
        f.writelines(','.join([str(x) for x in allspeed]) + '\n')
        f.writelines(','.join([str(x) for x in allrot]) + '\n')
        f.writelines(','.join([str(x) for x in allperiod]))
        f.close()
        