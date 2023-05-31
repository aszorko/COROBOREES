# -*- coding: utf-8 -*-
"""
Created on Fri May  5 23:40:35 2023

Generates Figure 5a. Must be run from virtual environment

@author: alexansz
"""

import run_hebb
import hebb
import evoplot
import numpy as np
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
    bodytype = 'AIRLhex'
    
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
        
    
    teacherindnum = 3
    studentindnum = 1



    teacherind = inds[teacherindnum]
    studentinds = [inds[studentindnum]]

    if 'hex' in indfiles[teacherindnum]:
        bodytype = 'AIRLhex'
    else:
        bodytype = 'shortquad'
    if 'hex' in bodytype:
        n_cpg = 28
        tilt = 1
    else:
        n_cpg = 23 #number of parameters in CPG generator
        tilt = 0.015

    env = UnityEnvironment(file_name=unitypath, seed = 4, side_channels = [], worker_id=0, no_graphics=False, additional_args = ["-bodytype",bodytype])
    out = run_hebb.runteacher(n_brain,n_cpg,n_body,bodytype,nframes,dc,tilt,env,teacherind)      
    teacherout = (-1,bodytype,out[0],out[1])
    env.close()
    
    if 'hex' in indfiles[studentindnum]:
        bodytype = 'AIRLhex'
    else:
        bodytype = 'shortquad'
    if 'hex' in bodytype:
        n_cpg = 28
        tilt = 1
    else:
        n_cpg = 23 #number of parameters in CPG generator
        tilt = 0.015
        
    env = UnityEnvironment(file_name=unitypath, seed = 4, side_channels = [], worker_id=0, no_graphics=False, additional_args = ["-bodytype",bodytype])
    output = run_hebb.evalstudent(studentinds,n_brain,n_cpg,n_body,bodytype,nframes,dc,tilt,env,teacherout,plot=True)      
 
    outputarr = np.array(output)
    print(outputarr)
    
    env.close()
 