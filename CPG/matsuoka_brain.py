# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 16:15:22 2021

@author: alexansz
"""   

import matsuoka_quad
import matsuoka_ext
import roborun
import numpy as np

def array2brain(n,m,ind):
    p = [float(x)/10 for x in ind]
    param = {
       'dt': 0.04,
       'b': 0.3,
       'gam': 0.03,
       'x0': 1,
       'c': 4,
       'a': 2
    }
    outbias = -0.15 # (-) maximum steady state output for above parameters

    #size of p depends on n and m    
    
    #outbias = 4*p.pop() - 4
    decay = 0.05 + 0.5*p.pop()
    
    bias = np.array([2.5 - 0.5*p.pop()])
    inpc = np.array([-1+2*p.pop() for i in range(n)]) #nx1

    adj = np.zeros([n,n])
    
    amax = 6 / (n-1) #2 / (n-1)
    for i in range(n):
        for j in range(n):
            if i!=j:
                adj[i,j] = -amax + amax*p.pop() #only inhibitory  
    
    
    #remaining items are brain to body connection
    if len(p) == n*m:
       outw = np.array(p).reshape(m,n)
       outw = -10 + 20*outw
    else:
       raise ValueError(f'p is the wrong length. need {m*n} but {len(p)} left')
       
    innerweights = np.array([[0]])
    drive = np.array([0])
    inpw = np.array([1])
    initx = -np.array([[np.random.rand()] for i in range(n)]) #nx1
    intn = 0
    

    cons = [matsuoka_ext.Controller(i,initx[i],innerweights,param,bias,drive,intn,inpw) for i in range(n)]
    brain = roborun.Robot(cons,param,adj,intn,inpc)
    
    return brain, outw, decay, outbias


def run_from_array(n,body,baseperiod,ind,plot=False,skipevery=-1):    
        
    m = len(body.cons)
    brain,outw,decay,outbias = array2brain(n,m,ind)
    
    #simulation params
    dc_in = 0.5
    #t_arr = np.array([60,110,160]) #np.arange(50,160,20)
    t_arr = baseperiod*np.array([2/3,1,3/2])
    tt = 40000

    score = roborun.runbrain(body,brain,outw,outbias,tt,[0,1],t_arr,dc_in,decay,plot=plot,skipevery=skipevery)
    #print(score)
    
    return score


def finaleval(n,bodyarray,brainarray,periods,niter,plot=False,skipevery=-1):
    for i,ind in enumerate(brainarray):
       body = matsuoka_quad.array2param(bodyarray[i])
       baseperiod = periods[i]
       
       scorearr = []
       for j in range(niter):
           score = run_from_array(n,body,baseperiod,ind,plot=plot,skipevery=skipevery)
           scorearr.append(np.array(score))
       
       medscore = np.median(np.array(scorearr),axis=0)
       print(medscore)
       print(np.mean(medscore))
    


    

if __name__ == "__main__":
    
    do_eval     = True # evaluates x 'numiters' and prints median scores 
    plotall     = True
    num_iters   = 2
    skip_every  = 4 #skip every nth pulse for evaluation. set to -1 to use isochronous   
    
    n = 4 # number of neurons. currently can be 2 or 4
    m = 4 # number of limbs in CPG. currently only 4
        
    bodyarray = [4, 2, 2, 10, 3, 5, 3, 10, 8, 10, 10, 3, 2, 5, 9, 5, 2, 10, 9, 7, 1, 5, 8]
    brainarray = [8, 8, 0, 4, 8, 9, 0, 1, 7, 8, 0, 6, 7, 7, 0, 6, 8, 8, 8, 9, 6, 10, 4, 6, 9, 6, 9, 9, 2, 10, 3, 1, 10, 9]
    period = 180.0
    
    body = matsuoka_quad.array2param(bodyarray)
    brain,outw,decay,outbias = array2brain(n,m,brainarray)
    
    if do_eval:
        finaleval(n,[bodyarray],[brainarray],[period],num_iters,plot=plotall,skipevery=skip_every)
    
    
    ##### transient response i.e. pulsing input in middle of time series

    skip_every  = 4 #skip every nth pulse for time series visualisation. set to -1 to use isochronous
    dc_in = 0.5    
    ratio = 0.8    
    amp = 0.7
    tt = 160000    
    t_bounds = [0.2,0.4,0.8] #first cuts off output before x times tt. Others define input start and stop times
    
    t = period*ratio
    newsim,times,outperiods,_,_ = roborun.periodvstime(body,brain,outw,outbias,tt,t_bounds,t,amp,dc_in,plot=True,skipevery=skip_every)


    

