# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 15:55:27 2021

This script runs either:
    -a manual configuration of a Matsuoka quadruped (when running directly)
    -an automatic configuration from an array of 23 floats defined on [0,1]
     this array is given as input at the command line, and is catered for a genetic algorithm
    
@author: alexansz
"""

import matsuoka_ext
import roborun
import numpy as np
import sys
import matplotlib.pyplot as plt

def array2param(ind):
    #ind is list of integers between 0 and 10
    p = [float(x)/10 for x in ind]
    #now all elements of p are on interval [0,1]
    
    param = {
       'b': 0.2*p.pop(),       # 0.1
       'gam': 0.1*p.pop(),     # 0.03
       'x0': p.pop(),          # 0.5
       'c': 5*p.pop(),      #-10+20*p[3],    # 3
       'a': 2*p.pop(),         # 1
    }
    
    
    #### robot layout
    
    n0 = -2.2 #2
    n1 = 2
    
    # vertical connections
    a_b = n0+n1*p.pop()
    a_f = n0+n1*p.pop()
    b_b = n0+n1*p.pop()
    b_f = n0+n1*p.pop()
    a = np.array([[0,a_b,0],[a_f,0,b_b],[0,b_f,0]])
    
    #lateral connections (off-diagonal=diagonal)
    c = np.array([[n0+n1*p.pop(),n0+n1*p.pop(),0],[n0+n1*p.pop(),n0+n1*p.pop(),n0+n1*p.pop()],[0,n0+n1*p.pop(),n0+n1*p.pop()]])     
    
    #adjacency matrix m*m
    adj = np.block([[a,c],[c,a]]) #np.array([[0,1],[1,0]])
    
    
    #### module setup
    
    #index of interneuron
    intn = 2
    
    #intercept from 1 to 2
    bias = 1 + np.array([p.pop(),p.pop(),p.pop()])
    
    #tonic coefficient from -1 to 1
    drive = 2 * np.array([p.pop()-0.55,p.pop()-0.55,p.pop()-0.55]) #p-0.5
    
    n2=-2.2 #2
    n3=4
    
    innerweights = np.array([[0,n0+n1*p.pop(),n2+n3*p.pop()],[n0+n1*p.pop(),0,n2+n3*p.pop()],[n2+n3*p.pop(),n2+n3*p.pop(),0]])
    
    initx = -np.random.rand(6,3)
        
    #weight of external input to each neuron
    inpw = np.array([1,1,1])
    
    ####
    
    #weight of external input to each controller
    inpc = np.array([1 for i in range(len(initx))])

    cons = [matsuoka_ext.Controller(i,initx[i],innerweights,param,bias,drive,intn,inpw) for i in range(len(initx))]
    rob = roborun.Robot(cons,param,adj,intn,inpc)

    return rob


def manual_param():
    #m oscillators with n dimensions each
    
    
    ####shared constants
    param = {
       'b': 0.1,
       'gam': 0.03,
       'x0': 0.4,
       'c': 0,
       'a': 1
    }
    
    
    #### module setup
    
    #index of interneuron
    intn = 2
    
    #"input" independent of tonic
    bias = np.array([0.8,0.8,0.8])
    
    #coefficient for tonic input
    drive = np.array([1,1,1])
    
    innerweights = np.array([[0,-2,0],[-2,0,0],[0,0,0]])
    
    #initial conditions m*n
    initx = -np.random.rand(6,3)
    #initx = np.array([[0.1,0,0.1],[0.1,0,0.2],[0.1,0,0.3],[0.1,0,0.4]])
    
    #weight of external input to each controller
    inpc = np.array([1,1,1,1,1,1])
    
    #weight of external input to each neuron
    inpw = np.array([1,1,1])
    
    
    
    #### robot layout
    
    # vertical connections
    a_b = -2.1
    a_f = -2.1
    b_b = -2.1
    b_f = -2.1
    a = np.array([[0,a_b,0],[a_f,0,b_b],[0,b_f,0]])
    
    #lateral connections (off-diagonal=diagonal)
    c = np.array([[-1.9,-2,0],[-2,-1.9,-2],[0,-2,-1.9]])     
    
    #adjacency matrix m*m
    adj = np.block([[a,c],[c,a]]) #np.array([[0,1],[1,0]])
    
    
       
    
    
    cons = [matsuoka_ext.Controller(i,initx[i],innerweights,param,bias,drive,intn,inpw) for i in range(len(initx))]
    rob = roborun.Robot(cons,param,adj,intn,inpc)


    return rob


def run_from_array(ind,plot=False):
    #interface for genetic algorithm. in=array, out=score
    tt = 40000 #number of time steps
    dt = 0.04  #size of time step
    z = np.zeros([tt,1])

    #tonic input, one value per simulation
    d_arr = np.arange(0,1.1,0.1)
    rob = array2param(ind)
    score = roborun.runcpg(rob,z,d_arr,dt=dt,plot=plot)
    #print(score)
    return score




if __name__ == "__main__":
    if len(sys.argv) > 1:
        ind = [x.split(',')[0] for x in sys.argv[1:]]
        rob = array2param(ind)
    else: #direct from python
        #vanilla GA, large c range
        #ind = [10, 1, 1, 9, 8, 4, 9, 2, 8, 5, 1, 1, 1, 4, 6, 4, 1, 5, 10, 6, 10, 5, 4] #0.899
        #ind = [7, 4, 2, 6, 8, 8, 9, 2, 10, 5, 2, 3, 2, 1, 10, 10, 1, 4, 7, 7, 5, 1, 2] #0.832
        #ind = [8, 9, 8, 9, 6, 1, 3, 2, 8, 7, 5, 3, 3, 2, 5, 5, 6, 4, 9, 5, 7, 1, 1] #0.782
        #ind = [10, 3, 6, 6, 5, 10, 6, 3, 4, 4, 1, 3, 1, 6, 9, 4, 10, 6, 3, 1, 5, 3, 6] #0.646
        #NSGA, small c range
        #ind = [6, 2, 6, 1, 10, 10, 10, 2, 4, 5, 1, 2, 6, 7, 7, 9, 0, 7, 8, 4, 1, 8, 5]
        #take 2, no zero values
        #ind = [9, 3, 7, 10, 10, 5, 5, 7, 3, 9, 8, 8, 1, 9, 3, 4, 2, 2, 1, 1, 6, 7, 4]
        #ind = [3, 2, 6, 2, 6, 8, 8, 1, 4, 2, 3, 4, 7, 5, 7, 7, 5, 1, 3, 10, 9, 8, 3]
        #only positive c
        #ind = [5, 3, 4, 7, 3, 4, 6, 9, 7, 2, 9, 5, 1, 5, 6, 8, 10, 3, 5, 10, 8, 4, 4]
        #ind = [9, 6, 5, 2, 4, 3, 3, 4, 5, 8, 8, 5, 6, 9, 9, 9, 5, 3, 5, 6, 2, 1, 2]
        #inverted duty function (h(A)-h(B)<0)
        #ind = [8, 6, 9, 6, 8, 4, 3, 7, 1, 10, 2, 9, 6, 8, 8, 9, 3, 6, 8, 10, 2, 1, 2]
        #ind = [9, 10, 4, 8, 8, 6, 3, 8, 2, 10, 8, 10, 5, 7, 1, 8, 3, 6, 8, 10, 1, 1, 7]
        #inverted duty function 2 (A<0)
        #ind = [2, 2, 10, 7, 6, 4, 8, 7, 7, 1, 7, 8, 7, 8, 8, 10, 9, 1, 6, 6, 4, 4, 1]
        #ind = [6, 1, 3, 6, 6, 9, 6, 6, 10, 5, 4, 1, 1, 7, 9, 6, 2, 5, 4, 1, 4, 4, 1]
        #ind = [6, 1, 1, 7, 8, 7, 6, 1, 7, 8, 7, 4, 10, 6, 5, 6, 10, 3, 1, 2, 10, 10, 5]
        #ind = [6, 1, 1, 7, 6, 6, 4, 9, 1, 6, 10, 3, 8, 2, 10, 6, 10, 3, 1, 2, 10, 9, 5]
        #ind = [6, 1, 1, 7, 6, 6, 6, 3, 1, 6, 4, 3, 9, 5, 10, 9, 8, 3, 1, 2, 10, 6, 5]
        #ind = [6, 1, 3, 6, 6, 9, 6, 6, 10, 5, 4, 1, 1, 5, 10, 9, 2, 5, 4, 1, 4, 4, 1]
        #inverted duty function 3 (A<0 and B<0)
        #ind = [8, 3, 1, 5, 10, 3, 10, 2, 10, 2, 3, 3, 2, 5, 10, 9, 1, 2, 4, 7, 1, 3, 1]
        #ind = [5, 1, 2, 6, 9, 6, 10, 2, 3, 1, 1, 7, 10, 2, 2, 9, 1, 7, 9, 6, 6, 4, 9]
        #ind = [10, 1, 2, 3, 5, 3, 10, 2, 4, 3, 8, 6, 7, 4, 9, 9, 9, 10, 1, 6, 1, 3, 3]
        #ind = [10, 1, 2, 3, 5, 3, 7, 5, 7, 3, 3, 5, 7, 5, 9, 9, 9, 10, 1, 6, 1, 3, 8]
        #ind = [8, 3, 1, 5, 10, 3, 9, 4, 4, 6, 6, 7, 7, 4, 10, 9, 1, 7, 9, 6, 7, 3, 1]
        #ind = [10, 2, 1, 5, 10, 3, 9, 4, 8, 7, 3, 3, 2, 5, 10, 9, 1, 2, 4, 7, 1, 3, 1]
        #differential output
        #ind = [1, 4, 7, 2, 4, 5, 1, 2, 4, 6, 4, 9, 9, 5, 10, 10, 9, 8, 6, 3, 5, 5, 6]
        #ind = [4, 8, 9, 5, 7, 1, 5, 1, 1, 7, 4, 4, 7, 6, 8, 7, 5, 3, 6, 4, 4, 5, 10]
        ind = [4, 1, 4, 4, 10, 4, 2, 8, 10, 10, 10, 4, 3, 7, 5, 7, 9, 9, 9, 3, 3, 5, 4]
        #ind = [4, 2, 4, 4, 10, 4, 2, 10, 7, 4, 8, 1, 1, 7, 6, 5, 10, 10, 6, 5, 1, 5, 8]
        #ind = [4, 2, 2, 10, 3, 5, 3, 10, 8, 10, 10, 3, 2, 5, 9, 5, 2, 10, 9, 7, 1, 5, 8]

        #rob = array2param(ind)
        rob = manual_param()

    tt = 20000
    dt = 0.04
    z = np.zeros([tt,1])
    d_arr = np.arange(0,1.1,0.1)

    #score = roborun.runcpg(rob,z,d_arr,dt=dt,plot=True)
    #print(score)


    x = x = np.arange(-2,2,0.02)
    nx = rob.cons[0].nullclinex(x,0,0.5)
    ny = rob.cons[0].nullcliney(x,0,0.5)
    plt.plot(nx,linewidth=2)
    plt.plot(ny,linewidth=2)
    plt.xticks([])
    plt.yticks([])
    plt.ylim([-1,4])


##running from python
#if len(sys.argv) == 1:
#    score = manual_param()
#running from command line
#else:
#    score = array2param(sys.argv[1:])

