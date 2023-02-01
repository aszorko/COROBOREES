# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 15:55:27 2021

This script runs either:
    -a manual configuration of a Matsuoka hexapod (when running directly)
    -an automatic configuration from an array of 28 floats defined on [0,1]
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
        rob = manual_param()

    tt = 20000
    dt = 0.04
    z = np.zeros([tt,1])
    d_arr = np.arange(0,1.1,0.1)


    x = x = np.arange(-2,2,0.02)
    nx = rob.cons[0].nullclinex(x,0,0.5)
    ny = rob.cons[0].nullcliney(x,0,0.5)
    plt.plot(nx,linewidth=2)
    plt.plot(ny,linewidth=2)
    plt.xticks([])
    plt.yticks([])
    plt.ylim([-1,4])

