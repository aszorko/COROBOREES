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

def array2param(ind):
    #ind is list of integers between 0 and 10
    p = [float(x)/10 for x in ind]
    #now all elements of p are on interval [0,1]
    
    param = {
       'dt': 0.02,
       'b': 0.2*p[0],       # 0.1
       'gam': 0.1*p[1],     # 0.03
       'x0': p[2],          # 0.5
       'c': -10+20*p[3],    # 3
       'a': 2*p[4],         # 1
    }
    
    
    #### robot layout
    
    n0 = -2.2 #2
    n1 = 2
    
    # vertical connections
    a_b = n0+n1*p[5]
    a_f = n0+n1*p[6]
    a = np.array([[0,a_b],[a_f,0]])
    
    #lateral connections (off-diagonal=diagonal)
    c = np.array([[n0+n1*p[7],n0+n1*p[8]],[n0+n1*p[9],n0+n1*p[10]]])     
    
    #adjacency matrix m*m
    adj = np.block([[a,c],[c,a]]) #np.array([[0,1],[1,0]])
    
    
    #### module setup
    
    #index of interneuron
    intn = 2
    
    #intercept from 1 to 2
    bias = 1 + np.array([p[11],p[12],p[13]])
    
    #tonic coefficient from -1 to 1
    drive = 2 * np.array([p[14]-0.55,p[15]-0.55,p[16]-0.55]) #p-0.5
    
    
    n2=-2.2 #2
    n3=4
    
    innerweights = np.array([[0,n0+n1*p[17],n2+n3*p[18]],[n0+n1*p[19],0,n2+n3*p[20]],[n2+n3*p[21],n2+n3*p[22],0]])
    
    initx = np.random.rand(4,3)
    
    #initial conditions m*n
    #initx = np.array([[0.1,0,0.1],[0.1,0,0.2],[0.1,0,0.3],[0.1,0,0.4]])
    
    #weight of external input to each controller
    inpc = np.array([1,1,1,1])
    
    #weight of external input to each neuron
    inpw = np.array([0,0,1])
    


    cons = [matsuoka_ext.Controller(i,initx[i],innerweights,param,bias,drive,intn,inpw) for i in range(len(initx))]
    rob = roborun.Robot(cons,param,adj,intn,inpc)

    return rob


def manual_param():
    #m oscillators with n dimensions each
    
    
    ####shared constants
    param = {
       'dt': 0.02,
       'b': 0.1,
       'gam': 0.03,
       'x0': 0.4,
       'c': 3.3,
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
    initx = np.random.rand(4,3)
    #np.array([[0.1,0,0.1],[0.1,0,0.2],[0.1,0,0.3],[0.1,0,0.4]])
    
    #weight of external input to each controller
    inpc = np.array([1,1,1,1])
    
    #weight of external input to each neuron
    inpw = np.array([0,0,1])
    
    
    
    #### robot layout
    
    # vertical connections
    a_b = -2.1
    a_f = -2.1
    a = np.array([[0,a_b],[a_f,0]])
    
    #lateral connections (off-diagonal=diagonal)
    c = np.array([[-1.9,-2],[-2,-1.9]])     
    
    #adjacency matrix m*m
    adj = np.block([[a,c],[c,a]]) #np.array([[0,1],[1,0]])
    
    
       
    
    
    cons = [matsuoka_ext.Controller(i,initx[i],innerweights,param,bias,drive,intn,inpw) for i in range(len(initx))]
    rob = roborun.Robot(cons,param,adj,intn,inpc)


    return rob


def run_from_array(ind,plot=False):
    #interface for genetic algorithm. in=array, out=score
    tt = 40000
    z = np.zeros([tt,1])

    #tonic input, one value per simulation
    d_arr = np.arange(0,1.1,0.1)
    rob = array2param(ind)
    score = roborun.stepdrive(rob,z,d_arr,plot)
    #print(score)
    return score




if __name__ == "__main__":
    if len(sys.argv) > 1:
        ind = [x.split(',')[0] for x in sys.argv[1:]]
        rob = array2param(ind)
    else: #direct from python
        ind = [10, 1, 1, 9, 8, 4, 9, 2, 8, 5, 1, 1, 1, 4, 6, 4, 1, 5, 10, 6, 10, 5, 4] #0.899
        #ind = [7, 4, 2, 6, 8, 8, 9, 2, 10, 5, 2, 3, 2, 1, 10, 10, 1, 4, 7, 7, 5, 1, 2] #0.832
        #ind = [8, 9, 8, 9, 6, 1, 3, 2, 8, 7, 5, 3, 3, 2, 5, 5, 6, 4, 9, 5, 7, 1, 1] #0.782
        #ind = [10, 3, 6, 6, 5, 10, 6, 3, 4, 4, 1, 3, 1, 6, 9, 4, 10, 6, 3, 1, 5, 3, 6] #0.646
        rob = array2param(ind)
        #rob = manual_param()

    tt = 40000
    z = np.zeros([tt,1])
    d_arr = np.arange(0,1,0.1)

    score = roborun.stepdrive(rob,z,d_arr,plot=True)
    print(score)

##running from python
#if len(sys.argv) == 1:
#    score = manual_param()
#running from command line
#else:
#    score = array2param(sys.argv[1:])




