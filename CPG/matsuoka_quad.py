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
       'dt': 0.04,
       'b': 0.2*p[0],       # 0.1
       'gam': 0.1*p[1],     # 0.03
       'x0': p[2],          # 0.5
       'c': 5*p[3],      #-10+20*p[3],    # 3
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
    
    initx = -np.random.rand(4,3)
    
    
    #weight of external input to each controller
    inpc = np.array([1,1,1,1])
    
    #weight of external input to each neuron
    inpw = np.array([1,1,1])
    


    cons = [matsuoka_ext.Controller(i,initx[i],innerweights,param,bias,drive,intn,inpw) for i in range(len(initx))]
    rob = roborun.Robot(cons,param,adj,intn,inpc)

    return rob


#this function lets you play around with setting parameters manually
def manual_param():
    #m oscillators with n dimensions each
    
    
    ####shared constants
    param = {
       'dt': 0.04,
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
    initx = -np.random.rand(4,3)
    #np.array([[0.1,0,0.1],[0.1,0,0.2],[0.1,0,0.3],[0.1,0,0.4]])
    
    #weight of external input to each controller
    inpc = np.array([1,1,1,1])
    
    #weight of external input to each neuron
    inpw = np.array([1,1,1])
    
    
    
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

#interface for genetic algorithm. in=array, out=score
def run_from_array(ind,plot=False):   
    tt = 40000
    z = np.zeros([tt,1])

    #tonic input, one value per simulation
    d_arr = np.arange(0,1.1,0.1)
    rob = array2param(ind)
    score = roborun.runcpg(rob,z,d_arr,plot)

    return score




if __name__ == "__main__":
    if len(sys.argv) > 1:
        ind = [x.split(',')[0] for x in sys.argv[1:]]
        rob = array2param(ind)
    else: #direct from python. insert CPG array here to check its period vs DC input
        ind = [4, 2, 2, 10, 3, 5, 3, 10, 8, 10, 10, 3, 2, 5, 9, 5, 2, 10, 9, 7, 1, 5, 8]
        
        rob = array2param(ind)
        #rob = manual_param()

    tt = 20000
    z = np.zeros([tt,1])
    d_arr = np.arange(0,1.1,0.1)

    score = roborun.runcpg(rob,z,d_arr,plot=True)
    print(score)




