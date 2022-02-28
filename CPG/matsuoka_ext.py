# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 10:37:54 2021

Extended Matsuoka controller model with y*sig(x) function
n oscillators containing 1 interneuron

global params (in "params" dict):
a:  overall coefficient of sigmoidal interaction in dx
b:  coefficient of hinge/ReLU function in dy
c:  steepness of sigmoid (0 -> standard Matsuoka)
x0: centre of sigmoid
gam: damping rate of y
dt: time constant

per-oscillator params:

bias: constant input independent of dc_in
drive: coefficient for additional tonic dc_in
inpw: coefficient for fast input z in fb_ext
    
@author: alexansz
"""

import numpy as np

def rot(a,x,y):
    return np.array([(x*np.cos(a)-y*np.sin(a)),(y*np.cos(a)+x*np.sin(a))])

def sig(x):
    return 1 / (1 + np.exp(-x))

def hinge(x):
    y = 0*x
    for i in range(len(x)):
        if x[i] < 0:
            y[i] = 0
        else:
            y[i] = x[i]            
    return y


class Controller:
    #contains oscillator equations
    def __init__(self,i,initx,weights,params,bias,drive,intn,inpw):
        self.i = i
        self.w = weights # n x n
        self.p = params
        self.x = initx   # 1 x n
        self.y = 0*initx
        self.b = bias    # 1 x n
        self.d = drive #slow input susceptibility
        self.intn = intn #interneuron
        self.inpw = inpw #fast input susceptibility
    
    def nullclinex(self,x,i,dc_in):
        y = (self.b[i] + dc_in*self.d[i] - x) / (self.p['a']*sig(-self.p['c']*(x-self.p['x0']))) 
        return y
    
    def nullcliney(self,x,i,dc_in):
        y = self.p['b']*hinge(x) / self.p['gam']
        return y
    
    def stepx(self): #intra module dynamics
        dx = 0*self.x
        dy = 0*self.x
        for j in range(len(self.x)):
            dx[j] = -self.x[j] + sum(self.w[j,:]*hinge(self.x)) - self.p['a']*sig(-self.p['c']*(self.x[j]-self.p['x0']))*self.y[j] + self.b[j]
        
        dy = -self.p['gam']*self.y + self.p['b']*hinge(self.x)
        
        self.x = self.x + dx*self.p['dt']
        self.y = self.y + dy*self.p['dt']
        
    def fb_int(self,currx,adj): #inter-module dynamics
        #interneuron (assumes all have same bias parameter)
        dx = sum(adj[self.i,:]*hinge(currx))
        
        self.x[self.intn] = self.x[self.intn] + dx*self.p['dt']
        
    def fb_ext(self,z,dc_in): #input from outside CPG
    
        self.x = self.x + self.inpw*z*self.p['dt'] + self.d*dc_in*self.p['dt']


