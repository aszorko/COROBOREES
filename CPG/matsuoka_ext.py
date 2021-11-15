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
        self.x = self.x + self.inpw*z + self.d*dc_in*self.p['dt']
            

"""
class Robot:
    #general class, can be used for several oscillator types
    def __init__(self,initx,innerweights,params,bias,drive,adj,intn,inpw,inpc):
        self.param = params
        self.adj = adj  
        self.cons = [Controller(i,initx[i],innerweights,params,bias,drive,intn,inpw) for i in range(len(initx))]
        self.intn = intn
        self.inpc = inpc
    
    def step(self,z):
        #z=current external input
        currx = np.array([c.x[self.intn] for c in self.cons])

        for i in range(len(self.cons)):
            self.cons[i].stepx()
            self.cons[i].fb_int(currx,self.adj)
            self.cons[i].fb_ext(z*self.inpc[i])

        #self.stepglob(z)
        
        return np.array([c.x for c in self.cons])
        
class Sim:
    def __init__(self,rob,z):
        self.rob = rob
        self.tt = len(z)
        self.input = z
        self.allx = [np.zeros([len(self.rob.cons[i].x),self.tt]) for i in range(len(rob.cons))]
    
    #iterate robot and gather data
    def run(self):    
        for t in range(self.tt):
            newx = self.rob.step(self.input[t])
            for i in range(len(newx)):
                self.allx[i][:,t] = newx[i]
                
    def plot(self):
       fig= plt.figure()
       t = self.rob.param['dt']*np.arange(len(self.input))
       axs1 = fig.add_axes([0.1,0.1,0.45,0.4])
       axs2 = fig.add_axes([0.1,0.6,0.45,0.4])
       axs3 = fig.add_axes([0.6,0.1,0.3,0.4])
       axs4 = fig.add_axes([0.6,0.6,0.3,0.4])
       for i in range(len(self.rob.cons)):
          axs1.plot(t,hinge(self.allx[i][1,:])-hinge(self.allx[i][0,:]))
          axs2.plot(self.allx[i][2,:])
       #axs3.plot(self.allx[0][0,:],self.allx[0][1,:])
       x = np.arange(-2,2,0.01)
       axs3.plot(x,self.rob.cons[0].nullclinex(x,0))
       axs3.plot(x,self.rob.cons[0].nullcliney(x,0))
       axs3.set_ylim(bottom=-1,top=5)
       axs4.plot(x,self.rob.cons[0].nullclinex(x,2))
       axs4.plot(x,self.rob.cons[0].nullcliney(x,2))
       axs4.set_ylim(bottom=-1,top=5)
       axs2.xaxis.set_ticks([])
       #axs3.xaxis.set_ticks([])
       axs3.yaxis.set_ticks([])       
       axs3.xaxis.set_ticks([])       
       axs4.yaxis.set_ticks([])       
       axs4.xaxis.set_ticks([])       
       plt.show()


    def fft(self,start):
        n_i = len(self.rob.cons)
        n_j = len(self.rob.cons[0].x)
        peaks = np.zeros([n_i,n_j])
        tstart = round(self.tt*start)
        for i in range(n_i):
            for j in range(n_j):
                x = self.allx[i][j,tstart:]
                out = np.fft.rfft(x - np.mean(x))
                peaks[i,j] = np.argmax(np.abs(out))
                
        return peaks
        
    def autocorr(self,start,mindelay):
        n_i = len(self.rob.cons)
        n_j = len(self.rob.cons[0].x)
        peaks = np.zeros([n_i,n_j])
        heights = np.zeros([n_i,n_j])
        tstart = round(self.tt*start)
        for i in range(n_i):
            for j in range(n_j):
                x = self.allx[i][j,tstart:]
                out = np.correlate(x - np.mean(x),x - np.mean(x),mode='full')
                peaks[i,j] = mindelay + np.argmax(out[len(x)+mindelay:]) 
                heights[i,j] = np.max(out[len(x)+mindelay:])
        return peaks, heights

    #flexor-extensor combined period
    def autocorr2(self,start,mindelay):
        n_i = len(self.rob.cons)
        peaks = np.zeros([n_i,1])
        heights = np.zeros([n_i,1])
        tstart = round(self.tt*start)
        for i in range(n_i):
            x = hinge(self.allx[i][1,tstart:])-hinge(self.allx[i][0,tstart:])
            out = np.correlate(x - np.mean(x),x - np.mean(x),mode='full')
            peaks[i] = mindelay + np.argmax(out[len(x)+mindelay:])
            if peaks[i] == mindelay:
                peaks[i] = 0
            heights[i] = np.max(out[len(x)+mindelay:])              
        return peaks, heights

    def getamps(self,start):
        amps = np.zeros([len(self.rob.cons),1])
        tstart = round(self.tt*start)
        for i in range(len(self.rob.cons)):
            x1 = hinge(self.allx[i][1,tstart:])
            x2 = hinge(self.allx[i][0,tstart:])
            pks1, _ = signal.find_peaks(x1)
            pks2, _ = signal.find_peaks(x2)
            #print(amps1)
            #print(amps2)
            if len(pks1) > 0 and len (pks2) > 0:
                amps[i] = np.mean(x1[pks1]) + np.mean(x2[pks2])
            else:
                amps[i] = 0
        return amps

    def period_amp(self,corrstart=0.5,mindelay=1000):
        peaks,heights = self.autocorr2(corrstart,mindelay)
        amps = self.getamps(corrstart)
        return self.rob.param['dt']*peaks,heights,amps


"""



