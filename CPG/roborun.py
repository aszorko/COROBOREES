# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 14:04:20 2021

General Robot class, Sim class and associated functions
Robot class requires:
 - cons: an array of Controller class objects for instantiation,
 - param: global params dict containing time constant 'dt'
 - adj: connection weights for interneurons as adjacency matrix 
 - intn: index of oscillator in each controller corresponding to the interneuron
 - inpc: array of coefficients (one per controller) for fast input z
 
Sim class provides input z to a robot for a number of time steps
and collects data from the robot over this time

@author: alexansz
"""
import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy import signal


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


def periodinput(zperiod,zstart,zend,tt,dt):
    z = np.zeros([tt,1])
    period = round(zperiod/dt)
    z[zstart:zend:period] = 1
    return z

def rc_lpf(z,f):
    #exponential impulse response, f=frequency*dt
    t = np.arange(len(z))
    w = f*np.exp(-f*t)
    return signal.convolve(z[:,0],w,mode='full')[0:len(t)]  


class Robot:
    #general class, can be used for several oscillator types
    def __init__(self,cons,param,adj,intn,inpc):
        self.cons = cons
        self.param = param
        self.adj = adj      #adjacency matrix
        self.intn = intn    #interneuron
        self.inpc = inpc    #each controller's input weight
    
    def step(self,z,dc_in):
        #z=current external input
        currx = np.array([c.x[self.intn] for c in self.cons])

        for i in range(len(self.cons)):
            self.cons[i].stepx()
            self.cons[i].fb_int(currx,self.adj)
            self.cons[i].fb_ext(z*self.inpc[i],dc_in)

        #self.stepglob(z)
        
        return np.array([c.x for c in self.cons])
        
class Sim:
    def __init__(self,rob,z,dc_in):
        self.rob = rob
        self.tt = len(z)
        self.input = z
        self.dc_in = dc_in #can be single number or same length as z
        self.allx = [np.zeros([len(self.rob.cons[i].x),self.tt]) for i in range(len(rob.cons))]
        self.outx = [np.zeros([1,self.tt]) for i in range(len(rob.cons))]
    
    #iterate robot and gather data
    def run(self, usehinge=True):    
        for t in range(self.tt):
            if len(self.dc_in)==1:
                newx = self.rob.step(self.input[t],self.dc_in[0])
            else:
                newx = self.rob.step(self.input[t],self.dc_in[t])               
            for i in range(len(newx)):
                self.allx[i][:,t] = newx[i]
                #if usehinge:
                #    self.outx[i][t] = hinge([self.allx[i][1,t]])-hinge([self.allx[i][0,t]])
                #else:
                #    self.outx[i][t] = self.allx[i][0,t]
                
        if usehinge: # for plotting: get h(flexor) - h(extensor)
            self.outx = [hinge(self.allx[i][1,:])-hinge(self.allx[i][0,:]) for i in range(len(self.rob.cons))]
        else:
            self.outx = [self.allx[i][0,:] for i in range(len(self.rob.cons))]

    def plot(self):
       fig= plt.figure()
       t = self.rob.param['dt']*np.arange(len(self.input))
       axs1 = fig.add_axes([0.1,0.1,0.45,0.4])
       axs2 = fig.add_axes([0.1,0.6,0.45,0.4])
       axs3 = fig.add_axes([0.6,0.1,0.3,0.4])
       axs4 = fig.add_axes([0.6,0.6,0.3,0.4])
       for i in range(len(self.rob.cons)):
          axs1.plot(t,self.outx[i])
          axs2.plot(t,self.allx[i][2,:])
       x = np.arange(-2,2,0.01)
       axs3.plot(x,self.rob.cons[0].nullclinex(x,0,self.dc_in[0]))
       axs3.plot(x,self.rob.cons[0].nullcliney(x,0,self.dc_in[0]))
       axs3.set_ylim(bottom=-1,top=5)
       axs4.plot(x,self.rob.cons[0].nullclinex(x,2,self.dc_in[0]))
       axs4.plot(x,self.rob.cons[0].nullcliney(x,2,self.dc_in[0]))
       axs4.set_ylim(bottom=-1,top=5)
       axs2.xaxis.set_ticks([])
       #axs3.xaxis.set_ticks([])
       axs3.yaxis.set_ticks([])       
       axs3.xaxis.set_ticks([])       
       axs4.yaxis.set_ticks([])       
       axs4.xaxis.set_ticks([])       
       plt.show()
    """
    def plot(self):
        for i in range(len(self.rob.cons)):
           #plt.plot(self.allx[i][0,:])
           plt.plot(self.allx[i][1,:]-self.allx[i][0,:])
           #plt.plot(self.allx[i][2,:])
        plt.show()
    """

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
    
    #all periods     
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
            x = self.outx[i][tstart:]
            out = np.correlate(x - np.mean(x),x - np.mean(x),mode='full')
            peaks[i] = mindelay + np.argmax(out[len(x)+mindelay:])
            #allpeaks, _ = signal.find_peaks(out[len(x)+mindelay:])
            #if len(allpeaks) == 0:
            #    peaks[i] = 0
            #else:
            #    peaks[i] = mindelay + min(allpeaks)
            if peaks[i] == mindelay:
                peaks[i] = 0
            heights[i] = np.max(out[len(x)+mindelay:])              
        return peaks, heights

    def getamps(self,start):
        amps = np.zeros([len(self.rob.cons),1])
        tstart = round(self.tt*start)
        for i in range(len(self.rob.cons)):
            x = self.outx[i][tstart:]
            pks1, _ = signal.find_peaks(x)
            pks2, _ = signal.find_peaks(-x)
            #print(amps1)
            #print(amps2)
            if len(pks1) > 0 and len (pks2) > 0:
                amps[i] = np.mean(x[pks1]) - np.mean(x[pks2])
            else:
                amps[i] = 0
        return amps

    def period_amp(self,corrstart=0.5,mindelay=1000):
        peaks,heights = self.autocorr2(corrstart,mindelay)
        amps = self.getamps(corrstart)
        return self.rob.param['dt']*peaks,heights,amps




def evalrobot(allperiods,allcorr,allamp):
    minpeak = 0.1
    maxpeak = 10
    minperiod = 1
    cv_thresh = 0.1
    allperiods = np.squeeze(allperiods)
    n = allperiods.shape[0]
    m = allperiods.shape[1]
    corrpeak_exists = np.zeros([n-1,1])
    ampcv = np.zeros([n-1,1])
    stdcv = np.zeros([n-1,1])
    for i in range(n-1):
        peak1_in_range = (allamp[i]>minpeak)*(allamp[i]<maxpeak)        
        peak2_in_range = (allamp[i+1]>minpeak)*(allamp[i+1]<maxpeak)        
        corrpeak_exists[i] = sum((allperiods[i]>minperiod)*peak1_in_range*(allperiods[i+1]>minperiod)*peak2_in_range)/m 
        ampcv[i] = np.std(allamp[i]) / (0.01 + np.mean(allamp[i])) + np.std(allamp[i+1]) / (0.01 + np.mean(allamp[i+1]))       
        stdcv[i] = np.std(allperiods[i]) / (0.01 + np.mean(allperiods[i])) + np.std(allperiods[i+1]) / (0.01 + np.mean(allperiods[i+1]))       
    meanperiod = np.mean(allperiods,1)
    meanperiod = meanperiod / max(1+meanperiod)
    diffperiod = np.diff(meanperiod)

    #remove very large jumps
    sumperiod = meanperiod[:-1] + meanperiod[1:] + 1
    diffperiod = diffperiod*(abs(diffperiod)/sumperiod < 0.2)

    tot1 = diffperiod*corrpeak_exists.T
    tot2 = cv_thresh * tot1 / (cv_thresh + ampcv.T + stdcv.T)
    
    #return abs(sum(diffperiod*corrpeak_exists/(1+ampcv)))
    return (abs(sum(np.squeeze(tot2))),)

def stepdrive(rob,z,d_arr,plot=False):
    #run the robot for various tonic input levels specified in d_arr
    allperiods = []
    allamp = []
    allcorr = []
    
    for i in range(len(d_arr)):
        newrob = copy.deepcopy(rob)
        newsim = Sim(newrob,z,[d_arr[i]])
        newsim.run()
        periods,corr,amp = newsim.period_amp()
        if plot:
            newsim.plot()
            print(periods)
            print(corr)
            print(amp)
        allperiods.append(periods)
        allcorr.append(corr)
        allamp.append(amp)
        del newrob
        del newsim
    
    if plot:
        plt.figure
        plt.plot(d_arr,np.squeeze(allperiods))
    
    return evalrobot(np.squeeze(np.array(allperiods)),np.squeeze(np.array(allcorr)),np.squeeze(np.array(allamp)))

