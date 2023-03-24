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
from MathUtils import hinge


def periodinput(zperiod,zstart,zend,tt,dt,skipevery=-1,sdev=0,asym=None,seed=None):
    #asym is a tuple (int a, float b): every a'th beat is moved by a fraction b where |b|<1
    rng = np.random.default_rng(seed)
    z = np.zeros([tt,1])
    period = round(zperiod/dt)
    inds = np.array(range(zstart,zend,period))
    if asym is not None:
        inds[::asym[0]] = inds[::asym[0]] + np.round(asym[1]*period)
    noise = np.round(rng.normal(0,sdev*period,len(inds)))
    inds = inds + noise
    inds = np.delete(inds,inds<0)
    inds = np.delete(inds,inds>=tt)
    z[inds.astype(int)] = 1
    if skipevery>0:
        z[inds[::skipevery].astype(int)] = 0        
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

    #randomise state again
    def reset(self, seed=None):
        rng = np.random.default_rng(seed)
        k = len(self.cons[0].x)
        for con in self.cons:
            initx = -rng.random(k)
            con.x = initx
            con.y = 0*initx
            
    def step(self,z,dc_in,dt):
        #z=current external input
        currx = np.array([c.x[self.intn] for c in self.cons])

        for i in range(len(self.cons)):
            self.cons[i].stepx(dt)
            self.cons[i].fb_int(currx,self.adj,dt)
            if len(z)>1:                
               self.cons[i].fb_ext(z[i]*self.inpc[i],dc_in,dt)
            else:
               self.cons[i].fb_ext(z[0]*self.inpc[i],dc_in,dt)

        #self.stepglob(z)
        
        return np.array([c.x for c in self.cons])
        
class Sim:
    def __init__(self,rob,z,dc_in,dt = 0.04, brain = None, outw = None, outbias = None):
        self.rob = rob
        self.brain = brain
        self.outw = outw
        self.outbias = outbias
        self.tt = len(z)
        self.dt = dt
        self.input = z
        self.dc_in = dc_in #can be single number or same length as z
        self.allx = [np.zeros([len(self.rob.cons[i].x),self.tt]) for i in range(len(rob.cons))]
        if brain == None:
            self.brainx = []
        else:
            self.brainx = np.zeros([len(self.rob.cons),self.tt])
        self.outx = [np.zeros([1,self.tt-1]) for i in range(len(rob.cons))]
    
    #iterate robot and gather data
    def run(self, usehinge=True):
        zz = np.zeros([len(self.rob.cons),len(self.rob.cons[0].x)])
        for t in range(self.tt):
            if self.brain == None:
                zz = [self.input[t]]
            else:
                brainx = self.brain.step([self.input[t]],0,self.dt).squeeze()
                if usehinge:
                    brainout = hinge(brainx + self.outbias)                    
                else:
                    brainout = brainx + self.outbias
                z = np.matmul(self.outw,brainout)
                self.brainx[:,t] = z #brainx
                for i in range(len(self.rob.cons)):
                    zz[i,self.rob.intn] = z[i]
            if len(self.dc_in)==1:
                newx = self.rob.step(zz,self.dc_in[0],self.dt)
            else:
                newx = self.rob.step(zz,self.dc_in[t],self.dt)               
            for i in range(len(newx)):
                self.allx[i][:,t] = newx[i]
                
        if usehinge: # for plotting: get h(flexor) - h(extensor)
            self.outx = [hinge(self.allx[i][1,:])-hinge(self.allx[i][0,:]) for i in range(len(self.rob.cons))]
        else:
            self.outx = [self.allx[i][0,:] for i in range(len(self.rob.cons))]



    def plot(self,tstart=0):
       fig= plt.figure()
       t = self.dt*np.arange(len(self.input))
       axs1 = fig.add_axes([0.1,0.1,0.45,0.25])
       axs2 = fig.add_axes([0.1,0.4,0.45,0.25])
       axs3 = fig.add_axes([0.1,0.7,0.45,0.25])
       axs4 = fig.add_axes([0.6,0.1,0.3,0.25])
       axs5 = fig.add_axes([0.6,0.4,0.3,0.25])
       axs6 = fig.add_axes([0.6,0.7,0.3,0.25])
       for i in range(len(self.rob.cons)):
          axs1.plot(t[tstart:],self.allx[i][0,tstart:])
          axs2.plot(t[tstart:],self.allx[i][1,tstart:])
          axs3.plot(t[tstart:],self.allx[i][2,tstart:])
       x = np.arange(-2,2,0.01)
       axs4.plot(x,self.rob.cons[0].nullclinex(x,0,self.dc_in[0]))
       axs4.plot(x,self.rob.cons[0].nullcliney(x,0,self.dc_in[0]))
       axs4.set_ylim(bottom=-1,top=10)
       axs5.plot(x,self.rob.cons[0].nullclinex(x,1,self.dc_in[0]))
       axs5.plot(x,self.rob.cons[0].nullcliney(x,1,self.dc_in[0]))
       axs5.set_ylim(bottom=-1,top=10)
       axs6.plot(x,self.rob.cons[0].nullclinex(x,2,self.dc_in[0]))
       axs6.plot(x,self.rob.cons[0].nullcliney(x,2,self.dc_in[0]))
       axs6.set_ylim(bottom=-1,top=10)
       axs2.xaxis.set_ticks([])
       axs3.xaxis.set_ticks([])       
       axs4.yaxis.set_ticks([])       
       axs5.yaxis.set_ticks([])       
       axs5.xaxis.set_ticks([])       
       axs6.yaxis.set_ticks([])       
       axs6.xaxis.set_ticks([])       
       plt.show()
       
    def plotbrain(self):
       plt.figure()
       t = self.dt*np.arange(len(self.input))
       for i in range(len(self.brain.cons)):
          plt.plot(t,self.brainx[i])   
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
    def autocorr2(self,start,mindelay,brain=False,maxdelay=-1):
        n_i = len(self.rob.cons)
        peaks = np.zeros([n_i,1])
        heights = np.zeros([n_i,1])
        tstart = round(self.tt*start)
        if maxdelay == -1:
            maxdelay = self.tt-tstart
        for i in range(n_i):
            if brain:
               x = self.brainx[i][tstart:]
            else:
               x = self.outx[i][tstart:]
            out = np.correlate(x - np.mean(x),x - np.mean(x),mode='full')
            peaks[i] = mindelay + np.argmax(out[(len(x)+mindelay):(len(x)+maxdelay)])
            if peaks[i] == mindelay:
                peaks[i] = 0
            heights[i] = np.max(out[len(x)+mindelay:])              
        return peaks, heights

    def getamps(self,start,brain=False,minwidth=0):
        amps = np.zeros([len(self.rob.cons),1])
        tstart = round(self.tt*start)
        for i in range(len(self.rob.cons)):
            if brain:
                x = self.brainx[i][tstart:]
            else:
                x = self.outx[i][tstart:]
            pks1, _ = signal.find_peaks(x,width=minwidth)
            pks2, _ = signal.find_peaks(-x,width=minwidth)
            if len(pks1) > 0 and len (pks2) > 0:
                amps[i] = np.mean(x[pks1]) - np.mean(x[pks2])
            else:
                amps[i] = 0
        return amps
    
    def allpeaks(self,start,brain=False):
        times = []
        periods = []
        tstart = round(self.tt*start)
        for i in range(len(self.rob.cons)):
            if brain:
                x = self.brainx[i][tstart:]
            else:
                x = self.outx[i][tstart:]
            pks1, _ = signal.find_peaks(x)
            pks2, _ = signal.find_peaks(-x)
            #print(amps1)
            #print(amps2)
            if len(pks1) > 0:
                times.append(self.dt*pks1[1:])
                periods.append(self.dt*np.diff(pks1))
            else:
                times.append([])
                periods.append([])
        return times,periods

    def allpeaks2(self,start,t,brain=False):
        times = []
        periods = []
        tstart = round(self.tt*start)
        for i in range(len(self.rob.cons)):
            if brain:
                x = self.brainx[i][tstart:]
            else:
                x = self.outx[i][tstart:]
            pks1, _ = signal.find_peaks(-x,width=t/8/self.dt)
            if len(pks1) > 0:
                times.append(self.dt*pks1)
                periods.append(np.mod(self.dt*(pks1-tstart)+t/2,t))
            else:
                times.append([])
                periods.append([])
        return times,periods
    
    #not quite working
    def bandpass(self,period):
        x = np.zeros([len(self.rob.cons),self.tt])
        sos = signal.butter(2, [1/(period+5), 1/(period-5)], 'bandpass', fs=1/self.dt, output='sos')
        for i in range(len(self.rob.cons)):
           filtered = signal.sosfilt(sos, self.outx[i])
           x[i,:] = filtered
        
        return x

    def period_amp(self,corrstart=0.5,mindelay=1000,brain=False,maxdelay=-1,minwidth=0):
        peaks,heights = self.autocorr2(corrstart,mindelay,brain,maxdelay)
        amps = self.getamps(corrstart,brain,minwidth)
        
        return self.dt*peaks,heights,amps
    
    #duty function, no differential
    def getduty(self,start=0.5):        
        tstart = round(self.tt*start)
        x = np.zeros([len(self.rob.cons),self.tt-tstart])
        x2 = np.zeros([len(self.rob.cons),self.tt-tstart])
        for i in range(len(self.rob.cons)):
            x[i,:] = self.allx[i][0,tstart:]        
            x2[i,:] = self.allx[i][1,tstart:]        
        return np.mean(np.sum(x<0,0)>1)/2 + np.mean(np.sum(x2<0,0)>1)/2 

    #duty function, uses differentials
    def getduty2(self,start=0.5):
        epsilon = 0.0001
        tstart = round(self.tt*start)
        x = np.zeros([len(self.rob.cons),self.tt-tstart-1])
        x2 = np.zeros([len(self.rob.cons),self.tt-tstart-1])
        for i in range(len(self.rob.cons)):
            x[i,:] = abs(np.diff(hinge(self.allx[i][0,tstart:])))       
            x2[i,:] = abs(np.diff(hinge(self.allx[i][1,tstart:])))        
        return np.mean(np.sum(x<epsilon,0)>1)/2 + np.mean(np.sum(x2<epsilon,0)>1)/2 

def periodvstime(rob,brain,outw,outbias,tt,t_bounds,t,amp,dc_in,dt=0.04, skipevery=-1, sdev=0, plot=True):
    z = amp*100*periodinput(int(t),int(tt*t_bounds[1]),int(tt*t_bounds[2]),tt,dt,skipevery=skipevery,sdev=sdev)
    z = rc_lpf(z,1/100)

    
    newsim = Sim(rob,z,[dc_in],dt=dt,brain=brain,outw=outw,outbias=outbias)
    newsim.run()
    
    times,periods = newsim.allpeaks2(t_bounds[0],t)
    braintimes,brainperiods = newsim.allpeaks2(t_bounds[0],t,brain=True)
    
    if plot:
        finalt = dt*tt
        
        tstart = int(tt*t_bounds[0])
    
        newsim.plot(tstart=tstart)
        newsim.plotbrain()
        print(newsim.brainx[:,-1])
    
        for i in range(len(rob.cons)):
            plt.plot(times[i]+finalt*t_bounds[0],2*(periods[i]/t-0.5),'x')
            
        plt.plot([finalt*t_bounds[1],finalt*t_bounds[1]],[-1,1],'k--')
        plt.plot([finalt*t_bounds[2],finalt*t_bounds[2]],[-1,1],'k--')

        plt.xlabel('Time')
        plt.ylabel(r'Phase of peak relative to input ($\pi$ rad)')
        plt.show()
    
        #xfilt = newsim.bandpass(t)
        #plt.figure()
        #plt.plot(np.arange(0,tt,1)*rob.param['dt'],xfilt.T) 

    return newsim,times,periods,braintimes,brainperiods
    

#run the robot for various tonic input levels specified in d_arr
def stepdrive(rob,z,d_arr,dt=0.04,plot=False):  
    allperiods = []
    allamp = []
    allcorr = []
    allduty = []
    
    for i in range(len(d_arr)):
        newrob = copy.deepcopy(rob)
        newsim = Sim(newrob,z,[d_arr[i]],dt=dt)
        newsim.run()
        periods,corr,amp = newsim.period_amp()
        dutymeasure = newsim.getduty2()
        if plot:
            print('drive =',d_arr[i])
            newsim.plot()
            print(periods)
            print(corr)
            print(amp)
            print(dutymeasure)
        allperiods.append(periods)
        allcorr.append(corr)
        allamp.append(amp)
        allduty.append(dutymeasure)
        del newrob
        del newsim
    
    if len(d_arr)==1:
        return None
    
    if plot:
        plt.figure
        plt.plot(d_arr,np.squeeze(allperiods))
        plt.show()
    
    #return evalrobot(np.squeeze(np.array(allperiods)),np.squeeze(np.array(allcorr)),np.squeeze(np.array(allamp)),np.array(allduty))
    return np.squeeze(np.array(allperiods)),np.squeeze(np.array(allcorr)),np.squeeze(np.array(allamp)),np.array(allduty)


#run the robot with various periodic drives
#will iterate through amp_arr (amplitude) if it has length >1
#otherwise iterate through t_arr (period)    
def stepdrive_ac(rob,brain,outw,outbias,tt,t_bounds,t_arr,amp_arr,dc_in,dt=0.04,decay=0.25,plot=False,brainstats=False,skipevery=-1,sdev=0):
    allperiods = []
    allamp = []
    allcorr = []
    allbrainamp = []
    allbrainperiods = []
    
    maxperiod = 2.9 #multiple of drive period, max delay for autocorrelation peak
    minwidth = 0 #multiple of drive period, min width for peak finding
    
    for i in range(max([len(t_arr),len(amp_arr)])):
        
        if len(amp_arr)>1:
           t = t_arr[0]
           amp = amp_arr[i]
        else:
           t = t_arr[i]
           amp = amp_arr[0]
        
        z = 100*amp*periodinput(t,int(tt*t_bounds[0]),int(tt*t_bounds[1]),tt,dt,skipevery=skipevery,sdev=0) 
        z = rc_lpf(z,decay*dt)

        newrob = copy.deepcopy(rob)
        newbrain = copy.deepcopy(brain)
        
        newsim = Sim(newrob,z,[dc_in],dt=dt,brain=newbrain,outw=outw,outbias=outbias)
        newsim.run()
        periods,corr,amp = newsim.period_amp(maxdelay=int(maxperiod*t/dt),minwidth=int(minwidth*t/dt))
        if plot:
            newsim.plot()
            newsim.plotbrain()
            print(periods)
            print(corr)
            print(amp)
        if brainstats:
            periodb,corrb,ampb = newsim.period_amp(brain=True)
            allbrainperiods.append(periodb)
            allbrainamp.append(ampb)
        allperiods.append(periods)
        allcorr.append(corr)
        allamp.append(amp)
        del newrob
        del newbrain
        del newsim
    
    allperiods = np.squeeze(np.array(allperiods))
    allamp = np.squeeze(np.array(allamp))
    allbrainperiods = np.squeeze(np.array(allbrainperiods))
    allbrainamp = np.squeeze(np.array(allbrainamp))
   
    return allperiods,allamp,allbrainperiods,allbrainamp


#fitness function for CPG
def evalrobot(allperiods,allcorr,allamp,allduty):
    minpeak = 0.1
    maxpeak = 10
    maxjump = 0.15 #0.2
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
    diffperiod = diffperiod*(abs(diffperiod)/sumperiod < maxjump)

    tot1 = abs(sum(np.squeeze(diffperiod*corrpeak_exists.T)))
    tot2 = np.mean(cv_thresh / (cv_thresh + ampcv.T + stdcv.T))
    tot3 = np.mean(allduty*(meanperiod>0))
    
    #return abs(sum(diffperiod*corrpeak_exists/(1+ampcv)))
    return (tot1,tot2,tot3)

#fitness function for brain
def evalbrain(allperiods,allamp,t_arr,zero_std):
    minpeak = 0.1
    maxpeak = 10
    epsilon = 0.1

    goodamp = [np.mean((allamp[i] < maxpeak)*(allamp[i] > minpeak)) for i in range(len(t_arr))]
    freqdiff = np.sqrt([np.mean((allperiods[i]-t_arr[i])**2) for i in range(len(t_arr))])
    
    score = goodamp / (1 + freqdiff + zero_std/epsilon)
    
    return tuple(np.squeeze(score)) 

#runs CPG and returns fitness
def runcpg(rob,z,d_arr,dt=0.04,plot=False):
    allperiods,allcorr,allamp,allduty = stepdrive(rob,z,d_arr,dt=dt,plot=plot)
    return evalrobot(allperiods,allcorr,allamp,allduty)

# runs CPG+brain system first with no input, then with various periods. returns fitness
def runbrain(rob,brain,outw,outbias,tt,t_bounds,t_arr,dc_in,decay,dt=0.04,plot=False,skipevery=-1,sdev=0):
    #first run with no input
    z = np.zeros([tt+1,1])
    firstsim = Sim(rob,z[:,0],[dc_in],dt=dt,brain=brain,outw=outw,outbias=outbias)
    firstsim.run()
    zero_std = np.mean(np.std(firstsim.brainx[:,int(tt/2):],1))
    if plot:
       firstsim.plot()
       firstsim.plotbrain()
       print('no input: std=',zero_std)
    
    #pass to function that iterates through t_arr
    allperiods,allamp,allbrainperiods,allbrainamp = stepdrive_ac(rob,brain,outw,outbias,tt,t_bounds,t_arr,[1],dc_in,decay,dt=dt,plot=plot,skipevery=skipevery,sdev=sdev)
    return evalbrain(allperiods,allamp,t_arr,zero_std)
