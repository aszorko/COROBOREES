# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 01:17:06 2022

Time series figure
Note: needs to be run inside virtual environment

@author: alexansz
"""

import UnityInterfaceBrain
import numpy as np
import matsuoka_quad
import matsuoka_brain
from mlagents_envs.environment import UnityEnvironment
import matplotlib as mpl
import roborun
from matplotlib import pyplot as plt
from scipy import signal


def convolve(sig,period,dt):
    wt = np.arange(-0.5,0.5,dt)
    gamma = 0.5
    wavelet = (np.cos(2*np.pi*wt/period) - np.exp(-0.5/gamma**2))*np.exp(-0.5*(wt*gamma/period)**2)
    out = signal.convolve(sig,wavelet,mode='same',method='direct')/sum(wavelet**2)
    
    wind = 2
    gamma2 = 2
    y = np.exp(-0.5*gamma2**2*np.arange(-wind,wind,dt)**2)
    
    return np.convolve(out**2, y, 'same')
    #return np.convolve(out**2,np.ones(round(2*period/dt)),mode='same')

if __name__ == "__main__":
    mpl.style.use('ggplot')
    mpl.style.use('seaborn-colorblind')
    p = [1, 10, 1, 2, 8, 5, 5, 3, 10, 10, 5, 9, 8, 8, 9, 6, 4, 9, 1, 2, 8, 8, 5, 9, 10, 7, 2, 10, 2, 1, 1, 2]
    b = [2, 5, 10, 5, 9, 8, 3, 9, 1, 6, 6, 1, 1, 2, 1, 6, 8, 8, 7, 8, 5, 5, 8, 3, 1, 7, 9, 9, 8, 9, 9, 8, 6, 7, 9, 5, 3, 7, 10, 3, 2, 7, 10, 1, 6, 9, 6, 6, 8, 6, 9, 9, 9, 8, 5, 6, 4, 5, 5, 5, 6, 1]
    baseperiod_orig = 8.16 #cpg units
    
    t_arr = [1.132,1.390] #135,110 bpm
    
    

    
    n_brain = 6
    m = 4
    n_cpg = 23
    
    dt = 0.15
    dt_unity = 0.1
    stepsperframe = 13 #12
    t_length = 30 #seconds
    nframes = int(t_length / dt_unity)
    stimstart = 5 #seconds
    stimend = 25 #seconds
    
    amp = 1.2 #S0.7

    # unity limb order = RF,LF,RH,LH
    # matsuoka_quad limb order = LH,RH,LF,RF
    limbamp = -np.array([-1, 1,-1, 1])
    legamp = 0.5*p[-4]/10
    
    baseperiod = baseperiod_orig * dt_unity/stepsperframe/dt #convert to seconds
    
    t1 = np.arange(0,nframes,1)*dt_unity #unity time
    t0 = np.arange(0,nframes*stepsperframe,1)*dt_unity/stepsperframe #cpg time
    
    tcon = stepsperframe*dt/dt_unity
    
    cpg = matsuoka_quad.array2param(p[:n_cpg])
    body_inds = p[n_cpg:]
    brain,outw,decay,outbias = matsuoka_brain.array2brain(n_brain,m,b)
    
    bodytype = 'shortquad'
    Path = UnityInterfaceBrain.getpath('Windows',bodytype)
    env = UnityEnvironment(file_name=Path, seed = 4, side_channels = [], worker_id=0, no_graphics=True, additional_args=['-nolog'])
    
    figlabels = ['A','B']
    textkw = {'fontdict':{'fontsize':16}}

    tt = stepsperframe*nframes

    z1 = 100*amp*roborun.periodinput(t_arr[0]*tcon*baseperiod,round(tt*stimstart/t_length),round(tt/2),round(tt/2),dt)
    z2 = 100*amp*roborun.periodinput(t_arr[1]*tcon*baseperiod,0,round(tt*(stimend-t_length/2)/t_length),round(tt/2),dt)
    z = np.concatenate((z1,z2))
    z = roborun.rc_lpf(z,decay*dt)
    
    UnityInterfaceBrain.evaluate(env, cpg, body_inds, bodytype)    
    
    cpg.reset(111)
    brain.reset(222)

    (pardist, perpdist, height, tilt, output) = UnityInterfaceBrain.evaluate(env, cpg, body_inds, bodytype, dc_in = [0.5,0.5], tilt_in=[0.0], brain=brain, brain_in=z, outw=outw, outbias=outbias, getperiod=True, nframes=nframes, timeseries=True)
    
    env.close()

    alloutput = output
    
    outpks = []
    outsync1 = []
    outsync2 = []
    outsync0 = []
    inpks,_ = signal.find_peaks(z,height=0.01)
    for k in range(4):
        alloutput[k,:] = limbamp[k]*alloutput[k,:]
        times,_ = signal.find_peaks(np.diff(np.real(alloutput[k,:])),height=0.05,prominence=0.1)
        sync0 = convolve(np.diff(np.real(alloutput[k,:])),baseperiod,dt_unity/stepsperframe)
        sync1 = convolve(np.diff(np.real(alloutput[k,:])),t_arr[0]*baseperiod,dt_unity/stepsperframe)
        sync2 = convolve(np.diff(np.real(alloutput[k,:])),t_arr[1]*baseperiod,dt_unity/stepsperframe)
        
        outpks.append(times)
        outsync1.append(sync1)
        outsync2.append(sync2)
        outsync0.append(sync0)
    
    xlim = [2,t_length-2]
    fig = plt.figure(figsize=(3,5))
    gs = fig.add_gridspec(4, 1, left=0.2, right=0.9, bottom=0.1, top=0.9, wspace=0.05, hspace=0.15)
    
    axs = [fig.add_subplot(gs[i, 0]) for i in range(4)]
    
    transtime = 15
    plt.grid(True, axis = 'both')
    
    for k in range(4):
       axs[0].plot(t0[outpks[k]],k+0*t0[outpks[k]],'|')
    axs[0].plot(t0[inpks],4+0*t0[inpks],'|k')
    axs[0].set_xlim(xlim)
    axs[0].set_xticks([])
    axs[0].set_yticks([0,1,2,3,4])
    axs[0].set_yticklabels(['LH','RH','LF','RF','Stim']) 
    axs[0].set_ylim([-1,5.5])
    axs[0].annotate(text='', xy=(25,5.3), xytext=(15,5.3), arrowprops=dict(arrowstyle='<->', color='k', lw=1, shrinkA=0, shrinkB=0))
    axs[0].annotate(text='', xy=(15,5.3), xytext=(5,5.3), arrowprops=dict(arrowstyle='<->', color='k', lw=1, shrinkA=0, shrinkB=0))
    axs[0].text(9,5.7,'135 bpm')
    axs[0].text(19,5.7,'110 bpm')
    axs[1].plot(t0[1:],2*stepsperframe*UnityInterfaceBrain.sig(2*legamp*np.diff(np.real(alloutput))).squeeze().T)
    axs[1].set_xlim(xlim)
    axs[1].set_xticks([])    
    axs[1].set_ylabel('Leg output')
    y1 = np.mean(np.array(outsync1),0)
    y2 = np.mean(np.array(outsync2),0)
    y0 = np.mean(np.array(outsync0),0)
    line1, = axs[2].plot(t0[1:],y1/np.max(y1))
    line2, = axs[2].plot(t0[1:],y2/np.max(y2))
    line3, = axs[2].plot(t0[1:],y0/np.max(y0))
    axs[2].set_xlim(xlim)
    axs[2].set_xticklabels([])
    axs[2].tick_params(axis='x', color='white')    
    axs[2].set_ylabel('Sync')
    axs[2].legend([line1,line2,line3],['135 bpm','110 bpm','153 bpm'],fontsize=8,loc='lower left')
    axs[3].plot(t1,tilt)
    axs[3].set_xlim(xlim)
    axs[3].set_ylabel('Tilt')
    axs[3].set_xlabel('Time (s)')


    plt.show()
    fig.savefig('paper3_figures/transient.eps')


    fig = plt.figure(figsize=(3,5))
    gs = fig.add_gridspec(3, 1, left=0.2, right=0.9, bottom=0.1, top=0.9, wspace=0.05, hspace=0.15)
    
    axs = [fig.add_subplot(gs[i, 0]) for i in range(3)]    
    #w = 130
    gamma = 2
    wind = 2
    
    transtimes = [stimstart,t_length/2,stimend]
    maxlag = 3
    
    for i,x in enumerate([sync1,sync2,sync0]):  
       dx = np.diff(x)
       ddx = np.diff(dx)
       axs[0].plot(t0[1:],x)
       axs[1].plot(t0[1:-1],dx)
       axs[2].plot(t0[2:-1],ddx)
       maxpeak = np.argmin(ddx[round(transtimes[i]/dt_unity*stepsperframe):round((transtimes[i]+maxlag)/dt_unity*stepsperframe)])
       print(maxpeak*dt_unity/stepsperframe)
    plt.show()