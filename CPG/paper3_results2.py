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
from matplotlib import pyplot as plt
from scipy import signal


def convolve(sig,period,dt):
    wt = np.arange(-0.5,0.5,dt)
    gamma = 0.5
    wavelet = (np.cos(2*np.pi*wt/period) - np.exp(-0.5/gamma**2))*np.exp(-0.5*(wt*gamma/period)**2)
    out = signal.convolve(sig,wavelet,mode='same',method='direct')
    
    return np.convolve(out**2,np.ones(round(2*period/dt)),mode='same')

if __name__ == "__main__":
    mpl.style.use('ggplot')
    mpl.style.use('seaborn-colorblind')
    p = [1, 10, 1, 2, 8, 5, 5, 3, 10, 10, 5, 9, 8, 8, 9, 6, 4, 9, 1, 2, 8, 8, 5, 9, 10, 7, 2, 10, 2, 1, 1, 2]
    b = [2, 5, 10, 5, 9, 8, 3, 9, 1, 6, 6, 1, 1, 2, 1, 6, 8, 8, 7, 8, 5, 5, 8, 3, 1, 7, 9, 9, 8, 9, 9, 8, 6, 7, 9, 5, 3, 7, 10, 3, 2, 7, 10, 1, 6, 9, 6, 6, 8, 6, 9, 9, 9, 8, 5, 6, 4, 5, 5, 5, 6, 1]
    baseperiod_orig = 8.16 #cpg units
    
    t_arr = [1.25]
    
    

    
    n_brain = 6
    m = 4
    n_cpg = 23
    
    dt = 0.15
    dt_unity = 0.1
    stepsperframe = 13 #12
    t_length = 20 #seconds
    nframes = int(t_length / dt_unity)
    stimstart = 6 #seconds
    stimend = 14 #seconds
    
    amp = 1 #S0.7

    # unity limb order = RF,LF,RH,LH
    # matsuoka_quad limb order = LH,RH,LF,RF
    limbamp = -np.array([-1, 1,-1, 1])
    legamp = 0.5*p[-4]/10
    
    baseperiod = baseperiod_orig * dt_unity/stepsperframe/dt #convert to seconds
    
    t1 = np.arange(0,nframes,1)*dt_unity #unity time
    t0 = np.arange(0,nframes*stepsperframe,1)*dt_unity/stepsperframe #cpg time
    
    cpg = matsuoka_quad.array2param(p[:n_cpg])
    body_inds = p[n_cpg:]
    brain,outw,decay,outbias = matsuoka_brain.array2brain(n_brain,m,b)
    
    bodytype = 'shortquad'
    Path = UnityInterfaceBrain.getpath('Windows',bodytype)
    env = UnityEnvironment(file_name=Path, seed = 4, side_channels = [], worker_id=0, no_graphics=True, additional_args=['-nolog'])
    
    figlabels = ['A','B']
    textkw = {'fontdict':{'fontsize':16}}

    
    for j in range(len(t_arr)):

        asym = None
        (alldist,allperpdist,allheight,alltilt,allinput,alloutput) = UnityInterfaceBrain.run_with_input(env,cpg,body_inds,bodytype,baseperiod,brain,outw,decay,outbias,[t_arr[j]],amp,nframes,0.5,0,skipevery=-1,sdev=0,tstart=stimstart/t_length,tend=stimend/t_length,seed=123,asym=asym,timeseries=True)
        i = 0

        
        outpks = []
        outsync = []
        inpks,_ = signal.find_peaks(allinput[i],height=0.01)
        for k in range(4):
            alloutput[i][k,:] = limbamp[k]*alloutput[i][k,:]
            times,_ = signal.find_peaks(np.diff(np.real(alloutput[i][k,:])),height=0.05,prominence=0.1)
            sync = convolve(np.diff(np.real(alloutput[i][k,:])),t_arr[j]*baseperiod,dt_unity/stepsperframe)
            outpks.append(times)
            outsync.append(sync)
        
        xlim = [2,t_length-2]
        fig = plt.figure(figsize=(3,4))
        gs = fig.add_gridspec(4, 1, left=0.2, right=0.9, bottom=0.1, top=0.9, wspace=0.05, hspace=0.15)
        
        axs = [fig.add_subplot(gs[i, 0]) for i in range(4)]
        
        for k in range(4):
           axs[0].plot(t0[outpks[k]],k+0*t0[outpks[k]],'|')
        axs[0].plot(t0[inpks],4+0*t0[inpks],'|k')
        axs[0].set_xlim(xlim)
        axs[0].set_xticks([])
        axs[0].set_yticks([0,1,2,3,4])
        axs[0].set_yticklabels(['LH','RH','LF','RF','Stim'])    
        axs[0].set_ylim([-1,5])
        axs[1].plot(t0[1:],2*stepsperframe*UnityInterfaceBrain.sig(2*legamp*np.diff(np.real(alloutput[i]))).squeeze().T)
        axs[1].set_xlim(xlim)
        axs[1].set_xticks([])    
        axs[1].set_ylabel('Leg output')
        axs[2].plot(t0[1:],np.mean(np.array(outsync),0))
        axs[2].set_xlim(xlim)
        axs[2].set_xticks([])    
        axs[2].set_ylim([1500,3000])
        axs[2].set_ylabel('Sync')
        axs[3].plot(t1,alltilt[i])
        axs[3].set_xlim(xlim)
        axs[3].set_ylabel('Tilt')
        axs[3].set_xlabel('Time (s)')
        plt.show()
        fig.savefig('paper3_figures/transient.eps')
    
    env.close()