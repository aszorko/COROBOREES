# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 01:17:06 2022

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
    gamma = 0.667 #0.5
    wavelet = (np.cos(2*np.pi*wt/period) - np.exp(-0.5/gamma**2))*np.exp(-0.5*(wt*gamma/period)**2)
    out = signal.convolve(sig,wavelet,mode='same',method='direct')/sum(wavelet**2)
    
    wind = 2
    gamma2 = 2
    y = np.exp(-0.5*gamma2**2*np.arange(-wind,wind,dt)**2)
    
    return np.convolve(out**2, y, 'same')


if __name__ == "__main__":
    mpl.style.use('ggplot')
    mpl.style.use('seaborn-colorblind')
    p = [1, 10, 1, 2, 8, 5, 5, 3, 10, 10, 5, 9, 8, 8, 9, 6, 4, 9, 1, 2, 8, 8, 5, 9, 10, 7, 2, 10, 2, 1, 1, 2]
    b = [2, 5, 10, 5, 9, 8, 3, 9, 1, 6, 6, 1, 1, 2, 1, 6, 8, 8, 7, 8, 5, 5, 8, 3, 1, 7, 9, 9, 8, 9, 9, 8, 6, 7, 9, 5, 3, 7, 10, 3, 2, 7, 10, 1, 6, 9, 6, 6, 8, 6, 9, 9, 9, 8, 5, 6, 4, 5, 5, 5, 6, 1]
    baseperiod = 8.16 #cpg units
    
    t_arr = [0.8,1.25]
    t_ratios = [1.5,1]
    legtext = ['Stimulus T x 3/2','Stimulus T']

    
    n_brain = 6
    m = 4
    n_cpg = 23
    
    dt = 0.16
    dt_unity = 0.1
    stepsperframe = 13 #12
    t_length = 22 #seconds
    nframes = int(t_length / dt_unity)
    stimstart = 6 #seconds
    stimend = 16 #seconds
    
    amp = 1 #S0.7

    # unity limb order = RF,LF,RH,LH
    # matsuoka_quad limb order = LH,RH,LF,RF
    limbamp = -np.array([-1, 1,-1, 1])
    legamp = 0.5*p[-4]/10
    
    baseperiod = baseperiod * dt_unity/stepsperframe/dt
    
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

    UnityInterfaceBrain.run_brain_array(n_brain,cpg,body_inds,baseperiod,bodytype,env,b,ratios=t_arr,dc=0.5,tilt=0,graphics=True,skipevery=-1,numiter=1,sdev=0,seed=None,combined=True)
    
    for j in range(len(t_arr)):

        (alldist,allperpdist,allheight,alltilt,allinput,alloutput) = UnityInterfaceBrain.run_with_input(env,cpg,body_inds,bodytype,baseperiod,brain,outw,decay,outbias,[t_arr[j]],amp,nframes,0.5,0,skipevery=-1,sdev=0,tstart=stimstart/t_length,tend=stimend/t_length,seed=123,timeseries=True)
        i = 0

        
        outpks = []
        outsync0 = []
        outsync1 = []
        inpks,_ = signal.find_peaks(allinput[i],height=0.01)
        for k in range(4):
            alloutput[i][k,:] = limbamp[k]*alloutput[i][k,:]
            times,_ = signal.find_peaks(np.diff(np.real(alloutput[i][k,:])),height=0.05,prominence=0.1)
            outpks.append(times)
            sync0 = convolve(np.diff(np.real(alloutput[i][k,:])),baseperiod,dt_unity/stepsperframe)
            sync1 = convolve(np.diff(np.real(alloutput[i][k,:])),t_ratios[j]*t_arr[j]*baseperiod,dt_unity/stepsperframe)            
            outsync0.append(sync0)
            outsync1.append(sync1)
            
        fig, axs = plt.subplots(5,1)
        for k in range(4):
           axs[0].plot(t0[outpks[k]],k+0*t0[outpks[k]],'|')
        axs[0].plot(t0[inpks],4+0*t0[inpks],'|k')
        axs[0].set_xlim([2,t_length-2])
        axs[0].set_xticks([])
        axs[0].set_yticks([0,1,2,3,4])
        #axs[0].set_yticklabels(['RH','LF','RF','LH','Stim'])    
        axs[0].set_yticklabels(['LH','RH','LF','RF','Stim'])    
        axs[0].set_ylim([-1,5])
        axs[1].plot(t0[1:],2*stepsperframe*UnityInterfaceBrain.sig(2*legamp*np.diff(np.real(alloutput[i]))).squeeze().T)
        axs[1].set_xlim([2,t_length-2])
        axs[1].set_xticks([]) 
        axs[1].set_yticks([0,0.2])         
        axs[1].set_ylabel('Leg output')
        y1 = np.max(np.array(outsync1),0)
        y0 = np.max(np.array(outsync0),0)
        line1, = axs[2].plot(t0[1:],y1/np.max(y1))
        line2, = axs[2].plot(t0[1:],y0/np.max(y0))
        axs[2].set_xlim([2,t_length-2])
        axs[2].set_xticklabels([])
        axs[2].tick_params(axis='x', color='white')    
        axs[2].set_ylabel('Sync')
        axs[2].legend([line1,line2],['Intrinsic T',legtext[j]],fontsize=8,loc='lower left')
        axs[3].plot(t1,allheight[i])
        axs[3].set_xlim([2,t_length-2])
        axs[3].set_xticks([])    
        axs[3].set_ylabel('Height')#,labelpad=20)
        axs[4].plot(t1,alltilt[i])
        axs[4].set_xlim([2,t_length-2])
        axs[4].set_ylabel('Tilt')#,labelpad=17)
        axs[4].set_xlabel('Time (s)')
        plt.gcf().text(0.01, 0.85,figlabels[j],**textkw)
        plt.show()
        fig.savefig('paper2_figures/timeseries_v3' + figlabels[j] + '.eps')
    
    env.close()