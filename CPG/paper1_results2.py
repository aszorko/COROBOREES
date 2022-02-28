# -*- coding: utf-8 -*-
"""
Generates results and figures for CPG/filter('brain') combinations in the article
"Rapid rhythmic entrainment in bio-inspired central pattern generators"
CPG arrays come from paper1_results1.py, while the corresponding brains come from the text data files
Periods are determined by running matsuoka_quad.py with the CPG array

@author: alexansz
"""   

import matsuoka_brain
import matsuoka_quad
import roborun
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def ampfigure(fig,numcpgs):
    for i,axis in enumerate(fig.axes):
       if i==0 or i==2:
           axis.set_title('Filter')
       if i==1 or i==3:
           axis.set_title('CPG')
       if i//4 == 0:
          axis.set_ylim([0,10])
       elif i//4 == 1:
          axis.set_ylim([0,2])
       else:
          axis.set_ylim([0,4])
       if i%4 == 1:
           axis.set_yticks([])
       if i%4 == 2:
           axis.set_yticks([])
       if i%4 == 3:
           axis.set_yticks([])
           axis.text(1,0.25,'CPG'+str(i//4),color='grey',weight='bold',fontsize=12,rotation=270,transform=axis.transAxes)
       if i==8:
           txt = axis.set_ylabel('Output amplitude',fontsize=16)
           txt.set_y(1)
       if i//4<(numcpgs-1):
           axis.set_xticks([])
       else:
           if i%4 == 0:
               txt = axis.set_xlabel(r'Input period$/T_{0.5}$',fontsize=16)
               txt.set_x(1.2)
           if i%4 == 2:
               txt = axis.set_xlabel('Input amplitude',fontsize=16)
               txt.set_x(1.2)
    
    
    
def periodfigure(fig,numcpgs):
    for i,axis in enumerate(fig.axes):
       if i==0 or i==2:
           axis.set_title('Filter')
       if i==1 or i==3:
           axis.set_title('CPG')
       if i%4 == 1:
           axis.set_yticks([])
       if i%4 == 2:
           axis.set_ylim([0,2])
       if i%4 == 3:
           axis.set_ylim([0,2])
           axis.set_yticks([])
           axis.text(1,0.25,'CPG'+str(i//4),color='grey',weight='bold',fontsize=12,rotation=270,transform=axis.transAxes)
       if i==8:
           txt = axis.set_ylabel(r'Output period $/ T_{0.5}$',fontsize=16)
           txt.set_y(1)
       if i%4 < 2:
           axis.set_xlim([0.5,2])
           axis.set_ylim([0,5])
       if i//4<(numcpgs-1):
           axis.set_xticks([])
       else:
           if i%4 == 0:
               txt = axis.set_xlabel(r'Input period$/T_{0.5}$',fontsize=16)
               txt.set_x(1.2)
           if i%4 == 2:
               txt = axis.set_xlabel('Input amplitude',fontsize=16)
               txt.set_x(1.2)
               
def multifigure(num,xarr,yarr,rows,cols,dotted=[]):
    fig = plt.figure(num)
    gs = fig.add_gridspec(rows,cols, hspace=0.25,wspace=0.25)
    axs = gs.subplots().flat
    for i in range(len(yarr)):
       axs[i].plot(xarr[i],yarr[i])
       if len(dotted)>0:
           if dotted[i]==0:
              axs[i].plot(xarr[i],2*xarr[i],'--',color='grey')
              axs[i].plot(xarr[i],0.5*xarr[i],'--',color='grey')
              axs[i].plot(xarr[i],xarr[i],'--k')
           else:
              axs[i].plot(xarr[i],0*xarr[i]+dotted[i],'--k')
              axs[i].plot(xarr[i],0*xarr[i]+0.5*dotted[i],'--',color='grey')
              axs[i].plot(xarr[i],0*xarr[i]+2*dotted[i],'--',color='grey')
       #at = AnchoredText('('+string.ascii_lowercase[i]+')',frameon=False,loc='upper left')
       #at.patch.set_boxstyle("square,pad=0")
       #axs[i].add_artist(at)
    return fig

def timeseries(newsim,times,periods,t,t0,dt,tt,t_bounds,skipevery=-1):
    limb = ['LH','LF','RH','RF']
    
    finalt = dt*tt
    
    tstart = int(tt*t_bounds[0]) 
    
    fig = plt.figure()
    maxval = np.max([np.max(-newsim.outx[i][tstart:]) for i in range(len(limb))])
    minval = np.min([np.min(-newsim.outx[i][tstart:]) for i in range(len(limb))])
    tickint = (maxval-minval) * 0.22
    totaltime = (tt-t_bounds[0])*dt*t0
    leftlim = tstart*dt*t0 - 0.1*totaltime
    rightlim = finalt*t0 + 0.1*totaltime
    for i in range(len(limb)):
        plt.plot(np.arange(tstart,tt,1)*dt*t0,-newsim.outx[i][tstart:],color='C'+str(i),linewidth=0.7)
        plt.plot((times[i]+tstart*dt)*t0,0*times[i] + maxval + tickint*(i+2),'|',color='C'+str(i))
        plt.text(leftlim+0.02*totaltime, maxval + tickint*(i+1.7), limb[i])
    inputtimes = np.arange(t_bounds[1]*finalt,t_bounds[2]*finalt,t)*t0
    if skipevery>0:
        inputtimes2 = np.array([inputtimes[i] for i in range(len(inputtimes)) if i%skipevery!=0])
    else:
        inputtimes2 = inputtimes
    plt.plot(inputtimes2,0*inputtimes2 + maxval + tickint*6,'|k')    
    plt.text(leftlim+0.02*totaltime, maxval + tickint*(5.7), 'Stimulus')
    plt.xlabel('Time (s)',fontsize=12)
    txt = plt.ylabel('Output',fontsize=12)
    txt.set_y(0.2)
    plt.xlim([leftlim,rightlim])
    ticks,labels = plt.yticks()
    plt.yticks([x for x in ticks if x<maxval])    
    
    return fig 

if __name__ == "__main__":
    
    do_eval     = False # evaluates all CPG/brain combos (x 'numiters') and prints median scores 
    num_iters   = 5
    skip_every  = 4 #skip every nth pulse for full evaluation. set to -1 to use isochronous
    
    do_Fig1and2 = False # warning: takes a long time. runs the first 4 CPG/brain combos for various periods+amplitudes
    do_Fig3and4 = True  # gets time series for the CPG/brain combo with index 'brainindex'
    brain_index = 4
    
    
    n = 4 # number of neurons. currently can be 2 or 4
    m = 4 # number of limbs in CPG. currently only 4

    t0 = 0.01 # time constant
        
    bodyarray = []
    brainarray = []
    brain2array = []
    periods = []

    
    #run 5, brain 0 (overall max)
    bodyarray.append([4, 2, 2, 10, 3, 5, 3, 10, 8, 10, 10, 3, 2, 5, 9, 5, 2, 10, 9, 7, 1, 5, 8])
    brainarray.append([1, 8, 10, 6, 5, 8, 9, 1, 6, 1, 9, 9, 3, 6, 4, 0, 10, 3, 9, 9, 5, 10, 10, 8, 10, 2, 2, 2, 6, 4, 10, 8, 8, 10])
    brain2array.append([7, 7, 3, 9, 7, 3, 5, 8, 4, 10, 6, 9, 4, 4])
    periods.append(180.0)

    #run 5, brain 1 (F1 weighted max)
    bodyarray.append([4, 1, 4, 4, 9, 4, 2, 8, 8, 4, 8, 1, 1, 7, 2, 5, 10, 9, 6, 3, 3, 5, 4])
    brainarray.append([6, 0, 2, 9, 10, 10, 1, 4, 7, 3, 0, 4, 8, 1, 8, 10, 8, 10, 8, 10, 9, 4, 6, 2, 10, 10, 6, 6, 9, 1, 6, 5, 7, 10])
    brain2array.append([6, 1, 1, 1, 10, 9, 9, 4, 9, 10, 3, 5, 6, 2])
    periods.append(135.0)
    
    #run 5, brain 2 (F2 weighted max)
    bodyarray.append([4, 1, 4, 4, 10, 9, 3, 10, 3, 5, 10, 4, 3, 9, 5, 4, 10, 9, 9, 7, 10, 2, 8])
    brainarray.append([9, 7, 6, 3, 9, 10, 2, 7, 3, 7, 7, 4, 6, 4, 0, 5, 8, 10, 6, 5, 10, 9, 2, 3, 9, 6, 3, 7, 3, 2, 1, 5, 8, 0])
    brain2array.append([3, 10, 2, 10, 2, 2, 3, 8, 10, 5, 7, 0, 4, 0])
    periods.append(47.0)
    
    #run 5, brain 3 (F3 weighted max)
    bodyarray.append([4, 1, 4, 4, 10, 4, 2, 8, 1, 10, 4, 1, 1, 7, 4, 3, 9, 2, 6, 5, 1, 5, 4])
    brainarray.append([1, 8, 10, 1, 10, 6, 2, 3, 0, 2, 2, 0, 1, 9, 9, 5, 2, 9, 2, 8, 10, 9, 3, 1, 9, 7, 9, 3, 7, 9, 1, 5, 10, 8])
    brain2array.append([1, 7, 2, 8, 4, 0, 0, 10, 10, 9, 1, 1, 3, 4])
    periods.append(115.0) 
    
    #run 5, brain 4 (complex max)
    bodyarray.append([4, 2, 2, 10, 3, 5, 3, 10, 8, 10, 10, 3, 2, 5, 9, 5, 2, 10, 9, 7, 1, 5, 8])
    brainarray.append([8, 8, 0, 4, 8, 9, 0, 1, 7, 8, 0, 6, 7, 7, 0, 6, 8, 8, 8, 9, 6, 10, 4, 6, 9, 6, 9, 9, 2, 10, 3, 1, 10, 9])
    brain2array.append([9, 5, 10, 0, 6, 4, 1, 4, 10, 5, 7, 1, 0, 6])
    periods.append(180.0)
    
    numcpgs = 4

    plotall = False #plot every iteration (LOTS of plots)

    dc_in = 0.5
    relperiod = np.arange(0.5,2.1,0.1)  # period range for fig 1
    constamp = 1                        # amplitude for fig 1
    ratio = 0.8                         # period ratio for fig 2
    ratio2 = 0.8                        # period ratio for fig 3,4
    amp_arr = np.arange(0,1.1,0.1)      # amplitude range for fig 2
    constamp2 = 0.7                     # amplitude for fig 3&4
    
    
    if do_eval:
        matsuoka_brain.finaleval(n,bodyarray,brainarray,periods,num_iters,skipevery=skip_every)
    
    
    ###Figures 1 and 2. Uses the first four CPG/brain combinations
    
    
    period_data = []
    period_x = []
    amp_data = []
    amp_x = []
    dotted = []
    
    if not do_Fig1and2:
       numcpgs = 0

    
    for i in range(numcpgs):
       tt = 80000
       body = matsuoka_quad.array2param(bodyarray[i])
       if n==2:
          ind = brain2array[i]
       else:
          ind = brainarray[i]
          
       t_arr = relperiod*periods[i]

       brain,outw,decay,outbias = matsuoka_brain.array2brain(n,m,ind)
        
       print('cpg',i)
       allperiods,allamp,allbrainperiods,allbrainamp = roborun.stepdrive_ac(body,brain,outw,outbias,tt,[0,1],t_arr,[1],dc_in,plot=plotall,brainstats=True)
       period_data.append(np.median(allbrainperiods/periods[i],axis=1))
       period_data.append(np.median(allperiods/periods[i],axis=1))
       amp_data.append(np.mean(allbrainamp,axis=1))
       amp_data.append(np.mean(allamp,axis=1))
       for i in range(2):
           period_x.append(relperiod)
           amp_x.append(relperiod)
           dotted.append(0)
           
       
       tt = 80000
       
       t = periods[i]*ratio
       
       allperiods,allamp,allbrainperiods,allbrainamp = roborun.stepdrive_ac(body,brain,outw,outbias,tt,[0,1],[t],amp_arr,dc_in,decay,plot=plotall,brainstats=True)
       period_data.append(np.median(allbrainperiods/periods[i],axis=1))
       period_data.append(np.median(allperiods/periods[i],axis=1))
       amp_data.append(np.mean(allbrainamp,axis=1))
       amp_data.append(np.mean(allamp,axis=1))
       for i in range(2):
           period_x.append(amp_arr)
           amp_x.append(amp_arr)
           dotted.append(ratio)
    
    if do_Fig1and2:
       mpl.style.use('seaborn-colorblind')
       fig1 = multifigure(1,period_x,period_data,4,4,dotted=dotted) 
       periodfigure(fig1,numcpgs)
       
       
       fig2 = multifigure(2,amp_x,amp_data,4,4) 
       ampfigure(fig2,numcpgs)
    
    
    

    ###Figures 3 and 4
    
    if do_Fig3and4:
        mpl.style.use('ggplot')
        mpl.style.use('seaborn-colorblind')
        
        amp = constamp2
        tt = 160000
        i = brain_index
        
        body = matsuoka_quad.array2param(bodyarray[i])
        ind = brainarray[i]
        t = periods[i]*ratio2
           
    
        brain,outw,decay,outbias = matsuoka_brain.array2brain(n,m,ind)
    
        ###transient dynamics
        t_bounds = [0.2,0.4,0.8]
        t0 = 0.01
        
        newsim,times,outperiods,_,_ = roborun.periodvstime(body,brain,outw,outbias,tt,t_bounds,t,amp,dc_in,plot=False)
        fig3 = timeseries(newsim,times,outperiods,t,t0,body.param['dt'],tt,t_bounds)
        fig3.axes[0].set_aspect(10.0)
        fig3.axes[0].text(-0.12,0.95,'(a)',fontsize=12,transform=fig3.axes[0].transAxes)
        
        newsim2,times2,outperiods2,_,_ = roborun.periodvstime(body,brain,outw,outbias,tt,t_bounds,t,amp,dc_in,plot=False,skipevery=4)
        fig4 = timeseries(newsim2,times2,outperiods2,t,t0,body.param['dt'],tt,t_bounds,skipevery=4)
        fig4.axes[0].set_aspect(10.0)
        fig4.axes[0].text(-0.12,0.95,'(b)',fontsize=12,transform=fig4.axes[0].transAxes)

    

