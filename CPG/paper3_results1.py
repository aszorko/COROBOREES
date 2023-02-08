# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 12:41:05 2022

Figures 2 & 4 + statistics
Can be run from IDE

@author: alexansz
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

def getdata(filebase,run,cpg,prefix,measures):
    
    measurearr =  []

    file = filebase + '_run' + str(run) + '_brain' + str(cpg) + '_audio2_info.txt' 
    with open(file,'r') as f:
        while True:
            line = f.readline()
            if len(line)==0:
                break
            if 'base' in line:
                baseperiod = float(line.split(':')[1])
            elif 'score' in line:
                score = float(line.split(':')[1])

    
    for i,measure in enumerate(measures):
        file = filebase + '_run' + str(run) + '_brain' + str(cpg) + prefix + measure + '.txt'
        data = []
        with open(file,'r') as f:
            while True:
                line = f.readline()
                if len(line)==0:
                    break
                splitline = line.split(',')                    
                dataline = [float(x) for x in splitline]
                data.append(dataline)
        measurearr.append(data)         
    
    
    return measurearr,baseperiod,score


def plot_complexity(directory,filebase,run,cpg,bpms,plot):
    
    measurearr,baseperiod,score = getdata(directory+filebase,run,cpg,'_audio2_',['corr','period','height'])
    
    t0 = 0.5 #0.5 seconds in real time = 120bpm
    t1 = 0.0075 #single CPG timestep correction
    
    max_r = 2.5
    min_r = -1.5
    tdiff_epsilon = 0.1
    t_arr = np.array(measurearr[1])
    period = (t_arr[0,:]+t1)*0.5/t0

    r0 = np.log2((t_arr[0,:]+t1)/t0)
    pscore0 = 1/(1+abs(np.round(r0) - r0)/tdiff_epsilon)
    inds = np.logical_or(r0 < min_r,r0 > max_r)
    pscore0[inds] = 0
    
    r = np.array([np.log2((t_arr[i+1,:]+t1)*bpms[i]/60) for i in range(len(bpms))])
    pscore = 1/(1+abs(np.round(r) - r)/tdiff_epsilon)
    inds = np.logical_or(r < min_r,r > max_r)
    pscore[inds] = 0
    
    dc = np.arange(0,1.1,0.1)
    
    
    
    if plot:
        
        fig = plt.figure(figsize=(3,4))
        gs = fig.add_gridspec(3, 1, height_ratios=(2,0.2,1), left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.05, hspace=0.15)
        
        d_dc = dc[1]-dc[0]
        ax = fig.add_subplot(gs[2, 0])
        bpm = 60/period
        ax.plot(dc,bpm)
        if min(bpm)<120 and max(bpm)>120:
            ax.plot([dc[0],dc[-1]],[120,120],'--k')
        ax.set_xlabel('DC input')
        ax.set_ylabel('Intrinsic BPM')
        
        ax2 = fig.add_subplot(gs[1, 0], sharex=ax)    
        ax2.imshow(np.array([pscore0]),vmin=0,vmax=1,extent=(dc[0]-d_dc/2,dc[-1]+d_dc/2,0,1),aspect='auto')

        plt.setp(ax2.get_xticklabels(), visible=False)
        ax2.set_yticks([])
        ax2.set_ylabel('no sample',rotation='horizontal')
        ax2.grid(False)    
    
        ax3 = fig.add_subplot(gs[0, 0], sharex=ax)    
        ax3.imshow(pscore,vmin=0,vmax=1,extent=(dc[0]-d_dc/2,dc[-1]+d_dc/2,0,1),aspect='auto',origin='lower')

        plt.setp(ax3.get_xticklabels(), visible=False)
        ax3.set_yticks([])
        ax3.set_ylabel('low entropy   high entropy')
        ax3.grid(False)
        
     
    else:
        fig = None
     

    print(run,cpg,np.mean(pscore>(1+pscore0)/2))
    


    measurearr,_,_ = getdata(directory+filebase,run,cpg,'_iso120_',['period','height'])
    t_arr = np.array(measurearr[0])
    period = (t_arr[0,:]+t1)*0.5/t0

    r1 = np.log2((t_arr[0,:]+t1)/t0)
    pscore1 = 1/(1+abs(np.round(r1) - r1)/tdiff_epsilon)
    inds = np.logical_or(r1 < min_r,r1 > max_r)
    pscore1[inds] = 0

    

    return fig,np.mean(pscore>(1+pscore0)/2),np.mean(pscore0),np.mean(pscore1),np.mean(pscore,axis=1)


def plot_flexibility(directory,filebase,run,cpg,plot,title=True):
    
    t1 = 0.0075 #single CPG timestep correction
    max_r = 2.5
    min_r = -1.5
    tdiff_epsilon = 0.1 
    
    bpm = 60*np.arange(1,3.1,0.2)
    t0 = 60/bpm

    minbpm = bpm[0] - (bpm[1]-bpm[0])/2
    maxbpm = bpm[-1] + (bpm[1]-bpm[0])/2

    measurearr,baseperiod,score = getdata(directory+filebase,run,cpg,'_impulse_',['period','height'])
      
    y = np.arange(0,1.01,0.1)

    miny = y[0] - (y[1]-y[0])/2
    maxy = y[-1] + (y[1]-y[0])/2

    r0 = np.log2(baseperiod/t0)
    pscore0 = 1/(1+abs(np.round(r0) - r0)/tdiff_epsilon)
    
    r = np.log2(np.array([(np.array(measurearr[0][i])+t1)/t0[i] for i in range(len(t0))]).T)
    pscore = 1/(1+abs(np.round(r) - r)/tdiff_epsilon)
    inds = np.logical_or(r < min_r,r > max_r)
    pscore[inds] = 0
        
    amp1score = np.mean(pscore[-1])
    
    if plot:
        fig = plt.figure(figsize=(3,4))
        gs = fig.add_gridspec(2, 1, left=0.2, right=0.7, bottom=0.1, top=0.9, wspace=0.05, hspace=0.15)
    
        ax2 = fig.add_subplot(gs[1, 0])    
        im = ax2.imshow(np.concatenate([np.array([pscore0]),pscore],axis=0),vmin=0,vmax=1,extent=(minbpm,maxbpm,miny,maxy),aspect='auto',origin='lower')

        ax2.set_ylabel('Amplitude')
        ax2.grid(False)    
        ax2.set_xlabel('BPM')
        ax2.set_xticks([60,120,180])
        
        ax3 = fig.add_axes([0.75, 0.15, 0.05, 0.7])  
        fig.colorbar(im,cax=ax3,label='Score')
        
        textkw = {'fontdict':{'fontsize':16}}
        plt.gcf().text(0, 0.87,'A',**textkw)
        plt.gcf().text(0, 0.45,'B',**textkw)
        
    else:
        fig = None


    measurearr,baseperiod,score = getdata(directory+filebase,run,cpg,'_asym_',['period','height'])
    y = np.arange(0,0.91,0.05)

    miny = y[0] - (y[1]-y[0])/2
    maxy = y[-1] + (y[1]-y[0])/2

    
    r = np.log2(np.array([(np.array(measurearr[0][i])+t1)/t0[i] for i in range(len(t0))]).T)
    pscore = 1/(1+abs(np.round(r) - r)/tdiff_epsilon)
    inds = np.logical_or(r < min_r, r > max_r)
    pscore[inds] = 0

    if plot:
        ax3 = fig.add_subplot(gs[0, 0], sharex=ax2)    
        ax3.imshow(pscore,vmin=0,vmax=1,extent=(minbpm,maxbpm,miny,maxy),aspect='auto',origin='lower')
        plt.setp(ax3.get_xticklabels(), visible=False)
        ax3.set_ylabel('Asymmetry')
        ax3.grid(False) 
        if title:
           ax3.set_title('run ' + str(run) + ' cpg ' + str(cpg))
    

    
    return fig,amp1score 


if __name__ == '__main__':
    directory = r'./paper3_data/'
    
    #fig1: entrainment vs entropy
    #fig2: heatmaps for specified CPGs
        
    figs = [1,2]
    plotall = False #for all heatmaps
    
    sampledf = pd.read_csv('clipmeasures.txt')
    sampleorder = [0,1,2,3,4,5,6,7,8,9,11,10,12,13,14] #reordering of sampledf to reproduce order in Unity
    bpms = [119,116,117,120,120,120,121,114,120,127,112,116,130,116,115] #Unity order
    
    if 1 in figs:
        filebase = 'unityshort'
        files = os.listdir(directory)
        scores1 = []
        scores2 = []
        allscores = []
        allnullscores = []
        allisoscores = []
        
        cpgs = sum([[(run,cpg) for cpg in range(1,5)] for run in range(1,6)],[])
        for curr in cpgs:
            run = curr[0]
            cpg = curr[1]
            if not filebase + '_run' + str(run) + '_brain' + str(cpg) + '_audio2_info.txt' in files:
                continue
            _,medscore = plot_flexibility(directory,filebase,run,cpg,plotall)
            _,medscore2,null_score,isoscore,scorepersample  = plot_complexity(directory,filebase,run,cpg,bpms,plotall)
            scores1.append(medscore)
            scores2.append(medscore2)
            allscores.append(scorepersample)
            allnullscores.append(null_score)
            allisoscores.append(isoscore)
        
        x1 = 0.4
        x2 = 0.43
        inds = np.array(scores1)>0.85
        
        textkw = {'fontdict':{'fontsize':16}}
        fig1 = plt.figure()
        plt.plot(sampledf.PulseEnt[sampleorder],np.array(allscores)[inds,:].T,'.b')
        plt.plot([x1 for i in range(sum(inds))],np.array(allisoscores)[inds],'.k')
        plt.plot([x2 for i in range(sum(inds))],np.array(allnullscores)[inds],'.k')
        plt.plot([0.46,0.46],[0,1],'--k')
        plt.xlabel('            Entropy Autocor.',**textkw)
        plt.ylabel('Entrainment Score',**textkw)
        plt.xticks([x1,x2,0.5,0.6,0.7,0.8],['Iso.','None','0.5','0.6','0.7','0.8'])
            
        scorearr = np.array(allscores)[inds,:].flatten()
        entarr = np.array([sampledf.PulseEnt[sampleorder] for i in range(sum(inds))]).flatten()
        strarr = np.array([sampledf.MetStrength[sampleorder] for i in range(sum(inds))]).flatten()
        indarr = np.array([[i for j in range(len(sampledf))] for i in range(sum(inds))]).flatten()
        data = pd.DataFrame(scorearr,columns=["score"])
        data['entropy'] = entarr
        data['metstrength'] = strarr
        data['ind'] = indarr
        md1 = smf.mixedlm("score ~ entropy + metstrength", data, groups=data["ind"], re_formula="~entropy + metstrength")
        free = sm.regression.mixed_linear_model.MixedLMParams.from_components(np.ones(3), np.eye(3))
        mdf1 = md1.fit(free=free)
        print(mdf1.summary())
        print(sm.stats.diagnostic.kstest_normal(mdf1.resid))

        md2 = smf.mixedlm("score ~ entropy", data, groups=data["ind"], re_formula="~entropy")
        free = sm.regression.mixed_linear_model.MixedLMParams.from_components(np.ones(2), np.eye(2))
        mdf2 = md2.fit(free=free)
        print(mdf2.summary())

        xvec = np.array([0.5,0.8])
        yvec = xvec*mdf2.params['entropy'] + mdf2.params['Intercept']
        plt.plot(xvec,yvec,'-r') 
            

        
    if 2 in figs:  
        cpgs = [('unityshort',4,3)]
        for curr in cpgs:
            filebase = curr[0]
            run = curr[1]
            cpg = curr[2]

            fig2,medscore = plot_flexibility(directory,filebase,run,cpg,True,title=False)
            _ = plot_complexity(directory,filebase,run,cpg,bpms,True)
        
            