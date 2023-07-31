# -*- coding: utf-8 -*-
"""
Figures 2-4 + 5b + statistics
Can be run from IDE

@author: alexansz
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy.stats import mannwhitneyu

def getdata(filebase,prefix,measures,cut=None):
    
    measurearr =  []


    for i,measure in enumerate(measures):
        file = filebase + prefix + measure + '.txt'
        data = []
        with open(file,'r') as f:
            while True:
                line = f.readline()
                if len(line)==0:
                    break
                splitline = line.split(',')                 
                dataline = [float(x) for x in splitline if x != '']
                if len(data)>0 and len(dataline)<len(data[0]):
                   line= f.readline()
                   if len(line)==0:
                      break
                   splitline = line.split(',')                    
                   dataline = dataline+[float(x) for x in splitline]
                data.append(dataline)
        dataarr = np.array(data)
        inds = dataarr==-1
        dataarr[inds] = np.nan
        if cut is not None: #reshape to n x n
            #rows = dataarr.shape[0]
            parts = np.array([dataarr[cut[j]:cut[j+1],:] for j in range(len(cut)-1)])
            dataarr = np.concatenate(parts,axis=1)
        measurearr.append(dataarr)         
    
    
    return measurearr

def plot(measuredict,measures,newcuts,minorcuts,plotall=False):
    
    if plotall:   
        for measure in measures:        
            plt.figure()
            plt.imshow(np.array(measuredict[measure]))
            plt.title(measure)
            plt.colorbar()
            plt.xlabel('Learner')
            plt.ylabel('Teacher')
            plt.show()
      
    
    
    textkw = {'fontsize':16}
    
    fig11 = plt.figure(figsize=(4,4))
    plt.imshow(np.log10(measuredict["sync_free_diff"]),vmin=-3,vmax=0)
    plotrange = [-0.5,newcuts[-1]-0.5]
    if minorcuts is not None:
        for cut in minorcuts[1:-1]:
           plt.plot([cut-0.5,cut-0.5],plotrange,'lightgrey')
           plt.plot(plotrange,[cut-0.5,cut-0.5],'lightgrey')
    for cut in newcuts[1:-1]:
       plt.plot([cut-0.5,cut-0.5],plotrange,'white',linewidth=2)
       plt.plot(plotrange,[cut-0.5,cut-0.5],'white',linewidth=2)
    plt.ylim(plotrange.reverse())
    plt.xlabel('Learner')
    plt.xticks([])
    plt.yticks([])
    plt.ylabel('Teacher')
    cbar = plt.colorbar(label='Sync-free difference $d(x_S,x_F)$',pad=0.15,shrink=0.9)
    cbar.set_ticks([-3,-2,-1,0])
    cbar.ax.set_yticklabels(['$10^{-3}$','$10^{-2}$','$10^{-1}$','$10^{0}$'])
    plt.gcf().text(0.05, 0.8,'A',**textkw)
    

    fig12 = plt.figure(figsize=(4,4))
    plt.imshow(np.log10(measuredict["learn_free_diff"]),vmin=-3,vmax=0)
    plotrange = [-0.5,newcuts[-1]-0.5]
    if minorcuts is not None:
        for cut in minorcuts[1:-1]:
           plt.plot([cut-0.5,cut-0.5],plotrange,'lightgrey')
           plt.plot(plotrange,[cut-0.5,cut-0.5],'lightgrey')
    for cut in newcuts[1:-1]:
       plt.plot([cut-0.5,cut-0.5],plotrange,'white',linewidth=2)
       plt.plot(plotrange,[cut-0.5,cut-0.5],'white',linewidth=2)
    plt.ylim(plotrange.reverse())
    plt.xlabel('Learner')
    plt.xticks([])
    plt.yticks([])
    plt.ylabel('Teacher')
    cbar = plt.colorbar(label='Learn-free difference $d(x_L,x_F)$',pad=0.15,shrink=0.9)
    cbar.set_ticks([-3,-2,-1,0])
    cbar.ax.set_yticklabels(['$10^{-3}$','$10^{-2}$','$10^{-1}$','$10^{0}$'])
    plt.gcf().text(0.05, 0.8,'B',**textkw)

    mpl.style.use('ggplot')
    fig21 = plt.figure(figsize=(4,4))
    plt.scatter(np.log10(0.001+measuredict["sync_free_diff"].flatten()),np.log10(0.001+measuredict["sync_target_diff"].flatten()),c=measuredict["learn_autocorr"].flatten(),s=10)
    #plt.scatter(measuredict["learn_free_diff"].flatten(),measuredict["learn_target_diff"].flatten(),c=measuredict["learn_autocorr"].flatten(),s=10)
    plt.xlabel("Sync-free difference $d(x_S,x_F)$")
    plt.ylabel("Sync-target difference $d(x_S,z_T)$")
    plt.ylim([-3,0])
    plt.xlim([-3,0])
    plt.xticks([-3,-2,-1,0],['$10^{-3}$','$10^{-2}$','$10^{-1}$','$10^{0}$'])
    plt.yticks([-3,-2,-1,0],['$10^{-3}$','$10^{-2}$','$10^{-1}$','$10^{0}$'])
    plt.gcf().text(0.01, 0.95,'A',fontsize=18)
    plt.tight_layout()

    mpl.style.use('ggplot')
    fig22 = plt.figure(figsize=(4,4))
    plt.scatter(np.log10(0.001+measuredict["learn_free_diff"].flatten()),np.log10(0.001+measuredict["learn_target_diff"].flatten()),c=measuredict["learn_autocorr"].flatten(),s=10)
    #plt.scatter(measuredict["learn_free_diff"].flatten(),measuredict["learn_target_diff"].flatten(),c=measuredict["learn_autocorr"].flatten(),s=10)
    plt.xlabel("Learn-free difference $d(x_L,x_F)$")
    plt.ylabel("Learn-target difference $d(x_L,z_T)$")
    plt.ylim([-3,0])
    plt.xlim([-3,0])
    plt.xticks([-3,-2,-1,0],['$10^{-3}$','$10^{-2}$','$10^{-1}$','$10^{0}$'])
    plt.yticks([-3,-2,-1,0],['$10^{-3}$','$10^{-2}$','$10^{-1}$','$10^{0}$'])
    plt.gcf().text(0.01, 0.95,'B',fontsize=18)
    plt.tight_layout()

    x1 = (measuredict["free_period"]/(measuredict["free_period"].T)).flatten()
    x2 = (measuredict["sync_period"]/(measuredict["free_period"].T)).flatten()
    x3 = (measuredict["learn_period"]/(measuredict["free_period"].T)).flatten()
    x1 = x1[np.isnan(x1)==False]
    x2 = x2[np.isnan(x2)==False]
    x3 = x3[np.isnan(x3)==False]

    fig31 = plt.figure(figsize=(4,3))
    
    edges = np.arange(0.08,4.4,0.08)
    he = (edges[1] - edges[0])/2
    d1,_ = np.histogram(x1,bins=edges,density=True)
    d2,_ = np.histogram(x2,bins=edges,density=True)
    d3,_ = np.histogram(x3,bins=edges,density=True)
    plt.plot(edges[1:]-he,d1)
    plt.plot(edges[1:]-he,d2)
    plt.plot(edges[1:]-he,d3)    
    plt.legend(['Free','Synchronized','Learned'])
    plt.xlabel('Period ratio')
    plt.ylabel('Density')
    plt.gcf().text(0.03, 0.9,'A',**textkw)
    plt.tight_layout()




    fig32 = plt.figure(figsize=(4,3))
    x1 = measuredict["free_autocorr"].flatten()
    x2 = measuredict["sync_autocorr"].flatten()
    x3 = measuredict["learn_autocorr"].flatten()
    x4 = measuredict["mean_feedback"].flatten()
    x1 = x1[np.isnan(x1)==False]
    x2 = x2[np.isnan(x2)==False]
    x3 = x3[np.bitwise_and(np.isnan(x3)==False,x4>0)]
    violin_parts = plt.violinplot([x1,x2,x3],bw_method=0.1,showextrema=False)
    #colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    colors = []
    for i, color in enumerate(plt.rcParams['axes.prop_cycle']):
       colors.append(color['color'])
    for i,pc in enumerate(violin_parts['bodies']):
       pc.set_facecolor(colors[i])
       pc.set_edgecolor(colors[i])
    #for partname in ('cbars','cmins','cmaxes'):
    #   vp = violin_parts[partname]
    #   vp.set_edgecolor('darkgrey')
    #   vp.set_linewidth(1.5)
    plt.ylabel("Max autocorr.")  
    plt.xticks([1,2,3],labels=['Free','Synchronized','Learned'])
    plt.gcf().text(0.08, 0.85,'B',**textkw)
    plt.tight_layout()




    fig41 = plt.figure(figsize=(4,4))
    plt.imshow(np.log10(measuredict["sync_target_diff"]),vmin=-3,vmax=0)
    plotrange = [-0.5,newcuts[-1]-0.5]
    if minorcuts is not None:
        for cut in minorcuts[1:-1]:
           plt.plot([cut-0.5,cut-0.5],plotrange,'lightgrey')
           plt.plot(plotrange,[cut-0.5,cut-0.5],'lightgrey')
    for cut in newcuts[1:-1]:
       plt.plot([cut-0.5,cut-0.5],plotrange,'white',linewidth=2)
       plt.plot(plotrange,[cut-0.5,cut-0.5],'white',linewidth=2)
    plt.ylim(plotrange.reverse())
    plt.xlabel('Learner')
    plt.xticks([])
    plt.yticks([])
    plt.ylabel('Teacher')
    cbar = plt.colorbar(label='Sync-target difference $d(x_S,z_T)$',pad=0.15,shrink=0.9)
    cbar.set_ticks([-3,-2,-1,0])
    cbar.ax.set_yticklabels(['$10^{-3}$','$10^{-2}$','$10^{-1}$','$10^{0}$'])
    plt.gcf().text(0.05, 0.8,'A',**textkw)
    

    fig42 = plt.figure(figsize=(4,4))
    plt.imshow(np.log10(measuredict["learn_target_diff"]),vmin=-3,vmax=0)
    plotrange = [-0.5,newcuts[-1]-0.5]
    if minorcuts is not None:
        for cut in minorcuts[1:-1]:
           plt.plot([cut-0.5,cut-0.5],plotrange,'lightgrey')
           plt.plot(plotrange,[cut-0.5,cut-0.5],'lightgrey')
    for cut in newcuts[1:-1]:
       plt.plot([cut-0.5,cut-0.5],plotrange,'white')
       plt.plot(plotrange,[cut-0.5,cut-0.5],'white')
    plt.ylim(plotrange.reverse())
    plt.xlabel('Learner')
    plt.xticks([])
    plt.yticks([])
    plt.ylabel('Teacher')
    cbar = plt.colorbar(label='Learn-target difference $d(x_L,z_T)$',pad=0.15,shrink=0.9)
    cbar.set_ticks([-3,-2,-1,0])
    cbar.ax.set_yticklabels(['$10^{-3}$','$10^{-2}$','$10^{-1}$','$10^{0}$'])
    plt.gcf().text(0.05, 0.8,'B',**textkw)

    print('Median synced autocorr', np.median(x2))
    print('Median learned autocorr', np.median(x3))
    result = mannwhitneyu(x2,x3)
    print(result)

    return [fig11,fig12,fig21,fig22,fig31,fig32,fig41,fig42],np.median(x3)


def main(prefix,sort=True):
    directory = r'./paper4_data/'
    print(directory+prefix)
    measures = ['mean_input','free_height','free_period','free_autocorr','free_target_diff','sync_height','sync_period','sync_autocorr','sync_target_diff','sync_free_diff','learn_height','learn_period','learn_autocorr','learn_target_diff','learn_free_diff','learn_sync_diff','mean_feedback']

    cuts = [0,18,36]
    minorcuts = [0,4,8,11,14,21,24,28,32,36]
    measurearr = getdata(directory,prefix,measures,cut=[0,36,72])
    
    measuredict = {}

    cutoff = 0.5 #1.0 #minimum mean_input
    goodinds = np.where(np.nanmean(measurearr[0],axis=1)>cutoff)[0]
    newcuts = [sum(goodinds<cuts[j]) for j in range(len(cuts))]    
    #minorcuts only for unsorted indices
    newminorcuts = [sum(goodinds<minorcuts[j]) for j in range(len(minorcuts))]    
    
    for i,measure in enumerate(measures):
        measuredict.update({measure: np.array([measurearr[i][x,goodinds] for x in goodinds])})
        
    sortbyvec = np.nanmean(measuredict["sync_free_diff"],axis=0)

    if sort:
        newminorcuts = None
        sortinds = []
        for i in range(len(newcuts)-1):
           sortinds.append(newcuts[i] + np.argsort(sortbyvec[newcuts[i]:newcuts[i+1]]))
        sortinds = np.concatenate(sortinds)
        for measure in measures:
            measuredict[measure] = np.array([measuredict[measure][x,sortinds] for x in sortinds])
    
    



    print('Sync within-learner variability:', np.mean(np.nanstd(measuredict["sync_free_diff"],axis=0)))
    print('Sync within-teacher variability:', np.mean(np.nanstd(measuredict["sync_free_diff"],axis=1)))
    
    print('Learned within-learner variability:', np.mean(np.nanstd(measuredict["learn_free_diff"],axis=0)))
    print('Learned within-teacher variability:', np.mean(np.nanstd(measuredict["learn_free_diff"],axis=1)))
    
    sync_success = np.sum(measuredict["sync_free_diff"]>measuredict["sync_target_diff"])/newcuts[-1]/(newcuts[-1]+1)
    learn_success =  np.sum(measuredict["learn_free_diff"]>measuredict["learn_target_diff"])/newcuts[-1]/(newcuts[-1]+1)
    learn_success2 =  np.sum(measuredict["learn_free_diff"]>measuredict["learn_target_diff"])/np.sum(np.isnan(measuredict["learn_target_diff"])==False)
    print('% synced closer to target than free:', sync_success)
    print('% learned closer to target than free:', learn_success)
    print('% excluding NaNs:', learn_success2)



    figs,med_autocorr = plot(measuredict,measures,newcuts,newminorcuts)
    
    return figs,learn_success,med_autocorr,measuredict,goodinds
    
if __name__ == '__main__':
    
    
    #main figures
    allgamma = np.array([5,10,15,20])
    allsuccess = []
    allstability = []
    _,perc_closer,med_autocorr,_,_ = main('singlet_adaptthresh3_')
    allsuccess.append(perc_closer)
    allstability.append(med_autocorr)
    figs,perc_closer,med_autocorr,_,_ = main('singlet_adaptthresh4_')
    allsuccess.append(perc_closer)
    allstability.append(med_autocorr)
    _,perc_closer,med_autocorr,_,_ = main('singlet_adaptthresh5_')
    allsuccess.append(perc_closer)
    allstability.append(med_autocorr)
    _,perc_closer,med_autocorr,_,_ = main('singlet_adaptthresh6_')
    allsuccess.append(perc_closer)
    allstability.append(med_autocorr)

    plt.style.use('default')
    metafig,ax1 = plt.subplots(figsize=(4,3))
    
    ax1.plot(allgamma,100*np.array(allsuccess),color='royalblue')
    ax1.set_ylabel('Learning success rate (%)',color='royalblue')
    ax1.set_xlabel(r'LIF decay rate $\Gamma$ (s$^{-1}$)')
    ax2 = ax1.twinx()
    ax2.plot(allgamma,allstability,color='tan')
    ax2.set_ylabel('Median autocorrelation peak height',color='tan')
    plt.gcf().text(0.03, 0.92,'B',fontsize=16)
    plt.tight_layout()
    

    #supplementary figures (unsorted) 
    #figs,perc_closer,med_autocorr,measuredict,goodinds = main('singlet_adaptthresh4_',sort=False)

    