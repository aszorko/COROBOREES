# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 11:28:16 2022

@author: alexansz
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import evoplot
import statsmodels.api as sm
import statsmodels.formula.api as smf
from matplotlib.offsetbox import AnchoredText


def bestcpg():
    
    allbestscores = []
    files1 = ['./paper2_data/nsga_unity_t4_g200_' + str(i) + '_final.txt' for i in range(1,6)]
    files2 = ['./paper2_data/nsga_unityshort_t4_g200_' + str(i) + '_final.txt' for i in range(1,6)]
    for file in files1:
       data,_,scores = evoplot.main(file,[],startmode=2) 

       scorearr = np.array(scores)
       inds = np.sum(scorearr<0,axis=1)==0
       allbestscores.append(np.max(np.mean(scorearr[inds,:],axis=1)))

    for file in files2:
       data,_,scores = evoplot.main(file,[],startmode=2) 

       scorearr = np.array(scores)
       inds = np.sum(scorearr<0,axis=1)==0
       allbestscores.append(np.max(np.mean(scorearr[inds,:],axis=1)))

    print(np.max(allbestscores))
    print(np.argmax(allbestscores))


def evodata():
    allnormal = []
    allshort = []

    files1 = ['./paper2_data/nsga_unity_t4_g200_' + str(i) + '.txt' for i in range(1,6)]
    files2 = ['./paper2_data/nsga_unityshort_t4_g200_' + str(i) + '.txt' for i in range(1,6)]
    for file in files1:
       data,_,scores = evoplot.main(file,[14,15,16,17]) 
       allnormal.append(data)

    for file in files2:
       data,_,scores = evoplot.main(file,[14,15,16,17]) 
       allshort.append(data)
    
    return np.array(allnormal), np.array(allshort)

def cpgdata(file1,plotmeasures=[],label=''):
    #markers = ['.','+','*','^','x']
    markers = ['.','.','.','.','.']
   
    textkw = {'fontsize':16}

    data,cpgs,scores = evoplot.main(file1,[8,9,10],startmode=2)
    scorearr = np.array(scores)       
    scoredf = pd.DataFrame(scorearr,columns = ['F1','F2','F3','F4','fdist','fdist2','period','period2','height','height2','tilt','tilt2','corrmax','corrmax2','corrind','corrind2','run'])

    #run = np.concatenate([[i+1 for j in range(n[i])] for i in range(len(n))])
    legends = []
    corrtypes = ['(Negative)','Diagonal','Left/Right','Hind/Front']
    fig1 = plt.figure()
    mpl.style.use('seaborn-colorblind')
    scoredf.corrind[scoredf.corrind==11] = 1
    scoredf.corrind[scoredf.corrind==7] = 2
    scoredf.corrind[scoredf.corrind==6] = 3
    for i in range(4):
        if sum(scoredf.corrind==i)>0:
            plt.hist(scoredf.corrmax[scoredf.corrind==i],bins=np.arange(-0.2,1,0.05),alpha=0.5)
            legends.append(corrtypes[i])
    plt.xlabel('Maximum correlation',**textkw)
    plt.ylabel('# CPGs',**textkw)
    plt.legend(legends)
    plt.gcf().text(0.03, 0.85,label,**textkw)
    plt.show()

    fig2 = plt.figure()
    plt.hist(scoredf.height,bins=np.arange(0,1.35,0.05))
    plt.xlabel('Average height (normalized)',**textkw)
    plt.ylabel('# CPGs',**textkw)
    plt.gcf().text(0.03, 0.85,label,**textkw)
    plt.show()
    
    scoredf.corrind[scoredf.corrmax<0.3] = 0
    scoredf.corrind2[scoredf.corrind2==11] = 1
    scoredf.corrind2[scoredf.corrind2==7] = 2
    scoredf.corrind2[scoredf.corrind2==6] = 3
    scoredf.corrind2[scoredf.corrmax2<0.3] = 0
    scoredf["diffcorr"] = (scoredf.corrmax2-scoredf.corrmax)
    scoredf["sumcorr"] = (scoredf.corrmax2+scoredf.corrmax)
    scoredf["diffperiod"] = (scoredf.period2-scoredf.period)
    scoredf["sumperiod"] = (scoredf.period+scoredf.period2)
    scoredf["diffdist"] = (scoredf.fdist2-scoredf.fdist)
    scoredf["Fdiff"] = scoredf.F3 - scoredf.F2
    #1,11 = trot
    #2,7 = biped walk
    #3,6 = bound
    
    print(len(scoredf),"unique individuals")
    
    #initial filtering
    inds0 = np.logical_and(scoredf.height>0.75,scoredf.period>0)
    inds0 = np.logical_and(inds0,scoredf.height2>0.75)
    
    #plot by run
    for measurepair in plotmeasures:
       for i in range(5):
           inds = np.logical_and(inds0,scoredf.run==i+1)
           newdf = scoredf
           if len(measurepair) > 2:
              plt.scatter(newdf[measurepair[0]][inds],newdf[measurepair[1]][inds],c=newdf[measurepair[2]][inds],marker=markers[i],vmin=measurepair[3],vmax=measurepair[4])
           else:
              plt.scatter(newdf[measurepair[0]][inds],newdf[measurepair[1]][inds],marker=markers[i])               
       plt.xlabel(measurepair[0])
       plt.ylabel(measurepair[1])
       plt.show()
       
    return scoredf[inds0], np.array(cpgs)[inds0], [fig1,fig2]

def braindata(file2base,plotmeasures,cpglist=[],plot=False,labels=[],titles=True):
    measures = ['dist','period','height','tilt','corrmax','corrind']
    measures_pretty = ['Distance','Period','Height','Tilt','Corr.','Corr ind']
    alldy = []
    alldz = []
    allz = []
    allheight = []
    allind = []
    allevodata = []
    allbrains = []
    brainscores = []
    brainheights = []
    allfigs = []
    minheight = 0.75
    mpl.style.use('default')
    textkw = {'fontdict':{'fontsize':16}}
    
    for run in range(1,6):
        for cpg in range(1,5):
            
            if plot and (run,cpg) in cpglist:
                #newfig = plt.figure()
                #gs = newfig.add_gridspec(1,len(plotmeasures),hspace=0.1,wspace=0.4)
                #axs = gs.subplots().flat
                newfig, axs = plt.subplots(ncols=len(plotmeasures), nrows=1, figsize=(10, 4), gridspec_kw={'wspace':0.25})
                j=0
            
            brainfile = file2base + str(run) + '_brain' + str(cpg) + '_final2.txt'
            try:
                _,brains,bscores = evoplot.main(brainfile,[11,12,13],startmode=2)
            except:
                continue

            brainfile = file2base + str(run) + '_brain' + str(cpg) + '.txt'
            evodata,_,_ = evoplot.main(brainfile,[11,12,13])
            allevodata.append(evodata)

            i=0
            currscore = np.nan
            currheight = np.nan
            currbrain = []
            while i<len(bscores):
               i += 1
               if np.sum(np.array(bscores[-i])[::2]<minheight)==0:
                   currheight = np.mean(bscores[-i][::2])
                   currscore = np.mean(bscores[-i][1::2])
                   currbrain = brains[-i]
                   break
            brainscores.append(currscore)
            brainheights.append(currheight)
            allbrains.append(currbrain)
            
            for i,meas in enumerate(measures):
                file = file2base + str(run) + '_cpg' + str(cpg) + '_' + meas + '.txt'
                df = pd.read_csv(file)           
                
                arr = df.to_numpy()
                if meas=='dist':
                    vmin=-10
                    vmax=10 
                if meas=='period':
                    vmin=0
                    vmax=15
                    y1 = arr[14,4]
                    y0 = arr[10,4]
                    if y0==0:
                        dy = 0
                    else:
                        dy = (y1-y0)/y0
                        
                        if dy<-0.4:
                            dy = (2*y1-y0)/y0
                        elif dy>0.8:
                            dy = (y1/2-y0)/y0
                        
                    #print(dy)
                    alldy.append(dy*y0/0.2)
                if meas=='height':
                    vmin=0
                    vmax=1
                    allheight.append(arr[10,4])
                if meas=='tilt':
                    vmin=0
                    vmax=1
                if meas=='corrmax':
                    vmin=0
                    vmax=1
                    if arr[10,4]==0:
                        dz = 0
                    else:
                        dz = (arr[14,4]-arr[10,4])/arr[10,4]
                    #print(dz)
                    alldz.append(dz)
                    allz.append(arr[10,4])
                if meas=='corrind':
                    arr[arr==7] = 2
                    arr[arr==6] = 3
                    arr[arr==11] = 1
                    allind.append(arr[10,4])
                if plot and meas in plotmeasures and (run,cpg) in cpglist:
                    im = axs[j].imshow(arr,extent=(-0.0225,0.0225,1.025,-0.025),aspect='auto') #,vmin=vmin,vmax=vmax)
                    #plt.title(' '.join(['run',str(run),'cpg',str(cpg),meas]))
                    axs[j].plot([-0.016,-0.016,0.016,0.016,-0.016],[1,0.5,0.5,1,1],'k')
                    if titles:
                        axs[j].set_title(measures_pretty[i],**textkw)
                    axs[j].set_xlabel(r'$\theta_C$',**textkw)
                    if j==0:
                        axs[j].set_ylabel(r'$I_{DC}$',**textkw)
                    else:
                        axs[j].set_yticks([])
                        
                    if meas=='corrmax':
                        fmt = '%.3f'
                    elif meas=='dist':
                        fmt = '%.1f'
                    else:
                        fmt = None
                    plt.colorbar(im,ax=axs[j],shrink=0.8,fraction=0.1,format=fmt)
                    
                    j += 1
                    
            if plot and (run,cpg) in cpglist:
                if len(labels)>0:
                    plt.gcf().text(0.05, 0.85,labels.pop(0),**textkw)
                plt.subplots_adjust(bottom=0.12)
                allfigs.append(newfig)
                
    #df = pd.DataFrame(data=np.array([brainscores,brainheights,np.sqrt(np.array(alldy)),np.sqrt(np.array(alldz)),allz,allheight,allind]).T,columns=["brainscore","brainheight","diffperiod","diffcorr","corrmax","height","corrind"])
    df = pd.DataFrame(data=np.array([brainscores,brainheights,alldy,alldz,allz,allheight,allind]).T,columns=["brainscore","brainheight","diffperiod","diffcorr","corrmax","height","corrind"])
    df.corrind[df.corrmax<0.3] = 0
    return df, np.array(allevodata), allbrains, allfigs

if __name__ == "__main__":
    
    #1: plot CPG scatter plots, 2: CPG stats, 3: CPG heat plots, 4: brain stats, 5: brain evolution, 6: get best CPG, 7: CPG evolution, 8: control parameters figure
    runmode = [1,2,3,4,5,6,7,8]

    mpl.style.use('default')
    textkw = {'fontsize':16}
    
    file1 = './paper2_data/cpg_short_alldata_5runs_forward.txt'
    file1_short = './paper2_data/cpg_alldata_5runs_forward.txt'
    file2base = './paper2_data/unity_run'
    file2base_short = './paper2_data/unityshort_run'

    if 1 in runmode:
       scatterplots  = [("F3","diffperiod","corrind",0,3)]
    else:
       scatterplots = []

    if 1 in runmode or 2 in runmode:
       cpg_df1, cpgs1, figs11 = cpgdata(file1,plotmeasures=scatterplots,label='A')
       cpg_df1["short"] = 0
       cpg_df2, cpgs2, figs12 = cpgdata(file1_short,plotmeasures=scatterplots,label='B')
       cpg_df2["short"] = 1
       cpg_df2["run"] = cpg_df2["run"] + np.max(cpg_df1["run"])
       scoredf = pd.concat([cpg_df1,cpg_df2])
       allcpgs = np.concatenate([cpgs1,cpgs2])
       
    if 2 in runmode:
    
       
       textkw = {'fontdict':{'fontsize':16}}
       
       
       fig20 = plt.figure()
       mpl.style.use('default')
       shortlab = ['Normal','Short']
       gaitlab = ['Walk','Trot','Pace','Bound']
       ax = fig20.add_subplot()
       scoredf.boxplot(column='F3',by=['short','corrind'],backend='matplotlib',ax=ax,patch_artist=True,boxprops={"facecolor": "lightgray", "edgecolor": "black","linewidth": 1},whiskerprops={"linewidth": 1})
       cats = scoredf.groupby(by=['short','corrind']).size()
       labs = ['\n'.join([shortlab[cats.index[i][0]],gaitlab[int(cats.index[i][1])],'('+str(cats.iat[i])+')']) for i in range(len(cats))]       
       plt.xticks(ticks=list(range(1,len(cats)+1)),labels=labs)
       plt.ylabel('$F_3$',**textkw)
       plt.xlabel('')
       plt.title('')
       plt.suptitle('')
       plt.gcf().text(0.03, 0.93,'A',**textkw)
       plt.tight_layout()
       plt.show()
       


       fig21 = plt.figure()
       mpl.style.use('ggplot')
       mpl.style.use('seaborn-colorblind')
       inds = scoredf.short==0
       plt.scatter(scoredf.corrmax2[inds],scoredf.period[inds],c=scoredf.F3[inds],marker='.',vmin=-15,vmax=15)
       inds = scoredf.short==1
       plt.scatter(scoredf.corrmax2[inds],scoredf.period[inds],c=scoredf.F3[inds],marker='*',vmin=-15,vmax=15)
       plt.xlabel('Interlimb correlation',**textkw)
       plt.ylabel('Period (s)',**textkw)
       plt.colorbar(label='$F_3$')
       plt.legend(['normal','short'])
       plt.gcf().text(0.03, 0.93,'B',**textkw)
       plt.tight_layout()
       plt.show()

       #stats
       data = scoredf 
       md0 = smf.mixedlm("F3 ~ (diffperiod + diffcorr + sumcorr + sumperiod)*short", data, groups=data["run"])
       mdf0 = md0.fit()
       print(mdf0.summary())

       data = scoredf[scoredf.short==0]
       meanF3_1 = np.mean(data.F3)
       meandiffc1 = np.mean(data.diffcorr)
       meandiffp1 = np.mean(data.diffperiod)
       data = data.apply(lambda x: (x - np.mean(x)))
       md1 = smf.mixedlm("F3 ~ diffperiod + diffcorr + sumcorr + sumperiod", data, groups=data["run"])
       #md = sm.OLS(data.F3,np.array([data.diffperiod,data.diffcorr]).T)
       mdf1 = md1.fit()
       print(mdf1.summary())
       print(sm.stats.diagnostic.kstest_normal(mdf1.resid))

       data = scoredf[scoredf.short==1]
       meanF3_2 = np.mean(data.F3)
       meandiffc2 = np.mean(data.diffcorr)
       meandiffp2 = np.mean(data.diffperiod)
       data = data.apply(lambda x: (x - np.mean(x)))       
       md2 = smf.mixedlm("F3 ~ diffperiod + diffcorr + sumcorr + sumperiod", data, groups=data["run"])
       #md = sm.OLS(data.F3,np.array([data.diffperiod,data.diffcorr]).T)
       mdf2 = md2.fit()
       print(mdf2.summary())
       print(sm.stats.diagnostic.kstest_normal(mdf2.resid))


       
       fig22 = plt.figure()
       inds = scoredf.short==0
       plt.scatter(scoredf.diffcorr[inds],scoredf.F3[inds],marker='.')
       inds = scoredf.short==1
       plt.scatter(scoredf.diffcorr[inds],scoredf.F3[inds],marker='*')
       x = np.array([-0.5,0.5])
       #intercept1 = mdf1.params['Intercept'] + sum([np.mean(scoredf[var]) for var in md1.exog_names if var not 'Intercept' and var not 'diffcorr'])
       plt.plot(x,meanF3_1 + mdf1.params['diffcorr']*(x-meandiffc1))
       plt.plot(x,meanF3_2 + mdf2.params['diffcorr']*(x-meandiffc2))       
       #plt.plot(x,mdf0.params['Intercept'] + (mdf0.params['diffcorr']+mdf0.params['diffcorr:short'])*x)
       plt.xlabel('Interlimb correlation difference',**textkw)
       plt.ylabel('$F_3$',**textkw)
       plt.legend(['normal','short'])
       plt.gcf().text(0.03, 0.93,'A',**textkw)
       plt.tight_layout()
       plt.show()

       fig23 = plt.figure()
       inds = scoredf.short==0
       plt.scatter(scoredf.diffperiod[inds],scoredf.F3[inds],marker='.')
       inds = scoredf.short==1
       plt.scatter(scoredf.diffperiod[inds],scoredf.F3[inds],marker='*')
       x = np.array([-0.5,0.5])
       plt.plot(x,meanF3_1 + mdf1.params['diffcorr']*(x-meandiffp1))
       plt.plot(x,meanF3_2 + mdf2.params['diffcorr']*(x-meandiffp2))       
       plt.xlim([-0.2,0.2])
       plt.xlabel('Period difference (s)',**textkw)
       plt.ylabel('$F_3$',**textkw)
       plt.legend(['normal','short'])
       plt.gcf().text(0.03, 0.93,'B',**textkw)
       plt.tight_layout()
       plt.show()
       
       fig24 = plt.figure()
       inds = scoredf.short==1
       plt.scatter(scoredf.diffperiod[inds],scoredf.diffcorr[inds],c=scoredf.F3[inds],marker='.')
       plt.xlim([-0.1,0.1])
       
       
       

    if 3 in runmode or 4 in runmode or 5 in runmode:
       #mpl.style.use('default') 
       
       measures = ['period','height','corrmax','dist']
       normalplots = [(5,4)] #best overall CPG score
       shortplots = [(3,4)] #best Q, most negative dT/dI
       
       if 3 in runmode:
           heatplot = True
       else:
           heatplot = False
       
       df1,ev1,brains1,figs1 = braindata(file2base,measures,cpglist=normalplots,plot=heatplot,labels=['A']) 
       df1["short"] = 0
       df2,ev2,brains2,figs2 = braindata(file2base_short,measures,cpglist=shortplots,plot=heatplot,labels=['B'],titles=False) 
       df2["short"] = 1
       braindf = pd.concat([df1,df2])
    
    if 4 in runmode:
        
       inds = np.isnan(braindf.diffcorr)==False

       data2 = braindf[inds]
       meandiff = np.mean(data2.diffperiod)
       meanscore = np.mean(data2.brainscore)
       data2 = data2.apply(lambda x: (x - np.mean(x)))       

       md = sm.OLS(data2.brainscore,np.array([np.ones([len(data2)]),data2.diffperiod,data2.diffcorr,data2.short,data2.corrmax]).T)
       mdf = md.fit()
       print(mdf.summary())
       print(sm.stats.diagnostic.kstest_normal(mdf.resid))
       
       
       fig4 = plt.figure()
       mpl.style.use('ggplot')
       mpl.style.use('seaborn-colorblind')
       for i in [0,1,3]:
          inds = braindf.corrind==i
          plt.scatter(braindf.diffperiod[inds],braindf.brainscore[inds])          
       plt.legend(['walk','trot','bound'])
       
       xvec = np.array([-0.2,0.4])
       yvec = (xvec-meandiff)*mdf.params['x1'] + meanscore
       plt.plot(xvec,yvec,'--k')          
       
       plt.xlabel('$\Delta T / \Delta I_{DC}$ (s)',**textkw)
       plt.ylabel('Mean of $Q_k$',**textkw)
       plt.gcf().text(0.03, 0.85,'B',**textkw)
       plt.show()
       
    if 5 in runmode:
       
       textkw = {'fontsize':18} 
       fig5 = plt.figure()
       mpl.style.use('default')
       mpl.style.use('seaborn-colorblind')
       maxfits1 = np.quantile(ev1,0.75,axis=0)
       medfits1 = np.median(ev1,axis=0)
       minfits1 = np.quantile(ev1,0.25,axis=0)
       maxfits2 = np.quantile(ev2,0.75,axis=0)
       medfits2 = np.median(ev2,axis=0)
       minfits2 = np.quantile(ev2,0.25,axis=0)
       x1 = list(range(len(medfits1)))
       for i in range(3):
           plt.fill_between(x1,minfits1[:,i],maxfits1[:,i],alpha=0.3)
           #plt.fill_between([x1[-1]+1,x1[-1]+5],[minfits1[-1,i],minfits1[-1,i]],[maxfits1[-1,i],maxfits1[-1,i]],alpha=0.4)
       plt.gca().set_prop_cycle(None)
       for i in range(3):
           plt.fill_between([x1[-1]+6,x1[-1]+10],[minfits2[-1,i],minfits2[-1,i]],[maxfits2[-1,i],maxfits2[-1,i]],alpha=0.3)
       plt.plot(medfits1)
       plt.legend(['short period','natural period','long period'],loc='lower right')
       plt.gca().set_prop_cycle(None)
       plt.plot(medfits2,'--')
       plt.gca().set_prop_cycle(None)
       plt.plot([x1[-1]+8],[medfits2[-1,:]],'.')
       plt.plot([x1[-1]+3.5,x1[-1]+3.5,x1[-1]+12,x1[-1]+12,x1[-1]+3.5],[0.35,1.25,1.25,0.35,0.35],color='grey')
       plt.xlabel('Generation',**textkw)
       plt.ylabel('Max fitness',**textkw)
       plt.gcf().text(0.02, 0.85,'A',**textkw)
       plt.show()
    
    if 6 in runmode:
       bestcpg()
       
    if 7 in runmode:
       textkw = {'fontsize':18}
       cpgev1,cpgev2 = evodata() 
       fig7 = plt.figure()
       mpl.style.use('default')
       mpl.style.use('seaborn-colorblind')
       maxfits1 = np.max(cpgev1,axis=0)
       medfits1 = np.median(cpgev1,axis=0)
       minfits1 = np.min(cpgev1,axis=0)
       maxfits2 = np.max(cpgev2,axis=0)
       medfits2 = np.median(cpgev2,axis=0)
       minfits2 = np.min(cpgev2,axis=0)
       x1 = list(range(len(medfits1)))
       for i in range(4):
           plt.fill_between(x1,minfits1[:,i],maxfits1[:,i],alpha=0.3)
           #plt.fill_between([x1[-1]+1,x1[-1]+5],[minfits1[-1,i],minfits1[-1,i]],[maxfits1[-1,i],maxfits1[-1,i]],alpha=0.4)
       plt.gca().set_prop_cycle(None)
       for i in range(4):
           plt.fill_between([x1[-1]+6,x1[-1]+10],[minfits2[-1,i],minfits2[-1,i]],[maxfits2[-1,i],maxfits2[-1,i]],alpha=0.3)
       plt.plot(medfits1)
       plt.legend(['$F_1$','$F_2$','$F_3$','$F_4$'])
       plt.gca().set_prop_cycle(None)
       plt.plot(medfits2,'--')
       plt.plot([x1[-1]+3.5,x1[-1]+3.5,x1[-1]+12,x1[-1]+12,x1[-1]+3.5],[0,20,20,0,0],color='grey')
       plt.gca().set_prop_cycle(None)
       plt.plot([x1[-1]+8],[medfits2[-1,:]],'.')
       plt.xlabel('Generation',**textkw)
       plt.ylabel('Max fitness',**textkw)
       plt.gcf().text(0.02, 0.85,'B',**textkw)
       plt.show() 
    
    if 8 in runmode:
        
       textkw = {'fontsize':14}
       
       t = [0,10,12,20,30]
       br = [1,0.5,0.5,0.5,1]
       tilt = [-0.016,-0.016,0.016,0.016,0.016]
       fig8 = plt.figure()
       mpl.style.use('default')
       gs = fig8.add_gridspec(3,1, hspace=0.25,wspace=0.25)
       axs = gs.subplots().flat
       axs[1].plot(t,br)
       axs[1].set_ylabel("$I_{DC}$",**textkw)
       axs[1].set_xticks([])
       axs[1].set_yticks([0.5,1])
       axs[1].set_xlim([0,30])
       axs[2].plot(t,tilt)
       axs[2].set_ylabel(r"$\theta_C$",**textkw)
       axs[2].set_xlabel("Time (s)",**textkw)
       axs[2].set_yticks([-0.016,0,0.016])
       axs[2].set_xlim([0,30])
       axs[0].set_ylim([-1,1])
       axs[0].set_xlim([0,30])
       axs[0].set_xticks([])
       axs[0].set_yticks([])
       axs[0].annotate(text='', xy=(30,-0.3), xytext=(0,-0.3), arrowprops=dict(arrowstyle='<->', color='k', lw=1, shrinkA=0, shrinkB=0))
       axs[0].annotate(text='', xy=(10,-0.9), xytext=(0,-0.9), arrowprops=dict(arrowstyle='<->', color='k', lw=1, shrinkA=0, shrinkB=0))
       axs[0].annotate(text='', xy=(20,-0.9), xytext=(10,-0.9), arrowprops=dict(arrowstyle='<->', color='k', lw=1, shrinkA=0, shrinkB=0))
       axs[0].annotate(text='', xy=(30,-0.9), xytext=(20,-0.9), arrowprops=dict(arrowstyle='<->', color='k', lw=1, shrinkA=0, shrinkB=0))
       axs[0].text(2,-0.8,'$F_1$ (backwards)')
       axs[0].text(12.5,-0.8,'$F_2$ (forwards)')
       axs[0].text(21.5,-0.8,'$F_3$ (accelerating)')
       axs[0].text(12.5,-0.2,'$F_4$ (upright)')
       axs[0].axis('off')
  