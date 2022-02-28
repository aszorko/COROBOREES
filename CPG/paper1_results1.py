# -*- coding: utf-8 -*-
"""
This file creates the evolution-related figures for the paper
"Rapid rhythmic entrainment in bio-inspired central pattern generators"
Set the runmode variable in the main function to choose a figure
Run mode 3 also generates the final set of CPGs for the next part of the analysis

@author: alexansz
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def main(filename,datacols,startmode=0):
    f = open(filename,'r')
    lines = f.readlines()

    data = []
    inds = []
    scores = []

    indata = startmode
    for line in lines:
        cols = line.replace('[','').replace(']','').split()
        if cols[0] == 'gen':
            indata = 1
            continue
        if indata == 0:
            continue
        if ',' in cols[0]:
            indata = 2
        if indata==1:
           data.append([float(cols[x]) for x in datacols])
        elif indata==2:
           if '(' in line:
               scores.append([float(x) for x in line.strip(')(\n').split(', ')])
           elif '[' in line:
               inds.append([int(x) for x in line.strip('][\n').split(', ')])

    return data,inds,scores


def weights(inds,scorearr,zstart=0,zinc=1,maxcount=100):
    #select individuals based on weighted fitness. set maxcount to zero to disable iteration

    n = len(scorearr[0,:])
    z = zstart
    w = np.array([0 for i in range(n)])

    #for i in range(n):
    #   colmax = np.max(scorearr[:,i])
    #   colmin = np.min(scorearr[:,i])
    #   scorearr[:,i] = (scorearr[:,i]-colmin)/(colmax-colmin)

    w0 = np.argmax(np.sum(scorearr,1))
    print('Overall:')
    print(inds[w0])
    print(scores[w0])
    scorearr[w0,:] = 0*scorearr[w0,:]


    count = 0
    while True:
        count +=1
        for i in range(n):
           w[i] = np.argmax(np.sum(scorearr,1) + z*scorearr[:,i])
        if count>maxcount or len(np.unique(w)) == n:
            print('Process terminated at z =', z)
            break
        z += zinc

    for i in range(n):
        print(f'F{i+1} weighted:')
        print(inds[w[i]])
        print(scores[w[i]])


def ndsort(scorearr):
    n = len(scorearr[:,0])
    m = len(scorearr[0,:])
    removed = 0
    for i in range(n):
        for j in range(n):
            if i != j and sum(scorearr[j,:]>scorearr[i,:])==m:
                scorearr[i,:] = 0*scorearr[i,:]
                removed += 1
                break
    print('Removed', removed, 'dominated individuals')
    return scorearr


if __name__ == "__main__":


    runmode = 2 #1=plot CPG evolution vs time, 2=filter evolution vs time, 3=get best individuals after final eval

    if runmode==1:
       mpl.style.use('seaborn-colorblind')
       data,inds,scores = main('paper1_data/cpg_nsga_master.txt',[8,9,10])
       dataarr = np.array(data)
       fig1 = plt.figure()
       lines = plt.plot(1+np.arange(len(data)),dataarr)
       plt.xlabel('Generation',fontsize=16)
       plt.ylabel('Mean fitness',fontsize=16)
       plt.legend(lines,[r'$F_1$',r'$F_2$',r'$F_3$'])
       plt.show()

    if runmode==2:
       mpl.style.use('seaborn-colorblind')
       lines = []
       styles = ['--','-']
       filelist = ['brain0_n2_g25.txt','brain0_n4_g50.txt','brain1_n2_g25.txt','brain1_n4_g50.txt','brain2_n2_g25.txt','brain2_n4_g50.txt','brain3_n2_g25.txt','brain3_n4_g50.txt']
       n = len(filelist)
       fig2 = plt.figure()
       for i,file in enumerate(filelist):
          data,inds,scores = main('paper1_data/' + file,[11,12,13])
          dataarr = np.array(data)
          line = plt.plot((1+np.arange(len(data))),np.mean(dataarr,axis=1),styles[i%2],color='C'+str(i//2))
          lines.append(line[0])
       plt.xlabel('Generation',fontsize=16)
       plt.ylabel('Max fitness',fontsize=16)
       plt.legend(lines[1:len(filelist):2],['CPG'+str(i) for i in range(n//2)],loc='lower right',fontsize='small')

    if runmode==3:
       mpl.style.use('default')
       data,inds,scores = main('paper1_data/cpg_finaleval.txt',[8,9,10],startmode=2)
       scorearr = np.array(scores)
       scorearr = ndsort(scorearr)
       weights(inds,scorearr)
       fig3 = plt.figure()
       ax = fig3.add_subplot(111, projection="3d")

       # the coordinate origin
       ax.scatter(0, 0, 0, c="k", marker="+", s=100)
       inds = np.sum(scorearr,axis=1)>0
       ax.scatter(scorearr[inds,0],scorearr[inds,1],scorearr[inds,2], c=scorearr[inds,2], cmap='viridis', marker="o", s=32)

       # final figure details
       ax.set_xlabel("$F_1$", fontsize=15)
       ax.set_ylabel("$F_2$", fontsize=15)
       ax.set_zlabel("$F_3$", fontsize=15)
       ax.view_init(elev=45, azim=230)
       ax.autoscale(tight=True)
       plt.tight_layout()
       plt.show()
