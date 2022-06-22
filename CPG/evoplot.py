# -*- coding: utf-8 -*-
"""
Contains functions for processing data from NSGA evolution
Run from command prompt with filename to get a CPG subset via fitness weighting

@author: alexansz
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys


def main(filename,datacols,startmode=0,getheader=False):
    #Text file parser. Used by several other scripts.
    f = open(filename,'r')
    lines = f.readlines()

    data = []
    inds = []
    scores = []
    header = []

    indata = startmode
    for line in lines:
        cols = line.replace('[','').replace(']','').split()
        if cols[0] == 'gen':
            indata = 1
            continue
        if indata == 0:
            header.append(line)
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
    if getheader:
        return data,inds,scores,header
    else:
        return data,inds,scores


def weights(inds,scorearr,zstart=0,zinc=1,maxcount=100):
    #select individuals based on weighted fitness. set maxcount to zero to disable iteration

    n = len(scorearr[0,:])
    z = zstart
    w = np.array([0 for i in range(n)])


    w0 = np.argmax(np.sum(scorearr,1))
    print('Overall:')
    print(list(inds[w0]))
    print(list(scorearr[w0]))
    #scorearr[w0,:] = 0*scorearr[w0,:]


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
        print(list(inds[w[i]]))
        print(list(scorearr[w[i]]))
        #print(np.sum(scorearr[w[i],:]) + z*scorearr[w[i],i])


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


    plot = False       #plot pareto front (or projection thereof)
    filter_nd = False  #remove dominated individuals
    filter_neg = True  #remove individuals with 1 or more negative weights
    
    filename = sys.argv[1]
    
    mpl.style.use('default')
    data,inds,scores = main(filename,[],startmode=2)
    scorearr = np.array(scores)
    inds_u,indices = np.unique(inds,axis=0,return_index=True)
    print(len(indices),"unique individuals")
    scorearr_u = scorearr[indices,:]
    if filter_neg:
       inds_pos = np.sum(scorearr_u<0,axis=1)==0
       inds_u = inds_u[inds_pos,:]
       scorearr_u = scorearr_u[inds_pos,:]
       print(len(inds_u),"with positive scores")
    if filter_nd:
       scorearr = ndsort(scorearr)
    weights(inds_u,scorearr_u)
    
    if plot:
       fig3 = plt.figure()
       ax = fig3.add_subplot(111, projection="3d")

       # the coordinate origin
       ax.scatter(0, 0, 0, c="k", marker="+", s=100)
       ins = np.sum(scorearr,axis=1)>0
       ax.scatter(scorearr[ins,0],scorearr[ins,1],scorearr[ins,2], c=scorearr[ins,2], cmap='viridis', marker="o", s=32)

       # final figure details
       ax.set_xlabel("$F_1$", fontsize=15)
       ax.set_ylabel("$F_2$", fontsize=15)
       ax.set_zlabel("$F_3$", fontsize=15)
       ax.view_init(elev=70, azim=230)
       ax.autoscale(tight=True)
       plt.tight_layout()
       plt.show()
