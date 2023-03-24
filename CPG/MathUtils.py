# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 11:01:09 2023

@author: alexansz
"""

import numpy as np

def hinge(x):
    #rectifying linear unit
    y = 0*x
    for i in range(len(x)):
        if x[i] < 0:
            y[i] = 0
        else:
            y[i] = x[i]
    return y


def sig(x):
    #sigmoid centred at zero
    return 1 / (1 + np.exp(-x)) - 0.5


def autocorr(allx, start, mindelay, maxdelay=-1):
    n_i = allx.shape[0]
    tt = allx.shape[1]
    tstart = round(tt*start)
    out = np.zeros([n_i, 2*(tt-tstart)-1])
    if maxdelay < 0:
        maxdelay = tt-tstart
    for i in range(n_i):
        x = allx[i, tstart:]
        out[i, :] = np.real(np.correlate(
            x - np.mean(x), x - np.mean(x), mode='full'))
        #new: remove limbs with no correlation peak
        if np.argmax(out[i,(len(x)+mindelay):(len(x)+maxdelay)]) == 0:
            out[i, :] = 0
    outtot = np.sum(out, axis=0)
    peak = mindelay + np.argmax(outtot[(len(x)+mindelay):(len(x)+maxdelay)])
    height = np.max(outtot[len(x)+mindelay:])
    if peak == mindelay:
        peak = 0

    return peak, height