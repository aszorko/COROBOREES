# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 15:04:36 2022

To be run after nsga_optimize_cpg.py
Goes through the entire final evolved population and evaluates n times
Uses multiprocessing

Run and output to file: python final_eval_cpg.py >finaleval.txt

@author: alexansz
"""

import matsuoka_quad
import evoplot
import multiprocessing
import numpy as np

n = 5
data,inds,scores = evoplot.main('./cpg_nsga_master.txt',[8,9,10])
futures = multiprocessing.Pool(12)

allfits = []

for i in range(n):
    fitnesses = futures.map(matsuoka_quad.run_from_array, inds)
    allfits.append(np.array(fitnesses))
    
    
    
meanfits = np.median(allfits,axis=0)
for i in range(len(inds)):
    print(inds[i])
    print(tuple(meanfits[i]))



    

