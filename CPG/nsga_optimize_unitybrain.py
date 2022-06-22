# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 13:21:02 2021

@author: alexansz
"""

from math import factorial
import random

import numpy

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

import UnityInterfaceBrain
import matsuoka_quad

import sys

import functools


osys = 'Linux'


# Problem definition
NOBJ = 3
P = 12
H = factorial(NOBJ + P - 1) / (factorial(P) * factorial(NOBJ - 1))
BOUND_LOW, BOUND_UP = 1.0, 10.0
##

# Algorithm parameters
MU = int(H + (4 - H % 4))
NGEN = 400
CXPB = 1.0
MUTPB = 1.0
##


# Create uniform reference point
ref_points = tools.uniform_reference_points(NOBJ, P)

# Create classes
creator.create("FitnessMin", base.Fitness, weights=(1.0,) * NOBJ)
creator.create("Individual", list, fitness=creator.FitnessMin)
##



# Toolbox initialization
def uniform(low, up, size=None):
    try:
        return [random.uniform(a, b) for a, b in zip(low, up)]
    except TypeError:
        return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]

toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, BOUND_LOW, BOUND_UP)


toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=BOUND_LOW, up=BOUND_UP, indpb=0.05)
toolbox.register("select", tools.selNSGA3, ref_points=ref_points)
##

def eval_all(pop,workers,sdev=0):
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]

    #send along an index
    invalid_tup = list(zip(range(len(invalid_ind)),invalid_ind))
    for ind in invalid_tup:
         workers.addtask((ind[0],ind[1],{'sdev':sdev}))

    #wait for completion
    workers.join()

    n = len(invalid_ind)
    print('Retrieving fitnesses')
    for i in range(n):
         newval = workers.outqueue.get()
         invalid_ind[newval[0]].fitness.values = newval[1]
         workers.outqueue.task_done()


    print('Sanity check:')
    print(invalid_ind[-1])
    print(invalid_ind[-1].fitness.values)

    return n


def main(NGEN, outfile, ncpu, n_brain, cpg, body_inds, baseperiod, bodytype, skipevery=-1, sdev=0, noise_start_gen=0, seed=None, port=9500):
    random.seed(seed)

    
    function = functools.partial(UnityInterfaceBrain.run_brain_array, n_brain, cpg, body_inds, baseperiod, bodytype, skipevery=skipevery)
    expath = UnityInterfaceBrain.getpath(osys,bodytype)    

    workers = UnityInterfaceBrain.WorkerPool(function,expath, nb_workers=ncpu, port=port)

    m = len(cpg.cons)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=n_brain*(n_brain+m)+2)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    #toolbox.register("evaluate", UnityInterface.run_from_array, 23, envlist)

    #futures = multiprocessing.Pool(6)
    #toolbox.register("map", futures.map)

    # Initialize statistics object
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean, axis=0)
    stats.register("std", numpy.std, axis=0)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    pop = toolbox.population(n=MU)

    n_evals = eval_all(pop,workers,sdev=0)

    #fitnesses = toolbox.map(toolbox.evaluate, invalid_tup)

    #for ind, fit in zip(invalid_ind, fitnesses):
    #    ind.fitness.values = fit

    # Compile statistics about the population
    record = stats.compile(pop)
    logbook.record(gen=0, evals=n_evals, **record)
    outfile.writelines(logbook.stream + '\n')

    # Begin the generational process
    for gen in range(1, NGEN):
        print('STARTING GEN',gen)
        offspring = algorithms.varAnd(pop, toolbox, CXPB, MUTPB)

        if gen < noise_start_gen:
            n_evals = eval_all(offspring,workers,sdev=0)
        else:
            n_evals = eval_all(offspring,workers,sdev=sdev)

        # Select the next generation population from parents and offspring
        pop = toolbox.select(pop + offspring, MU)

        # Compile statistics about the new population
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=n_evals, **record)
        outfile.writelines(logbook.stream + '\n')

    #close down workers
    for i in range(ncpu):
        workers.addtask(None)
    workers.join()
    workers.terminate()

    return pop, logbook


if __name__ == "__main__":
    #BEFORE RUNNING: prepare list of CPGs below and make sure body type and other parameters are correct
    #Then script can be run for each cpg in list (e.g. via bash script)
    #Usage: nsga_optimize_body.py [n_cpg] [n_cpu] [n_brain] [n_gen] [output_file]
    
    n_cpg = 23

    skipevery = 4
    sdev = 0.02 #noise in impulse timings
    noise_start_gen = 50

    bodytype='shortquad'
    timefactor = 0.0481 #below base periods are in CPG timescale. change to 1 if in seconds.

    b = []
    baseperiod = []

    """
    #RUN 1
    b.append([3, 9, 4, 1, 7, 7, 4, 4, 10, 10, 3, 10, 1, 1, 8, 2, 1, 10, 2, 10, 10, 9, 5, 7, 9, 8, 9, 9, 6, 8, 9, 1])
    baseperiod.append(5.12)
    b.append([2, 9, 4, 6, 6, 4, 1, 3, 10, 10, 6, 6, 1, 1, 8, 5, 1, 10, 2, 10, 10, 8, 5, 7, 9, 3, 7, 9, 6, 8, 9, 4])
    baseperiod.append(7.2)
    b.append([2, 9, 4, 6, 6, 4, 1, 3, 10, 10, 6, 6, 1, 1, 8, 5, 1, 10, 2, 10, 10, 9, 5, 7, 9, 8, 5, 9, 6, 8, 9, 4])
    baseperiod.append(5.76)
    b.append([1, 9, 5, 3, 1, 4, 4, 4, 9, 10, 3, 5, 1, 1, 9, 5, 4, 10, 2, 3, 10, 9, 4, 9, 10, 3, 5, 9, 5, 4, 5, 1])
    baseperiod.append(4.96)
    
    #RUN 2
    b.append([6, 5, 4, 1, 4, 2, 1, 1, 10, 8, 1, 7, 5, 8, 9, 6, 8, 6, 1, 7, 10, 6, 3, 5, 4, 5, 6, 10, 7, 8, 9, 2])
    baseperiod.append(6.72)
    b.append([7, 4, 5, 1, 8, 1, 1, 10, 10, 8, 1, 7, 4, 8, 9, 9, 8, 6, 1, 7, 10, 6, 3, 5, 4, 5, 6, 10, 7, 8, 9, 2])
    baseperiod.append(7.2)
    b.append([1, 5, 4, 6, 7, 2, 1, 1, 10, 9, 1, 7, 6, 9, 9, 6, 7, 6, 4, 6, 8, 6, 3, 5, 4, 5, 5, 10, 3, 8, 9, 2])
    baseperiod.append(10.08)
    b.append([3, 5, 4, 1, 7, 2, 1, 9, 9, 5, 3, 6, 8, 10, 1, 9, 1, 7, 1, 3, 10, 5, 4, 9, 6, 8, 1, 10, 10, 1, 1, 1])
    baseperiod.append(9.12)
    
    #RUN 3 take 1
    b.append([8, 4, 10, 2, 4, 6, 4, 9, 4, 2, 9, 4, 10, 2, 2, 8, 2, 7, 10, 6, 4, 5, 8, 2, 10, 10, 7, 8, 8, 9, 10, 1])
    baseperiod.append(8.0)
    b.append([10, 6, 10, 5, 2, 6, 5, 9, 5, 2, 9, 4, 10, 2, 2, 8, 2, 7, 10, 6, 4, 5, 8, 3, 9, 10, 8, 7, 10, 10, 10, 8])
    baseperiod.append(8.32)
    b.append([4, 3, 10, 1, 3, 6, 4, 10, 4, 2, 9, 4, 10, 2, 2, 10, 4, 7, 10, 6, 4, 5, 8, 9, 10, 10, 7, 8, 8, 9, 10, 1])
    baseperiod.append(7.84)
    b.append([2, 6, 2, 7, 2, 8, 4, 10, 4, 2, 2, 4, 10, 2, 3, 8, 4, 7, 10, 9, 2, 2, 8, 9, 5, 9, 2, 5, 8, 1, 1, 1])
    baseperiod.append(3.68)
    
    #RUN 3 take 2
    b.append([1, 10, 7, 6, 4, 2, 1, 4, 8, 10, 5, 10, 2, 2, 10, 2, 10, 7, 2, 6, 9, 10, 3, 1, 7, 6, 8, 9, 4, 9, 10, 1])
    baseperiod.append(4.48)
    b.append([1, 10, 6, 7, 1, 2, 4, 4, 8, 7, 6, 10, 1, 5, 9, 2, 3, 7, 2, 8, 9, 10, 3, 1, 7, 3, 5, 9, 10, 1, 1, 1])
    baseperiod.append(4.0)
    b.append([1, 10, 6, 1, 9, 2, 5, 4, 8, 10, 5, 9, 1, 3, 10, 2, 10, 7, 2, 6, 9, 10, 3, 1, 9, 3, 10, 9, 4, 8, 9, 1])
    baseperiod.append(4.48)

    """
    #RUN 4
    b.append([5, 9, 9, 5, 2, 4, 3, 3, 10, 10, 4, 10, 8, 10, 5, 1, 4, 10, 1, 9, 10, 7, 4, 3, 6, 1, 4, 8, 8, 8, 9, 7])
    baseperiod.append(6.08)
    b.append([6, 8, 4, 3, 2, 1, 3, 3, 8, 8, 1, 10, 7, 9, 6, 6, 3, 10, 1, 3, 10, 7, 4, 3, 2, 6, 10, 9, 4, 5, 6, 1])
    baseperiod.append(5.92)
    b.append([1, 8, 9, 5, 2, 4, 3, 3, 10, 10, 4, 10, 8, 10, 5, 6, 2, 8, 3, 2, 10, 10, 2, 5, 6, 1, 10, 8, 8, 8, 9, 4])
    baseperiod.append(4.48)
    b.append([3, 8, 4, 3, 2, 1, 3, 7, 10, 10, 4, 10, 8, 9, 5, 1, 4, 10, 1, 3, 10, 7, 4, 10, 1, 9, 10, 9, 4, 5, 6, 1])
    baseperiod.append(4.48)

    #RUN 5
    b.append([2, 5, 9, 5, 3, 3, 1, 1, 10, 10, 4, 5, 6, 8, 10, 4, 10, 10, 1, 5, 10, 8, 2, 1, 8, 7, 3, 7, 9, 8, 9, 1])
    baseperiod.append(4.32)
    b.append([2, 7, 5, 8, 3, 3, 1, 1, 10, 10, 4, 6, 6, 9, 10, 4, 1, 7, 2, 4, 8, 10, 3, 3, 6, 5, 3, 7, 8, 8, 9, 1])
    baseperiod.append(5.44)
    b.append([2, 7, 5, 5, 3, 3, 4, 4, 10, 10, 3, 3, 10, 8, 10, 10, 10, 10, 1, 4, 10, 8, 3, 7, 7, 10, 4, 7, 4, 8, 9, 2])
    baseperiod.append(4.16)
    b.append([2, 7, 5, 2, 3, 3, 4, 4, 10, 10, 3, 3, 10, 8, 10, 10, 10, 10, 1, 4, 10, 8, 3, 7, 7, 10, 4, 7, 4, 8, 9, 2])
    baseperiod.append(4.16)

    """
    #SHORT RUN 1
    b.append([4, 6, 2, 3, 9, 3, 6, 6, 9, 10, 5, 5, 10, 6, 9, 7, 9, 10, 2, 5, 9, 6, 2, 9, 9, 1, 10, 7, 10, 8, 9, 2])
    baseperiod.append(7.36)
    b.append([4, 6, 2, 3, 3, 3, 6, 5, 9, 10, 5, 4, 8, 9, 8, 3, 8, 10, 3, 4, 9, 7, 2, 9, 9, 1, 10, 7, 4, 8, 9, 2])
    baseperiod.append(4.96)
    b.append([3, 6, 2, 3, 5, 1, 9, 8, 9, 10, 2, 10, 5, 9, 4, 6, 10, 3, 3, 4, 10, 7, 4, 3, 1, 4, 10, 10, 1, 8, 9, 7])
    baseperiod.append(11.68)
    b.append([3, 7, 2, 3, 3, 3, 6, 5, 9, 10, 3, 3, 7, 8, 4, 2, 10, 10, 4, 4, 9, 6, 4, 3, 1, 10, 9, 6, 10, 4, 5, 1])
    baseperiod.append(9.6)

    
    #SHORT RUN 2
    b.append([1, 10, 8, 7, 6, 2, 2, 6, 9, 9, 5, 10, 2, 1, 10, 1, 7, 10, 1, 5, 9, 9, 3, 9, 6, 9, 9, 9, 10, 6, 8, 1])
    baseperiod.append(4.32)
    b.append([1, 7, 8, 7, 3, 2, 3, 4, 9, 9, 3, 5, 2, 8, 9, 1, 8, 7, 1, 5, 9, 10, 3, 5, 1, 2, 9, 10, 10, 6, 8, 1])
    baseperiod.append(4.96)
    b.append([1, 5, 1, 7, 8, 4, 8, 4, 10, 9, 1, 7, 3, 10, 2, 1, 8, 8, 1, 2, 8, 8, 3, 9, 6, 7, 3, 9, 3, 7, 8, 5])
    baseperiod.append(6.72)
    b.append([1, 2, 8, 8, 8, 1, 5, 4, 10, 9, 1, 8, 1, 8, 3, 1, 8, 7, 4, 5, 9, 9, 3, 9, 6, 9, 9, 9, 10, 6, 8, 1])
    baseperiod.append(6.88)
    
    
 
    #SHORT RUN 3
    b.append([1, 10, 3, 2, 8, 5, 5, 3, 10, 10, 5, 10, 8, 3, 9, 6, 4, 9, 1, 2, 8, 9, 2, 9, 10, 7, 6, 2, 10, 8, 10, 2])
    baseperiod.append(4.32)
    b.append([1, 10, 4, 2, 8, 5, 5, 8, 10, 8, 5, 10, 8, 3, 9, 6, 4, 9, 1, 2, 8, 9, 3, 9, 10, 7, 6, 8, 10, 8, 10, 2])
    baseperiod.append(4.16)
    b.append([10, 10, 5, 2, 3, 1, 5, 7, 10, 10, 9, 10, 8, 3, 10, 7, 5, 6, 1, 5, 8, 8, 5, 2, 5, 7, 4, 10, 1, 9, 9, 2])
    baseperiod.append(8.48)
    b.append([1, 10, 1, 2, 8, 5, 5, 3, 10, 10, 5, 9, 8, 8, 9, 6, 4, 9, 1, 2, 8, 8, 5, 9, 10, 7, 2, 10, 2, 1, 1, 2])
    baseperiod.append(8.16)

    
    #SHORT RUN 4
    b.append([2, 7, 4, 10, 1, 9, 9, 2, 6, 3, 5, 10, 4, 8, 10, 4, 9, 5, 1, 1, 10, 9, 1, 6, 2, 8, 1, 10, 10, 8, 9, 1])
    baseperiod.append(4.64)
    b.append([2, 7, 4, 2, 10, 10, 9, 6, 8, 2, 1, 10, 4, 4, 8, 6, 6, 10, 1, 7, 10, 9, 1, 6, 2, 8, 1, 10, 4, 8, 9, 2])
    baseperiod.append(6.72)
    b.append([2, 3, 4, 2, 6, 5, 10, 4, 8, 10, 2, 10, 4, 6, 8, 4, 6, 10, 7, 8, 10, 7, 1, 7, 9, 5, 3, 10, 1, 8, 9, 5])
    baseperiod.append(4.64)

    #SHORT RUN 5
    b.append([1, 8, 8, 3, 1, 9, 10, 5, 5, 2, 10, 9, 8, 1, 4, 9, 8, 2, 9, 10, 2, 3, 9, 8, 10, 8, 9, 10, 6, 9, 10, 3])
    baseperiod.append(4.16)
    b.append([1, 8, 9, 9, 8, 10, 10, 5, 5, 1, 8, 2, 4, 5, 3, 9, 8, 2, 9, 10, 2, 3, 9, 4, 1, 8, 4, 9, 1, 3, 4, 1])
    baseperiod.append(4.8)
    b.append([1, 3, 8, 3, 3, 9, 8, 4, 6, 2, 10, 10, 8, 1, 1, 8, 8, 2, 9, 10, 2, 3, 10, 1, 5, 5, 9, 9, 10, 9, 10, 1])
    baseperiod.append(4.0)
    b.append([1, 8, 1, 8, 8, 6, 1, 2, 10, 4, 9, 1, 5, 1, 3, 2, 3, 2, 9, 4, 2, 3, 9, 4, 10, 10, 9, 9, 8, 1, 1, 1])
    baseperiod.append(9.28)
    
    """



    

    if len(sys.argv) > 5:
        cpgnum = int(sys.argv[1])
        ncpu = int(sys.argv[2])
        n_brain = int(sys.argv[3])
        ngen = int(sys.argv[4])
        outpath = sys.argv[5]
        
        port = 9200 + 100*cpgnum
        bodyarray = b[cpgnum]
        cpg = matsuoka_quad.array2param(bodyarray[:n_cpg])
        body_inds = bodyarray[n_cpg:]
        with open(outpath,'w') as outfile:
           outfile.writelines('Matsuoka quadruped, body parameters:')
           outfile.writelines(str(bodyarray) + '\n')
           outfile.writelines(str(cpg.param) + '\n')
           outfile.writelines('Inter-module:\n' + str(cpg.adj) + '\n')
           outfile.writelines('Intra-module:\n' + str(cpg.cons[0].w) + '\n')
           outfile.writelines('Bias:' + str(cpg.cons[0].b) + '\n')
           outfile.writelines('Drive coeff:' + str(cpg.cons[0].d) + '\n')
           outfile.writelines('Base period:' + str(baseperiod[cpgnum]) + '\n')

           pop, stats = main(ngen, outfile, ncpu, n_brain, cpg, body_inds, baseperiod[cpgnum]*timefactor, bodytype, skipevery=skipevery, sdev=sdev,noise_start_gen=noise_start_gen,port=port)
           pop_fit = numpy.array([ind.fitness.values for ind in pop])

           mean_fit = numpy.array([numpy.mean(numpy.array(ind.fitness.values)) for ind in pop])
           inds = numpy.argsort(mean_fit)

           for ind in inds:
               outfile.writelines([str(pop[ind]) + '\n'])
               outfile.writelines([str(pop[ind].fitness.values) + '\n'])

        outfile.close()


    else:
        print('Usage: nsga_optimize_body.py [n_cpg] [n_cpu] [n_brain] [n_gen] [output_file]')


