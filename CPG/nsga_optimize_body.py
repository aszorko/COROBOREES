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
#from deap.benchmarks.tools import igd
from deap import creator
from deap import tools

import UnityInterfaceBrain

import sys

import functools



osys = 'Linux'

# Problem definition
NOBJ = 4 # number of objective functions
P = 8    # points per side of reference plane. determines population size
H = factorial(NOBJ + P - 1) / (factorial(P) * factorial(NOBJ - 1))
BOUND_LOW, BOUND_UP = 1.0, 10.0
##

# Algorithm parameters
MU = int(H + (4 - H % 4))
NGEN = 400
CXPB = 0.7 
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

def eval_all(pop,workers):
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]

    #send along an index
    invalid_tup = list(zip(range(len(invalid_ind)),invalid_ind))
    for ind in invalid_tup:
         workers.addtask(ind)

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


def main(NGEN,outfile,ncpu,bodytype,port=9600,seed=None):
    random.seed(seed)
    
    function = functools.partial(UnityInterfaceBrain.run_from_array,23,bodytype)
    expath = UnityInterfaceBrain.getpath(osys,bodytype)    
    
    workers = UnityInterfaceBrain.WorkerPool(function,expath,port=port,nb_workers=ncpu)

    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=32)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    
    # Initialize statistics object
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean, axis=0)
    stats.register("std", numpy.std, axis=0)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    pop = toolbox.population(n=MU)
    
    n_evals = eval_all(pop,workers)


    # Compile statistics about the population
    record = stats.compile(pop)
    logbook.record(gen=0, evals=n_evals, **record)
    outfile.writelines(logbook.stream + '\n')

    # Begin the generational process
    for gen in range(1, NGEN):
        print('STARTING GEN',gen)
        offspring = algorithms.varAnd(pop, toolbox, CXPB, MUTPB)

        n_evals = eval_all(offspring,workers)

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
    if len(sys.argv) > 5:
        ncpu = int(sys.argv[1])
        port = int(sys.argv[2])
        ngen = int(sys.argv[3])
        bodytype = sys.argv[4]
        outpath = sys.argv[5]
        with open(outpath,'w') as outfile:
           pop, stats = main(ngen,outfile,ncpu,bodytype,port=port)
           pop_fit = numpy.array([ind.fitness.values for ind in pop])
        
           mean_fit = numpy.array([numpy.mean(numpy.array(ind.fitness.values)) for ind in pop])
           unsort_fit = numpy.array([numpy.array(ind.fitness.values[3]) for ind in pop])
           inds = numpy.argsort(unsort_fit)
    
           for ind in inds:
               outfile.writelines([str(pop[ind]) + '\n'])
               outfile.writelines([str(pop[ind].fitness.values) + '\n'])
        
        outfile.close()
        
        
    else:
        print('Usage: nsga_optimize_body.py [n_cpu] [base_port] [n_gen] [bodytype] [output_file]')
        
    
