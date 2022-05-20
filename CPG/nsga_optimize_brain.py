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

import multiprocessing
import matsuoka_brain
import matsuoka_quad

import sys

N_PROCESSES = 12

#NSGA3 parameters
NOBJ = 3
P = 10
H = factorial(NOBJ + P - 1) / (factorial(P) * factorial(NOBJ - 1))
BOUND_LOW, BOUND_UP = 0.0, 10.0

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


def main(NGEN,n,body,baseperiod,seed=None,skipevery=-1):
    random.seed(seed)

    m = len(body.cons)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=n*(n+m)+2)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", matsuoka_brain.run_from_array,n,body,baseperiod,skipevery=skipevery)

    futures = multiprocessing.Pool(N_PROCESSES)
    toolbox.register("map", futures.map)
    
    # Initialize statistics object
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean, axis=0)
    stats.register("std", numpy.std, axis=0)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    pop = toolbox.population(n=MU)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # Compile statistics about the population
    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)
    print(logbook.stream)

    # Begin the generational process
    for gen in range(1, NGEN):
        offspring = algorithms.varAnd(pop, toolbox, CXPB, MUTPB)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Select the next generation population from parents and offspring
        pop = toolbox.select(pop + offspring, MU)

        # Compile statistics about the new population
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        print(logbook.stream)

    return pop, logbook


if __name__ == "__main__":
    bodyarray = [4, 2, 2, 10, 3, 5, 3, 10, 8, 10, 10, 3, 2, 5, 9, 5, 2, 10, 9, 7, 1, 5, 8]
    baseperiod = 180.0
    skipevery = 4
    body = matsuoka_quad.array2param(bodyarray)
    
    print('Matsuoka quadruped, body parameters:')
    print(bodyarray)
    print(body.param)
    print('Inter-module:\n', body.adj)
    print('Intra-module:\n', body.cons[0].w)
    print('Bias:', body.cons[0].b)
    print('Drive coeff:', body.cons[0].d)

    if len(sys.argv) > 1:
        n = int(sys.argv[1])
        ngen = int(sys.argv[2])
    else:
        n = 4
        ngen = 2
    pop, stats = main(ngen,n,body,baseperiod,skipevery=skipevery)
    pop_fit = numpy.array([ind.fitness.values for ind in pop])
    
    mean_fit = numpy.array([numpy.mean(numpy.array(ind.fitness.values)) for ind in pop])

    bestfit = numpy.max(mean_fit)
    bestind = numpy.argmax(mean_fit)
    
    print(pop[bestind])
    print(pop[bestind].fitness.values)
    print(bestfit)
