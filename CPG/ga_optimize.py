# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 15:25:56 2021

@author: alexansz
"""

import random
from scoop import futures
import matsuoka_quad
from deap import creator, base, tools, algorithms
import sys

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

toolbox.register("evaluate", matsuoka_quad.run_from_array)
toolbox.register("attr_bool", random.randint, 1, 10)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=23)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    #def evalOneMax(individual):
    #    return sum(individual),
    
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=1, up=10, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("map", futures.map)

def main(NGEN):       
    population = toolbox.population(n=72)
    
    for gen in range(NGEN):
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.75, mutpb=0.4)
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        print(tools.selBest(offspring, k=1))
        print(max([ind.fitness for ind in offspring]))
        #population = toolbox.select(offspring, k=len(population))
        p = tools.selBest(offspring, k=36) #take best 50% and double
        population = []
        population.extend(p)
        population.extend(p) 
    top10 = tools.selBest(offspring, k=10)
    #good abs-period optimised score ~0.77
    print(top10)
    print(max([ind.fitness for ind in offspring]))


if __name__ == '__main__':
    if len(sys.argv) > 1:
        NGEN = int(sys.argv[1])
    else:
        NGEN = 50
    main(NGEN)
