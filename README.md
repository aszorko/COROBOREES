# Rapid rhythmic entrainment in bio-inspired central pattern generators

Guide to files (see comments inside for details):

`$python CPG/nsga_optimize_cpg.py [# generations] >[output file]`

This generates a population of CPGs using the NSGA3 algorithm. The variable P determines the population size (P=8 and NOBJ=3 -> 48 individuals) and N_PROCESSES specifies the number of processes for multiprocessing.

`$python CPG/final_eval_cpg.py >[output file]`

Runs the final population *n* times. The output of the CPG evolution must be specified in the file. Uses multiprocessing.

`$python CPG/nsga_optimize_brain.py [# filter neurons] [# generations] >[output file]`

A CPG array is specified in the main function, and a base stimulus period. Population size and number of processes are set similarly to the CPG evolution. 

`$python CPG/paper1_results1.py`

Produces Figures 2-3.

`$python CPG/paper1_results2.py`

Produces Figures 4-6.

**Requires (version used):**
numpy (1.19.2), scipy (1.6.2), matplotlib (3.3.4), deap (1.3.1)
