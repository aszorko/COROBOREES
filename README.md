# Central pattern generators evolved for real-time adaptation to rhythmic stimuli

[![Youtube: example of entrainment in simulation](https://img.youtube.com/vi/k9cmEjPEbkM/0.jpg)](https://www.youtube.com/watch?v=k9cmEjPEbkM)

(link to Youtube video)

## Running Unity simulations (currently Linux only):

1. Clone branch and unzip files in `CPG/Unity` directory (in this or another location)
2. Create a python virtual environment using `CPG/requirements.txt` and activate
3. Set the Linux paths in the `getpaths()` function of `UnityInterfaceBrain.py`

After activating the virtual environment, running

`$python CPG/UnityInterfaceBrain.py`

directly will run a graphical simulation of either the CPG evaluation or the CPG+filter entrainment evaluation. Simulation type, CPGs and filters can be set in the `__main__` function.


## Evolution

Evolutions use multiprocessing and are run from the command line (e.g. via Slurm). See code for arguments.

`CPG/nsga_optimize_body.py`: CPG evolution

`CPG/nsga_optimize_unitybrain.py`: Filter evolution. Uses CPGs contained in the main function.

## Post-evolution steps

The final generations are then run for more iterations and more periods in the case of the filter. These use multiprocessing and are run from the command line. See code for arguments.

`CPG/final_eval_body.py`: final CPG evaluation

`CPG/final_eval_unitybrain.py`: final filter evolution

## Other scripts

`CPG/evoplot.py`: functions to analyse evolution output and select CPGs from the Pareto front

`CPG/getperiods.py`: iterates over control parameters and extracts periods, correlations and other measures

## Data

Data is contained in the `CPG/paper2_data/` directory.

`$python CPG/paper2_results1.py`
will run analyses on these data and produce Figures 2-5 and statistics.

`$python CPG/paper2_results2.py`
generates Figure 6 using the Unity interface (see below).

Note there are three versions of the final entrainment data:

`*final.txt`: Same periods as in evolution, combined fitness.

`*final2.txt`: Same periods as in evolution, fitness separated into period matching and height.

`*final3.txt`: Two additional periods, fitness separated into period matching and height.
