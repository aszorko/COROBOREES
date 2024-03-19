# From real-time adaptation to social learning in robot ecosystems


[![Youtube: example of social learning in simulation](https://img.youtube.com/vi/rARNARMTKqo/0.jpg)](https://www.youtube.com/watch?v=rARNARMTKqo)


## Unity executables (Linux only)

Download and unzip Unity.zip (contains auxilary files) and UnityHebb.zip (contains quadruped and hexapod executables with foot sensors).

## Evaluation

The evaluation script `CPG/run_hebb.py` uses multiprocessing and is run from the command line in the virtual environment (e.g. via Slurm, see master branch for info). No arguments needed.


## Data

Data is contained in the `CPG/paper4_data/` directory.

`$python CPG/paper4_results1.py`
will run analyses on these data and produce Figures 2,4,5,6b and statistics.

`$python CPG/paper4_results2.py`
generates Figure 3 and Figure 6a using the Unity interface. Must be run from command line in the virtual environment.
