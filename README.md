# Embodied Tempo Tracking with a Virtual Quadruped

[![Example of tempo tracking in simulation](https://img.youtube.com/vi/k9cmEjPEbkM/0.jpg)](https://www.youtube.com/watch?v=k9cmEjPEbkM)


## Interactive unity interface *to be uploaded*


## Evaluation

These use multiprocessing and are run from the command line (e.g. via Slurm). See code for arguments.

`CPG/run_impulse_entrainment.py`: test entrainment to various pulses

`CPG/run_audio_entrainment.py`: test entrainment to audio clips

## Data

Data is contained in the `CPG/paper3_data/` directory.

`$python CPG/paper3_results1.py`
will run analyses on these data and produce Figures 2&4 and statistics.

`$python CPG/paper3_results2.py`
generates Figure 3 using the Unity interface (see below).

