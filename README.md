# Embodied Tempo Tracking with a Virtual Quadruped


[![Youtube: example of tempo tracking in simulation](https://img.youtube.com/vi/k9cmEjPEbkM/0.jpg)](https://www.youtube.com/watch?v=k9cmEjPEbkM)

(link to Youtube video)

## Unity executables (Linux only)

Download and unpack Unity_short.zip and Unity_int.zip to the CPG folder. Make sure the Linux executable paths in UnityInterfaceBrain.py are correct.

## Evaluation

These scripts use multiprocessing and are run from the command line in the virtual environment (e.g. via Slurm, see master branch for info). See code for arguments.

`CPG/run_impulse_entrainment.py`: test entrainment to various pulses

`CPG/run_audio_entrainment.py`: test entrainment to audio clips

## Data

Data is contained in the `CPG/paper3_data/` directory.

`$python CPG/paper3_results1.py`
will run analyses on these data and produce Figures 2&4 and statistics.

`$python CPG/paper3_results2.py`
generates Figure 3 using the Unity interface. Must be run from command line in the virtual environment.

