# Central pattern generators evolved for real-time adaptation

Data is contained in the CPG/paper2_data/ directory.

`$python CPG/paper2_results1.py`
will run analyses on these data and produce Figures 2-5 and statistics.

`$python CPG/paper2_results2.py`
generates Figure 6 using the Unity interface (see below).

Instructions for running Unity software (currently Linux only):

1. Clone branch and unzip files in `/CPG/Unity` directory (in this or another location)
2. Create a python virtual environment using `requirements.txt` and activate
3. Set the Linux paths in the `getpaths()` function of `UnityInterfaceBrain.py`

After activating the virtual environment, running

`$python CPG/UnityInterfaceBrain.py`

directly will run a graphical simulation of either the CPG evaluation or the CPG+filter entrainment evaluation. Simulation type, CPGs and filters can be set in the `__main__` function.
