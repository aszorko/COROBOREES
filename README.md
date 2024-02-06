# COROBOREES
***Collective robotics through real-time entrainment of evolved dynamical systems***

The master branch is updated continuously. Branches are made for reproduction of paper-specific data.

## Installing:
Clone the repository. To enable the Unity simulations (currently Linux only), create a Python virtual environment using CPG/requirements.txt.

## Paper1: Rapid rhythmic entrainment in bio-inspired central pattern generators

This is a Python-only implementation that evolves disembodied CPGs directly for flexible periods. Filters are evolved for period matching.

Preprint: https://arxiv.org/abs/2206.01638

Szorkovszky A, Veenstra F, and Glette K (2022) in *2022 International Joint Conference on Neural Networks (IJCNN)*

## Paper2: Central pattern generators evolved for real-time adaptation to rhythmic stimuli

This is a Python+Unity implementation. CPGs are evolved for stable backwards+forward+accelerating motion as control parameters are swept. Filters are evolved for period matching.

Szorkovszky A, Veenstra F and Glette K (2023) *Bioinspiration and Biomimetics* 18:046020

https://doi.org/10.1088/1748-3190/ace017

## Paper3: Embodied tempo tracking with a virtual quadruped

This is a Python+Unity implementation. Short-legged quadruped agents from Paper2 are tested with a wide range of pulse inputs and musical excerpts.

Szorkovszky A, Veenstra F, Lartillot O, Jensenius AR and Glette K (2023), in *Proceedings of the Sound and Music Conference 2023*, pp 283-288

https://zenodo.org/doi/10.5281/zenodo.10060970

## Paper4: From real-time adaptation to social learning in robot ecosystems

This is a Python+Unity implementation. Short-legged quadruped agents from Paper2 and newly evolved hexapods are run in teacher-learner pairs. A quasi-Hebbian process is used to turn synchronized gaits into autonomous gaits.

Szorkovszky A, Veenstra F and Glette K (2023). *Frontiers in Robotics and AI*, 10:1232708.

https://www.frontiersin.org/articles/10.3389/frobt.2023.1232708/full
