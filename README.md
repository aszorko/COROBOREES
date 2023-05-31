# COROBOREES
***Collective robotics through real-time entrainment of evolved dynamical systems***

The master branch is updated continuously. Branches are made for reproduction of paper-specific data.

## Installing:
Clone the repository. To enable the Unity simulations (currently Linux only), create a Python virtual environment using CPG/requirements.txt.

## Paper1: Rapid rhythmic entrainment in bio-inspired central pattern generators

This is a Python-only implementation that evolves disembodied CPGs directly for flexible periods. Filters are evolved for period matching.

https://arxiv.org/abs/2206.01638

Appears in Proceedings of the IEEE International Joint Conference on Neural Networks 2022

## Paper2: Central pattern generators evolved for real-time adaptation

This is a Python+Unity implementation. CPGs are evolved for stable backwards+forward+accelerating motion as control parameters are swept. Filters are evolved for period matching.

https://arxiv.org/abs/2210.08102

*(submitted)*

## Paper3: Embodied tempo tracking with a virtual quadruped

This is a Python+Unity implementation. Short-legged quadruped agents from Paper2 are tested with a wide range of pulse inputs and musical excerpts.

To appear in Proceedings of the Sound and Music Conference 2023

## Paper4: From real-time adaptation to social learning in robot ecosystems

This is a Python+Unity implementation. Short-legged quadruped agents from Paper2 and newly evolved hexapods are run in teacher-learner pairs. A quasi-Hebbian process is used to turn synchronized gaits into autonomous gaits.
