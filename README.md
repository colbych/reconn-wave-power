# Reconn Wave Power

Python package to compute time- and space-domain power spectra for electric and magnetic fields output by hybrid simulations (uses `bwostler/dhybridrpy` for reading raw simulation data).

Goals:
- read simulation outputs and represent them as xarray Datasets
- compute PSDs (time, space, and k-omega) with configurable parameters
- reproducible figure generation pipeline and LaTeX draft for the paper

Structure:
- src/reconn_wave_power/: package code
- examples/: scripts and notebooks
- paper/: LaTeX draft and generated figures
- tests/: unit tests

See examples/analysis_example.py for usage.