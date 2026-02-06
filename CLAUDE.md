# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python package for computing time- and space-domain power spectra from hybrid simulation electric and magnetic field data. Uses dhybridrpy to read raw simulation outputs and converts them to xarray Datasets for analysis.

## Build & Development Commands

```bash
# Install in development mode
python -m pip install -e .

# Install with optional dask support
python -m pip install -e ".[dask]"

# Run all tests (with conda env active)
python -m pytest tests/

# Run a specific test file
python -m pytest tests/test_io.py
python -m pytest tests/test_spectrum.py
```

## Conda Environment

```bash
# Activate the project environment
conda activate reconn-wave-power

# Or create it if it doesn't exist
conda create -n reconn-wave-power python=3.11 numpy scipy xarray matplotlib pandas pytest -y
conda activate reconn-wave-power
pip install git+https://github.com/bwostler/dhybridrpy.git
pip install -e .
```

## Architecture

The package (`src/reconn_wave_power/`) has four modules:

- **io.py**: Wraps dhybridrpy to load simulation data as xarray Datasets. Main entry point is `read_simulation()` which accepts flexible inputs (dhybridrpy instance, file paths, or tuples). Extracts metadata (dt, dx, dy, dz) from simulation inputs.

- **spectrum.py**: FFT and Welch-based spectral analysis. Key functions:
  - `compute_psd_time()` - Time-domain PSD using Welch's method
  - `compute_psd_space()` - Spatial PSD using FFT
  - `compute_komega_2d()` - 2D k-omega spectra

- **processing.py**: Preprocessing utilities (currently `detrend_inplace()`)

- **plotting.py**: Matplotlib helpers for PSD visualization

## Key Conventions

- **xarray-first design**: All data flows through xarray.Dataset/DataArray with standardized dimension names ("time", "x", "y", "z")
- **Type hints**: All public functions have type annotations
- **Field specifications**: Support both group names ("E", "B") and explicit components ("Ex", "Ey", "Ez", "Bx", "By", "Bz")
- **Testing strategy**: Tests mock dhybridrpy (inject fake module into sys.modules) so no real simulation files are needed

## Dependencies

Core: numpy, scipy, xarray, matplotlib, pandas, dhybridrpy (from git+https://github.com/bwostler/dhybridrpy.git)
