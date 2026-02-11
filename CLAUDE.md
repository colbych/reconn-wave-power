# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python package for analyzing wave power in hybrid (particle-ion, fluid-electron) reconnection simulations. Reads raw simulation outputs via dhybridrpy, converts them to xarray Datasets, computes time- and space-domain power spectra (Eulerian and Lagrangian), and traces particle trajectories through E×B drift fields.

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
python -m pytest tests/test_lagrangian.py
python -m pytest tests/test_processing.py
```

## Conda Environment

```bash
# Activate the project environment
conda activate reconn-wave-power

# Or create it if it doesn't exist
conda create -n reconn-wave-power python=3.11 numpy scipy xarray matplotlib pandas pytest dhybridrpy -y
conda activate reconn-wave-power
pip install -e .
```

## Architecture

The package (`src/reconn_wave_power/`) has five modules:

- **io.py**: Wraps dhybridrpy to load simulation data as xarray Datasets. Main entry point is `read_simulation()` which accepts flexible inputs (dhybridrpy instance, file paths, or tuples). Extracts metadata (dt, dx, dy, dz) from simulation inputs. Supports lazy loading via dask and optional tqdm progress bars.

- **spectrum.py**: FFT and Welch-based spectral analysis. Key functions:
  - `compute_psd_time()` — Time-domain PSD using Welch's method or windowed FFT
  - `compute_psd_space()` — Spatial PSD using FFT
  - `compute_komega_2d()` — 2D k-omega spectra

- **lagrangian.py**: Lagrangian-frame frequency spectra via E×B drift trajectories. Key functions:
  - `compute_exb_velocity()` — In-plane E×B drift velocity from E and B fields
  - `trace_trajectory()` / `trace_trajectories()` — Single and batch trajectory integration with periodic boundary conditions
  - `sample_along_trajectory()` / `sample_along_trajectories()` — Bilinear field interpolation along trajectories with dask-aware batch preloading
  - `lagrangian_psd()` / `lagrangian_psds()` — Convenience wrappers: trace → sample → PSD

  The batch functions (`trace_trajectories`, `sample_along_trajectories`) are vectorized and significantly faster than looping over the single-trajectory versions. Dask arrays are loaded in 64-timestep chunks to manage memory.

- **processing.py**: Preprocessing utilities:
  - `detrend_inplace()` — Detrend along time axis
  - `compute_flux_function()` — Magnetic flux function ψ = ∫ Bx dy via cumulative trapezoidal integration (for field line visualization)

- **plotting.py**: Matplotlib helpers for PSD visualization (`plot_psd()`)

## Notebooks

Located in `notebooks/`:

- **01_exploration.ipynb** — Initial data exploration
- **02_alfven_wave_validation.ipynb** — Validates FFT methods with synthetic Alfvén wave
- **03_komega_slices.ipynb** — k-ω spectra at selected spatial slices
- **04_lagrangian_spectra.ipynb** — Lagrangian vs Eulerian frequency PSDs
- **05_lagrangian_alfven_validation.ipynb** — End-to-end validation of Lagrangian tracer with synthetic data
- **06_lagrangian_batch_analysis.ipynb** — Batch trajectory tracing on a grid of starting positions, saves results to NetCDF
- **07_lagrangian_movies_and_psds.ipynb** — Per-trajectory movies (Bz + field line contours + particle track) and truncated PSDs

## Key Conventions

- **xarray-first design**: All data flows through xarray.Dataset/DataArray with standardized dimension names ("time", "x", "y", "z")
- **Type hints**: All public functions have type annotations
- **Field specifications**: Support both group names ("E", "B") and explicit components ("Ex", "Ey", "Ez", "Bx", "By", "Bz")
- **Testing strategy**: Tests mock dhybridrpy (inject fake module into sys.modules) so no real simulation files are needed. Performance benchmarks validate batch speedups. Synthetic Alfvén wave tests validate spectral recovery.
- **Periodic boundaries**: Trajectory tracing wraps at domain edges

## Dependencies

Core: numpy, scipy, xarray, matplotlib, pandas, dhybridrpy
Optional: dask (lazy array loading), tqdm (progress bars)
