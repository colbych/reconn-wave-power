# Reconn Wave Power

Python package to compute wave power spectra and trace Lagrangian trajectories from hybrid simulation (particle-ion, fluid-electron) electric and magnetic field data. Uses [`dhybridrpy`](https://github.com/bwostler/dhybridrpy) for reading raw simulation outputs and represents all data as xarray Datasets.

## Features

- **Simulation I/O** — Load hybrid simulation outputs into xarray Datasets with lazy (dask) or eager loading
- **Spectral analysis** — Time-domain PSDs (Welch or FFT), spatial PSDs, and 2D k-ω spectra
- **Lagrangian trajectories** — Trace E×B drift trajectories with periodic boundary conditions; sample fields along trajectories and compute co-moving frame PSDs
- **Preprocessing** — Detrending, magnetic flux function computation (ψ = ∫ Bx dy) for field line visualization
- **Visualization** — PSD plots, trajectory movies with Bz color maps and field line contour overlays

## Installation

```bash
# Create conda environment
conda create -n reconn-wave-power python=3.11 numpy scipy xarray matplotlib pandas pytest dhybridrpy -y
conda activate reconn-wave-power

# Install in development mode
pip install -e .

# Or with optional dask support for lazy loading
pip install -e ".[dask]"
```

## Structure

```
src/reconn_wave_power/   # Package source
  io.py                  # Simulation data → xarray Datasets
  spectrum.py            # FFT / Welch spectral analysis
  lagrangian.py          # E×B trajectory tracing and Lagrangian PSDs
  processing.py          # Detrending, flux function computation
  plotting.py            # Matplotlib helpers
notebooks/               # Analysis notebooks (exploration → movies)
tests/                   # Unit and integration tests
paper/                   # LaTeX draft and figures
examples/                # Usage examples
```

## Quick Start

```python
from reconn_wave_power.io import read_simulation
from reconn_wave_power.spectrum import compute_psd_time

# Load simulation magnetic field data
ds = read_simulation(input_file="path/to/input", output_folder="path/to/Output", fields=("B",))

# Compute time-domain PSD at a spatial location
bz = ds["Bz"].sel(x=100, y=50, method="nearest")
f, Pxx = compute_psd_time(bz, dt=ds.attrs["dt"])
```

## Tests

```bash
python -m pytest tests/
```

## Dependencies

**Core:** numpy, scipy, xarray, matplotlib, pandas, dhybridrpy
**Optional:** dask (lazy array loading), tqdm (progress bars)
