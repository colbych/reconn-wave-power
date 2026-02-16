Standalone Scripts for Batch Lagrangian Analysis
=================================================

These scripts are standalone versions of notebooks 06 and 07, designed for
running on remote or headless machines. Both use non-interactive matplotlib
rendering and accept a single argument: the path to a JSON config file.

Prerequisites: the reconn-wave-power package must be installed (pip install -e .)
and the conda environment activated (conda activate reconn-wave-power).


Script 1: run_batch_analysis.py
-------------------------------

Traces E x B drift trajectories from a grid of starting positions, samples all
6 field components (Ex, Ey, Ez, Bx, By, Bz) along each trajectory, computes
frequency PSDs, and saves everything to a NetCDF file.

Usage:
    python scripts/run_batch_analysis.py my_batch_config.json

Outputs:
    <save_name>.nc                      -- NetCDF with trajectories, sampled fields, and PSDs
    <save_name>_starting_positions.png  -- Plot of starting positions over Bz snapshot
    <save_name>_trajectories.png        -- Plot of all traced trajectories

Config keys:

    input_file     (required)  Path to the dHybridR input file
    output_folder  (required)  Path to the dHybridR Output directory
    save_dir       (optional)  Directory for output files. Default: "."
    save_name      (optional)  Base name for output files. Default: "lagrangian_batch"
    field          (optional)  Field for background snapshot plots. Default: "Bz"
    t0_idx         (optional)  Starting time index for trajectory tracing. Default: 0
    psd_method     (optional)  PSD method: "fft" or "welch". Default: "fft"

Starting positions -- three modes:

  Mode 1: Evenly-spaced grid (default)
    Uses n_lines vertical lines with n_per_line points each, spread evenly
    across the simulation domain.

        "n_lines": 3,
        "n_per_line": 20

  Mode 2: Explicit list of [x, y] pairs
    Specify arbitrary points directly. Values are in simulation units (d_i).

        "positions": [
            [100.0, 50.0],
            [200.0, 75.0],
            [300.0, 100.0]
        ]

  Mode 3: Range-based grid
    Specify x and y arrays using range-like syntax. Every x is paired with
    every y (outer product), so 5 x-values and 20 y-values gives 100 points.

        "positions": {
            "x": [100, 600, 100],
            "y": [0, 200, 21, "linspace"]
        }

    Array formats:
        [start, stop, step]           -- np.arange (stop is exclusive)
        [start, stop, n, "linspace"]  -- np.linspace (stop is inclusive)

    All values are in simulation units (d_i).

  When "positions" is present, "n_lines" and "n_per_line" are ignored.

Minimal example config (Mode 1):

    {
        "input_file": "/data/sim01/input",
        "output_folder": "/data/sim01/Output"
    }

Range-based example config (Mode 3):

    {
        "input_file": "/data/sim01/input",
        "output_folder": "/data/sim01/Output",
        "save_dir": "results",
        "save_name": "my_batch",
        "positions": {
            "x": [100, 600, 100],
            "y": [0, 200, 21, "linspace"]
        }
    }


Script 2: run_truncated_psds.py
-------------------------------

Loads the NetCDF output from Script 1, optionally truncates trajectories at
user-specified end times, optionally generates per-trajectory MP4 movies
(Bz field + magnetic field line contours + particle track), computes PSDs for
all 6 field components plus 5 derived quantities (Sx, Sy, Sz, uE, uB), and
generates 3-panel PSD plots.

Usage:
    python scripts/run_truncated_psds.py my_psd_config.json

Outputs:
    <save_name>.nc           -- NetCDF with truncated PSDs
    <psd_dir>/psd_*.png      -- Per-trajectory 3-panel PSD plots (E, B, Poynting/energy)
    <movie_dir>/*.mp4        -- Per-trajectory movies (if make_movies is true)

Config keys:

    input_file     (required)  Path to the dHybridR input file (same as Script 1)
    output_folder  (required)  Path to the dHybridR Output directory
    batch_file     (required)  Path to the NetCDF from Script 1
    end_times_file (optional)  Path to a JSON file mapping trajectory index (string)
                               to raw end time (number). Trajectories are truncated at
                               time_index = raw_value // 2 + 1. Default: null (no truncation)
    save_dir       (optional)  Directory for output NetCDF. Default: "."
    save_name      (optional)  Base name for output NetCDF. Default: "lagrangian_batch_truncated"
    psd_dir        (optional)  Directory for PSD plot PNGs. Default: "psd_plots"
    psd_method     (optional)  PSD method: "fft" or "welch". Default: "fft"
    make_movies    (optional)  Whether to generate MP4 movies. Default: false
    movie_dir      (optional)  Directory for movie files. Default: "movies"
    frame_stride   (optional)  Use every Nth frame in movies. Default: 10
    tail_len       (optional)  Number of past frames to show as trajectory tail. Default: 20
    movie_fps      (optional)  Movie frames per second. Default: 20
    movie_dpi      (optional)  Movie resolution. Default: 150
    n_contours     (optional)  Number of magnetic field line contours. Default: 20
    bz_min         (optional)  Bz color scale minimum. Default: null (-0.5)
    bz_max         (optional)  Bz color scale maximum. Default: null (+0.5)

Minimal example config (no truncation, no movies):

    {
        "input_file": "/data/sim01/input",
        "output_folder": "/data/sim01/Output",
        "batch_file": "results/my_batch.nc"
    }

Full example config:

    {
        "input_file": "/data/sim01/input",
        "output_folder": "/data/sim01/Output",
        "batch_file": "results/my_batch.nc",
        "end_times_file": "end_times.json",
        "save_dir": "results",
        "save_name": "my_batch_truncated",
        "psd_dir": "results/psd_plots",
        "psd_method": "fft",
        "make_movies": true,
        "movie_dir": "results/movies",
        "frame_stride": 5,
        "tail_len": 30,
        "movie_fps": 15,
        "movie_dpi": 200,
        "n_contours": 25,
        "bz_min": -0.3,
        "bz_max": 0.3
    }


Typical Workflow
----------------

1. Create a config for Script 1 and run it:

       python scripts/run_batch_analysis.py batch_config.json

2. Inspect the starting positions and trajectory plots.

3. (Optional) Identify trajectories that leave the region of interest and
   create an end_times.json file to truncate them.

4. Create a config for Script 2 pointing to the batch NetCDF, and run it:

       python scripts/run_truncated_psds.py psd_config.json

5. Review the PSD plots in the psd_dir.
