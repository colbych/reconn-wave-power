"""
Subregion PSD — local save with descriptive filename

Edit the USER SETTINGS block below, then run:
    python scripts/subregion_psd_local.py

Output is saved to SAVE_DIR as an HDF5 file whose name encodes the
subregion bounds, timestep range, and PSD method so it is self-describing,
e.g.:
    psd_fft_x8000-9000_y2500-3500_t0-500.h5
"""
import numpy as np
import xarray as xr
from pathlib import Path

from reconn_wave_power.io import read_simulation_subregion
from reconn_wave_power.spectrum import compute_psd_map

# =============================================================================
# USER SETTINGS — edit these
# =============================================================================

SIM_PATH   = "/anvil/scratch/x-colbych/Reconnection/wave_power/unoptimized/"
INPUT_FILE = SIM_PATH + "input/input"
OUTPUT_DIR = SIM_PATH + "Output"

X_RANGE = (8000, 9000)   # physical x coords (x_min, x_max)
Y_RANGE = (2500, 3500)   # physical y coords (y_min, y_max)

# Set to None to load all available timesteps, or e.g. list(range(0, 200))
TIMESTEPS = None

# Where to write the output HDF5 file (script directory by default)
SAVE_DIR = Path(__file__).parent

# =============================================================================
# (nothing below here should need editing)
# =============================================================================

COMPONENTS  = ["Ex", "Ey", "Ez", "Bx", "By", "Bz"]
PSD_METHOD  = "fft"


def make_filename(x_range, y_range, timesteps):
    """Build a self-describing filename from the run parameters."""
    x_str = f"x{int(x_range[0])}-{int(x_range[1])}"
    y_str = f"y{int(y_range[0])}-{int(y_range[1])}"
    if timesteps is None:
        t_str = "tAll"
    else:
        t_str = f"t{timesteps[0]}-{timesteps[-1]}"
    return f"psd_{PSD_METHOD}_{x_str}_{y_str}_{t_str}.h5"


def main():
    print("=" * 60)
    print("Subregion PSD (local save)")
    print("=" * 60)
    print(f"  input:      {INPUT_FILE}")
    print(f"  output dir: {OUTPUT_DIR}")
    print(f"  x_range:    {X_RANGE}")
    print(f"  y_range:    {Y_RANGE}")
    print(f"  timesteps:  {'all' if TIMESTEPS is None else f'{len(TIMESTEPS)} ({TIMESTEPS[0]}–{TIMESTEPS[-1]})'}")
    print(f"  method:     {PSD_METHOD}")

    # --- Load subregion ---
    print("\nLoading subregion...")
    ds = read_simulation_subregion(
        input_file=INPUT_FILE,
        output_folder=OUTPUT_DIR,
        x_range=X_RANGE,
        y_range=Y_RANGE,
        timesteps=TIMESTEPS,
        fields=("E", "B"),
        progress=True,
    )
    print(ds)

    dt = ds.attrs.get("dt", float(ds.time[1] - ds.time[0]))
    t0 = int(ds.time[0].values) if TIMESTEPS is None else TIMESTEPS[0]
    t1 = int(ds.time[-1].values) if TIMESTEPS is None else TIMESTEPS[-1]
    print(f"\ndt = {dt}  |  grid: {len(ds.x)} x {len(ds.y)}  |  timesteps: {len(ds.time)}")

    # --- Compute per-pixel PSDs ---
    psd_vars = {}
    freq = None

    for comp in COMPONENTS:
        print(f"Computing PSD map: {comp}...")
        psd_da = compute_psd_map(
            ds[comp],
            dt=dt,
            method=PSD_METHOD,
            detrend=True,
        )
        psd_vars[f"psd_{comp}"] = psd_da
        if freq is None:
            freq = psd_da.coords["frequency"].values

    print(f"\nFrequency axis: {len(freq)} points, [{freq[0]:.4g}, {freq[-1]:.4g}]")

    # --- Build output Dataset ---
    ds_out = xr.Dataset(psd_vars)
    ds_out.attrs = {
        "input_file":    INPUT_FILE,
        "output_folder": OUTPUT_DIR,
        "x_range":       f"{X_RANGE[0]},{X_RANGE[1]}",
        "y_range":       f"{Y_RANGE[0]},{Y_RANGE[1]}",
        "t_range":       f"{t0},{t1}",
        "psd_method":    PSD_METHOD,
        "dt":            dt,
        "components":    ",".join(COMPONENTS),
    }

    # --- Save ---
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    fname = make_filename(X_RANGE, Y_RANGE, TIMESTEPS if TIMESTEPS is not None else [t0, t1])
    out_path = SAVE_DIR / fname
    print(f"\nSaving → {out_path}")
    ds_out.to_netcdf(out_path, engine="h5netcdf")

    # Verify
    ds_check = xr.open_dataset(out_path, engine="h5netcdf")
    print(f"Saved: {dict(ds_check.sizes)}")
    ds_check.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
