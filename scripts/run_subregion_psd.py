"""
Subregion PSD Script

Load a spatial subregion of the simulation (using h5py hyperslab reads to
avoid loading the full domain) and compute a per-pixel frequency PSD for
the 6 E/B field components.  Results are saved as a NetCDF file in the
same directory as the simulation output.

Usage:
    python scripts/run_subregion_psd.py config.json

See scripts/example_subregion_config.json for config format.
"""
import sys
import json
import numpy as np
import xarray as xr
from pathlib import Path

from reconn_wave_power.io import read_simulation_subregion
from reconn_wave_power.spectrum import compute_psd_map


COMPONENTS = ["Ex", "Ey", "Ez", "Bx", "By", "Bz"]


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <config.json>")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        config = json.load(f)

    # --- Configuration ---
    INPUT_FILE    = config["input_file"]
    OUTPUT_FOLDER = config["output_folder"]
    X_RANGE       = tuple(config["x_range"])   # [x_min, x_max]
    Y_RANGE       = tuple(config["y_range"])   # [y_min, y_max]
    SAVE_NAME     = config.get("save_name", "subregion_psd")
    PSD_METHOD    = config.get("psd_method", "fft")
    NPERSEG       = config.get("nperseg", None)
    TIMESTEPS     = config.get("timesteps", None)

    # Save next to the simulation output
    save_dir = Path(OUTPUT_FOLDER)

    print("=" * 60)
    print("Subregion PSD Analysis")
    print("=" * 60)
    print(f"  input_file:    {INPUT_FILE}")
    print(f"  output_folder: {OUTPUT_FOLDER}")
    print(f"  x_range:       {X_RANGE}")
    print(f"  y_range:       {Y_RANGE}")
    print(f"  psd_method:    {PSD_METHOD}")
    if NPERSEG is not None:
        print(f"  nperseg:       {NPERSEG}")
    if TIMESTEPS is not None:
        print(f"  timesteps:     {len(TIMESTEPS)} specified")
    print("=" * 60)

    # --- Load subregion ---
    print("\nLoading subregion from disk...")
    ds = read_simulation_subregion(
        input_file=INPUT_FILE,
        output_folder=OUTPUT_FOLDER,
        x_range=X_RANGE,
        y_range=Y_RANGE,
        timesteps=TIMESTEPS,
        fields=("E", "B"),
        progress=True,
    )
    print(ds)

    dt = ds.attrs.get("dt", float(ds.time[1] - ds.time[0]))
    print(f"\ndt = {dt}")
    print(f"Subregion grid: {len(ds.x)} x {len(ds.y)}, {len(ds.time)} timesteps")

    # --- Compute per-pixel PSDs ---
    psd_arrays = {}
    freq = None

    for comp in COMPONENTS:
        print(f"Computing PSD map for {comp}...")
        psd_da = compute_psd_map(
            ds[comp],
            dt=dt,
            method=PSD_METHOD,
            nperseg=NPERSEG,
            detrend=True,
        )
        psd_arrays[f"psd_{comp}"] = psd_da
        if freq is None:
            freq = psd_da.coords["frequency"].values

    print(f"\nFrequency axis: {len(freq)} points, [{freq[0]:.4g}, {freq[-1]:.4g}]")

    # --- Build output Dataset ---
    ds_out = xr.Dataset(psd_arrays)
    ds_out.attrs.update({
        "input_file":    INPUT_FILE,
        "output_folder": OUTPUT_FOLDER,
        "x_range":       list(X_RANGE),
        "y_range":       list(Y_RANGE),
        "psd_method":    PSD_METHOD,
        "dt":            dt,
    })

    out_path = save_dir / f"{SAVE_NAME}.nc"
    print(f"\nSaving to {out_path} ...")
    ds_out.to_netcdf(out_path)

    # --- Verify ---
    ds_check = xr.open_dataset(out_path)
    print(f"Saved dataset: {dict(ds_check.sizes)}")
    ds_check.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
