"""
Subregion complex FFT — local save with descriptive filename

Loads a spatial subregion and saves the complex per-pixel FFT coefficients
for all 6 E/B components.  Storing the complex amplitudes (not just |FFT|²)
preserves phase, which is required for cross-spectral quantities such as
the true Poynting flux S = Re(E × B*).

Output filename example:
    fft_x8000-9000_y2500-3500_t0-6250.h5

The file can then be loaded in the analysis notebook to compute:
    Sx(f) = Re( fft_Ey * conj(fft_Bz)  -  fft_Ez * conj(fft_By) )

Edit the USER SETTINGS block below, then run:
    python scripts/subregion_fft_local.py
"""
import numpy as np
import xarray as xr
from pathlib import Path

from reconn_wave_power.io import read_simulation_subregion
from reconn_wave_power.spectrum import compute_fft_map

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

COMPONENTS = ["Ex", "Ey", "Ez", "Bx", "By", "Bz"]
WINDOW     = "hann"


def make_filename(x_range, y_range, timesteps):
    x_str = f"x{int(x_range[0])}-{int(x_range[1])}"
    y_str = f"y{int(y_range[0])}-{int(y_range[1])}"
    t_str = "tAll" if timesteps is None else f"t{timesteps[0]}-{timesteps[-1]}"
    return f"fft_{x_str}_{y_str}_{t_str}.h5"


def main():
    print("=" * 60)
    print("Subregion complex FFT (local save)")
    print("=" * 60)
    print(f"  input:      {INPUT_FILE}")
    print(f"  output dir: {OUTPUT_DIR}")
    print(f"  x_range:    {X_RANGE}")
    print(f"  y_range:    {Y_RANGE}")
    print(f"  timesteps:  {'all' if TIMESTEPS is None else f'{len(TIMESTEPS)} ({TIMESTEPS[0]}–{TIMESTEPS[-1]})'}")
    print(f"  window:     {WINDOW}")

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

    dt_output = ds.attrs.get("dt_output", float(ds.time[1] - ds.time[0]))
    t0 = int(ds.time[0].values)  if TIMESTEPS is None else TIMESTEPS[0]
    t1 = int(ds.time[-1].values) if TIMESTEPS is None else TIMESTEPS[-1]
    print(f"\ndt_output = {dt_output}  (dt={ds.attrs.get('dt','?')}, ndump={ds.attrs.get('ndump','?')})")
    print(f"grid: {len(ds.x)} x {len(ds.y)}  |  timesteps: {len(ds.time)}")

    # --- Compute per-pixel complex FFTs ---
    fft_vars = {}
    freq = None

    for comp in COMPONENTS:
        print(f"Computing FFT map: {comp}...")
        fft_da = compute_fft_map(
            ds[comp],
            dt=dt_output,
            window=WINDOW,
            detrend=True,
        )
        fft_vars[f"fft_{comp}"] = fft_da
        if freq is None:
            freq = fft_da.coords["frequency"].values

    print(f"\nFrequency axis: {len(freq)} points, [{freq[0]:.4g}, {freq[-1]:.4g}]")
    print(f"Output dtype:   {fft_da.dtype}  (complex — file size ~2× auto-PSD)")

    # --- Build output Dataset ---
    ds_out = xr.Dataset(fft_vars)
    ds_out.attrs = {
        "input_file":    INPUT_FILE,
        "output_folder": OUTPUT_DIR,
        "x_range":       f"{X_RANGE[0]},{X_RANGE[1]}",
        "y_range":       f"{Y_RANGE[0]},{Y_RANGE[1]}",
        "t_range":       f"{t0},{t1}",
        "window":        WINDOW,
        "dt":            ds.attrs.get("dt", "unknown"),
        "ndump":         ds.attrs.get("ndump", "unknown"),
        "dt_output":     dt_output,
        "components":    ",".join(COMPONENTS),
        "normalization": "|fft|**2 = one-sided PSD (pre factor-of-2); Sx = Re(fft_Ey*conj(fft_Bz) - fft_Ez*conj(fft_By))",
    }

    # --- Save ---
    # invalid_netcdf=True is required for h5netcdf to write complex arrays
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    fname = make_filename(X_RANGE, Y_RANGE, TIMESTEPS if TIMESTEPS is not None else [t0, t1])
    out_path = SAVE_DIR / fname
    print(f"\nSaving → {out_path}")
    ds_out.to_netcdf(out_path, engine="h5netcdf", invalid_netcdf=True)

    # Verify
    ds_check = xr.open_dataset(out_path, engine="h5netcdf", invalid_netcdf=True)
    print(f"Saved: {dict(ds_check.sizes)}")
    print(f"Dtypes: { {k: str(v.dtype) for k, v in ds_check.data_vars.items()} }")
    ds_check.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
