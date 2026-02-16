"""
Batch Lagrangian Analysis Script

Run the Lagrangian tracer on a grid of starting positions (N vertical lines x M points each),
sample all 6 field components along each trajectory, compute frequency PSDs, and save
everything to a NetCDF file.

Usage:
    python scripts/run_batch_analysis.py config.json

See scripts/example_batch_config.json for config format.
"""
import matplotlib
matplotlib.use("Agg")

import sys
import json
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path

from reconn_wave_power.io import read_simulation
from reconn_wave_power.spectrum import compute_psd_time
from reconn_wave_power.lagrangian import (
    compute_exb_velocity,
    trace_trajectories,
    sample_along_trajectories,
)


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <config.json>")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        config = json.load(f)

    # --- Configuration ---
    INPUT_FILE = config["input_file"]
    OUTPUT_FOLDER = config["output_folder"]
    SAVE_DIR = Path(config.get("save_dir", "."))
    SAVE_NAME = config.get("save_name", "lagrangian_batch")
    FIELD = config.get("field", "Bz")
    T0_IDX = config.get("t0_idx", 0)
    N_LINES = config.get("n_lines", 3)
    N_PER_LINE = config.get("n_per_line", 20)
    PSD_METHOD = config.get("psd_method", "fft")

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Batch Lagrangian Analysis")
    print("=" * 60)
    print(f"  input_file:    {INPUT_FILE}")
    print(f"  output_folder: {OUTPUT_FOLDER}")
    print(f"  save_dir:      {SAVE_DIR}")
    print(f"  save_name:     {SAVE_NAME}")
    print(f"  field:         {FIELD}")
    print(f"  t0_idx:        {T0_IDX}")
    print(f"  n_lines:       {N_LINES}")
    print(f"  n_per_line:    {N_PER_LINE}")
    print(f"  psd_method:    {PSD_METHOD}")
    print("=" * 60)

    # --- Load simulation data ---
    print("\nLoading simulation data...")
    ds = read_simulation(
        input_file=INPUT_FILE,
        output_folder=OUTPUT_FOLDER,
        fields=("E", "B"),
        progress=True,
    )
    print(ds)

    dx = float(ds.x[1] - ds.x[0])
    dy = float(ds.y[1] - ds.y[0])
    dt = ds.attrs.get("dt", float(ds.time[1] - ds.time[0]))
    Lx = float(ds.x[-1] - ds.x[0]) + dx
    Ly = float(ds.y[-1] - ds.y[0]) + dy

    print(f"\ndx = {dx}, dy = {dy}, dt = {dt}")
    print(f"Lx = {Lx:.1f}, Ly = {Ly:.1f}")
    print(f"Grid: {len(ds.x)} x {len(ds.y)}, {len(ds.time)} timesteps")

    # --- Build starting positions ---
    print("\nBuilding starting positions...")
    x_min = float(ds.x[0])
    y_min = float(ds.y[0])
    x_lines = x_min + Lx * np.arange(1, N_LINES + 1) / (N_LINES + 1)
    y_starts = y_min + np.linspace(0, Ly, N_PER_LINE, endpoint=False)

    x0s = np.repeat(x_lines, N_PER_LINE)
    y0s = np.tile(y_starts, N_LINES)
    N_total = len(x0s)

    print(f"x lines: {x_lines}")
    print(f"Total trajectories: {N_total}")

    # Plot starting grid
    fig, ax = plt.subplots(figsize=(10, 5))
    snapshot = ds[FIELD].isel(time=T0_IDX)
    im = ax.pcolormesh(ds.x, ds.y, snapshot.values.T, shading="auto", cmap="RdBu_r")
    colors = plt.cm.tab10(np.linspace(0, 1, N_LINES))
    for i, xl in enumerate(x_lines):
        mask = x0s == xl
        ax.plot(x0s[mask], y0s[mask], "o", color=colors[i], ms=4,
                label=f"x = {xl:.1f}")
    ax.set_xlabel(r"x [$d_i$]")
    ax.set_ylabel(r"y [$d_i$]")
    ax.set_title("Starting positions")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    plt.colorbar(im, ax=ax, label=FIELD)
    plt.tight_layout()
    fig.savefig(SAVE_DIR / f"{SAVE_NAME}_starting_positions.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"Saved starting positions plot.")

    # --- Trace trajectories ---
    print("\nTracing trajectories...")
    x_traj, y_traj, t_traj, active_mask = trace_trajectories(
        ds, x0s, y0s, T0_IDX, progress=True,
    )
    print(f"x_traj shape: {x_traj.shape}")
    print(f"Active fraction: {active_mask.mean():.2%}")

    # Plot trajectories
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = plt.cm.Paired(np.linspace(0, 1, N_LINES))
    for i, xl in enumerate(x_lines):
        idx_start = i * N_PER_LINE
        idx_end = (i + 1) * N_PER_LINE
        for j in range(idx_start, idx_end):
            valid = active_mask[j]
            ax.plot(x_traj[j, valid], y_traj[j, valid], "-",
                    color=colors[i], lw=0.5, alpha=1.0)
        ax.plot([], [], "-", color=colors[i], lw=1.5, label=f"x0 = {xl:.1f}")
    ax.set_xlabel(r"x [$d_i$]")
    ax.set_ylabel(r"y [$d_i$]")
    ax.set_title("Trajectories")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    fig.savefig(SAVE_DIR / f"{SAVE_NAME}_trajectories.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("Saved trajectories plot.")

    # --- Sample fields along trajectories ---
    COMPONENTS = ["Ex", "Ey", "Ez", "Bx", "By", "Bz"]
    sampled = {}

    for comp in COMPONENTS:
        print(f"Sampling {comp}...")
        sampled[comp] = sample_along_trajectories(
            ds, comp, x_traj, y_traj, t_traj, active_mask,
        )

    print(f"Sampled shape per component: {sampled[COMPONENTS[0]].shape}")

    # --- Compute PSDs ---
    print("\nComputing PSDs...")
    psd_results = {}
    freq = None

    for comp in COMPONENTS:
        print(f"  Computing PSDs for {comp}...")
        psd_list = []
        for i in range(N_total):
            valid = active_mask[i]
            vals = sampled[comp].values[i, valid]
            t_valid = t_traj[valid]
            da = xr.DataArray(vals, dims=("time",), coords={"time": t_valid})
            f, Pxx = compute_psd_time(da, dt=dt, method=PSD_METHOD)
            psd_list.append(Pxx)
            if freq is None:
                freq = f
        psd_results[comp] = np.array(psd_list)

    print(f"Frequency array: {len(freq)} points, [{freq[0]:.4f}, {freq[-1]:.4f}]")
    print(f"PSD shape per component: {psd_results[COMPONENTS[0]].shape}")

    # --- Save results ---
    print("\nSaving results...")
    ds_out = xr.Dataset(
        {
            "x_traj": (["trajectory", "time"], x_traj),
            "y_traj": (["trajectory", "time"], y_traj),
            "active_mask": (["trajectory", "time"], active_mask),
            "x0": (["trajectory"], x0s),
            "y0": (["trajectory"], y0s),
        },
        coords={
            "time": t_traj,
            "trajectory": np.arange(N_total),
            "frequency": freq,
        },
        attrs={
            "input_file": INPUT_FILE,
            "output_folder": OUTPUT_FOLDER,
            "t0_idx": T0_IDX,
            "n_lines": N_LINES,
            "n_per_line": N_PER_LINE,
            "psd_method": PSD_METHOD,
            "dt": dt,
        },
    )

    for comp in COMPONENTS:
        ds_out[f"{comp}_sampled"] = (["trajectory", "time"], sampled[comp].values)
        ds_out[f"psd_{comp}"] = (["trajectory", "frequency"], psd_results[comp])

    out_path = SAVE_DIR / f"{SAVE_NAME}.nc"
    ds_out.to_netcdf(out_path)
    print(f"Saved to {out_path}")

    # --- Verify ---
    print("\nVerifying saved data...")
    ds_check = xr.open_dataset(out_path)
    assert ds_check.sizes["trajectory"] == N_total
    assert ds_check.sizes["time"] == len(t_traj)
    assert ds_check.sizes["frequency"] == len(freq)
    print(f"Trajectories: {ds_check.sizes['trajectory']}")
    print(f"Timesteps: {ds_check.sizes['time']}")
    print(f"Frequencies: {ds_check.sizes['frequency']}")
    print("All shapes match.")
    ds_check.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
