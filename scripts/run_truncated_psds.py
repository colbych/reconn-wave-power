"""
Truncated PSD & Movie Script

Load saved trajectory data from the batch analysis, optionally generate
per-trajectory movies, truncate trajectories at user-specified end times,
recompute frequency PSDs, and save results.

Usage:
    python scripts/run_truncated_psds.py config.json

See scripts/example_psd_config.json for config format.
"""
import matplotlib
matplotlib.use("Agg")

import os
import sys
import json
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
from reconn_wave_power.io import read_simulation
from reconn_wave_power.spectrum import compute_psd_time


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <config.json>")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        config = json.load(f)

    # --- Configuration ---
    INPUT_FILE = config["input_file"]
    OUTPUT_FOLDER = config["output_folder"]
    BATCH_FILE = config["batch_file"]
    END_TIMES_FILE = config.get("end_times_file", None)
    SAVE_DIR = Path(config.get("save_dir", "."))
    SAVE_NAME = config.get("save_name", "lagrangian_batch_truncated")
    PSD_DIR = Path(config.get("psd_dir", "psd_plots"))
    PSD_METHOD = config.get("psd_method", "fft")
    MAKE_MOVIES = config.get("make_movies", False)
    MOVIE_DIR = Path(config.get("movie_dir", "movies"))
    FRAME_STRIDE = config.get("frame_stride", 10)
    TAIL_LEN = config.get("tail_len", 20)
    MOVIE_FPS = config.get("movie_fps", 20)
    MOVIE_DPI = config.get("movie_dpi", 150)
    N_CONTOURS = config.get("n_contours", 20)
    BZ_MIN = config.get("bz_min", None)
    BZ_MAX = config.get("bz_max", None)

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    PSD_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Truncated PSD & Movie Pipeline")
    print("=" * 60)
    print(f"  input_file:      {INPUT_FILE}")
    print(f"  output_folder:   {OUTPUT_FOLDER}")
    print(f"  batch_file:      {BATCH_FILE}")
    print(f"  end_times_file:  {END_TIMES_FILE}")
    print(f"  save_dir:        {SAVE_DIR}")
    print(f"  save_name:       {SAVE_NAME}")
    print(f"  psd_dir:         {PSD_DIR}")
    print(f"  psd_method:      {PSD_METHOD}")
    print(f"  make_movies:     {MAKE_MOVIES}")
    if MAKE_MOVIES:
        print(f"  movie_dir:       {MOVIE_DIR}")
        print(f"  frame_stride:    {FRAME_STRIDE}")
        print(f"  tail_len:        {TAIL_LEN}")
        print(f"  movie_fps:       {MOVIE_FPS}")
        print(f"  movie_dpi:       {MOVIE_DPI}")
        print(f"  n_contours:      {N_CONTOURS}")
        print(f"  bz_min:          {BZ_MIN}")
        print(f"  bz_max:          {BZ_MAX}")
    print("=" * 60)

    # --- Load batch data ---
    print("\nLoading batch data...")
    ds_batch = xr.open_dataset(BATCH_FILE)
    print(ds_batch)

    x_traj = ds_batch["x_traj"].values
    y_traj = ds_batch["y_traj"].values
    active_mask = ds_batch["active_mask"].values.astype(bool)
    t_traj = ds_batch["time"].values
    x0s = ds_batch["x0"].values
    y0s = ds_batch["y0"].values
    N_total = len(x0s)
    dt = ds_batch.attrs["dt"]

    print(f"Trajectories: {N_total}, Timesteps: {len(t_traj)}, dt: {dt}")

    # --- Load end times ---
    if END_TIMES_FILE is not None:
        print(f"\nLoading end times from {END_TIMES_FILE}...")
        with open(END_TIMES_FILE) as f:
            end_times_raw = json.load(f)
        # Convert raw end times to indices: end_time_idx = time_value // 2 + 1
        # (matching the notebook convention)
        end_time_idx = np.full(N_total, len(t_traj), dtype=int)
        for key, val in end_times_raw.items():
            idx = int(key)
            end_time_idx[idx] = int(val) // 2 + 1
        print(f"Loaded end times for {len(end_times_raw)} trajectories.")
    else:
        print("\nNo end times file provided; using full trajectories.")
        end_time_idx = np.full(N_total, len(t_traj), dtype=int)

    # Print summary table
    print(f"\n{'Traj':>4s}  {'x0':>8s}  {'y0':>8s}  {'end_idx':>7s}  {'end_time':>10s}")
    print("-" * 45)
    for i in range(N_total):
        t_end = t_traj[end_time_idx[i] - 1] if end_time_idx[i] > 0 else 0.0
        flag = "  <-- truncated" if end_time_idx[i] < len(t_traj) else ""
        print(f"{i:4d}  {x0s[i]:8.1f}  {y0s[i]:8.1f}  {end_time_idx[i]:7d}  {t_end:10.1f}{flag}")

    # --- Movies ---
    if MAKE_MOVIES:
        MOVIE_DIR.mkdir(parents=True, exist_ok=True)
        print("\nLoading simulation B fields for movies...")
        ds = read_simulation(
            input_file=INPUT_FILE,
            output_folder=OUTPUT_FOLDER,
            fields=("B",),
            progress=True,
        )

        # Color limits
        if BZ_MIN is not None and BZ_MAX is not None:
            vlim = max(abs(BZ_MIN), abs(BZ_MAX))
        else:
            vlim = 0.5  # default
        print(f"Symmetric color limit: +/-{vlim:.4f}")

        # Preload strided frames
        import dask

        nt = len(t_traj)
        frame_indices = np.arange(0, nt, FRAME_STRIDE)
        n_frames = len(frame_indices)
        print(f"Strided frames: {n_frames} (stride={FRAME_STRIDE}, original={nt})")

        BATCH_SIZE = 64
        bz_frames = np.empty((n_frames, len(ds.x), len(ds.y)), dtype=np.float32)
        bx_frames = np.empty((n_frames, len(ds.x), len(ds.y)), dtype=np.float32)
        by_frames = np.empty((n_frames, len(ds.x), len(ds.y)), dtype=np.float32)

        for batch_start in range(0, n_frames, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, n_frames)
            delayed_bz = [ds["Bz"].isel(time=int(frame_indices[i])).data
                          for i in range(batch_start, batch_end)]
            delayed_bx = [ds["Bx"].isel(time=int(frame_indices[i])).data
                          for i in range(batch_start, batch_end)]
            delayed_by = [ds["By"].isel(time=int(frame_indices[i])).data
                          for i in range(batch_start, batch_end)]
            results = dask.compute(*(delayed_bz + delayed_bx + delayed_by))
            n_batch = batch_end - batch_start
            for j in range(n_batch):
                bz_frames[batch_start + j] = results[j]
                bx_frames[batch_start + j] = results[n_batch + j]
                by_frames[batch_start + j] = results[2 * n_batch + j]
            print(f"  Loaded frames {batch_start}-{batch_end - 1} / {n_frames - 1}")

        print(f"bz_frames: {bz_frames.nbytes / 1e6:.1f} MB")
        print(f"bx_frames: {bx_frames.nbytes / 1e6:.1f} MB")
        print(f"by_frames: {by_frames.nbytes / 1e6:.1f} MB")

        # Compute flux function using two-step method (Bx and By)
        x_coords = ds.x.values
        y_coords = ds.y.values
        dx = x_coords[1] - x_coords[0]
        dy = y_coords[1] - y_coords[0]
        psi_frames = np.zeros_like(bx_frames)
        # Step 1: boundary at x=0 — integrate Bx along y
        psi_frames[:, 0, 1:] = np.cumsum(bx_frames[:, 0, 1:], axis=-1) * dy
        # Step 2: interior — extend using -∂ψ/∂x = By
        psi_frames[:, 1:, :] = psi_frames[:, 0:1, :] - np.cumsum(by_frames[:, 1:, :], axis=-2) * dx
        print(f"psi_frames: {psi_frames.nbytes / 1e6:.1f} MB")

        # Determine movie writer and file extension
        from matplotlib.animation import FFMpegWriter
        if FFMpegWriter.isAvailable():
            movie_writer = "ffmpeg"
            movie_ext = ".mp4"
        else:
            print("WARNING: ffmpeg not found; falling back to Pillow (.gif)")
            movie_writer = "pillow"
            movie_ext = ".gif"

        # Generate movies
        print("\nGenerating movies...")
        for traj_idx in range(N_total):
            x0 = x0s[traj_idx]
            y0 = y0s[traj_idx]
            fname = MOVIE_DIR / f"trajectory_{traj_idx:02d}_x0_{x0:.1f}_y0_{y0:.1f}{movie_ext}"

            fig, ax = plt.subplots(figsize=(12, 4))
            im = ax.pcolormesh(
                ds.x, ds.y, bz_frames[0].T, shading="auto", cmap="RdBu_r",
                vmin=-vlim, vmax=vlim,
            )
            # Use a mutable container instead of nonlocal
            contour_holder = [ax.contour(
                ds.x.values, ds.y.values, psi_frames[0].T,
                levels=N_CONTOURS, colors="k", linewidths=0.5,
            )]
            (tail_line,) = ax.plot([], [], "-", color="black", lw=1.5, alpha=0.7)
            (marker,) = ax.plot([], [], "o", color="black", ms=6, mec="white", mew=0.8)
            ax.set_xlabel(r"x [$d_i$]")
            ax.set_ylabel(r"y [$d_i$]")
            ax.set_aspect("equal")
            plt.colorbar(im, ax=ax, label="Bz")
            title = ax.set_title("")

            xt = x_traj[traj_idx]
            yt = y_traj[traj_idx]

            def update(frame_idx, xt=xt, yt=yt, contour_holder=contour_holder):
                im.set_array(bz_frames[frame_idx].T.ravel())
                contour_holder[0].remove()
                contour_holder[0] = ax.contour(
                    ds.x.values, ds.y.values, psi_frames[frame_idx].T,
                    levels=N_CONTOURS, colors="k", linewidths=0.5,
                )
                orig_idx = frame_indices[frame_idx]
                t_start = max(0, frame_idx - TAIL_LEN)
                tail_orig = frame_indices[t_start:frame_idx + 1]
                tail_line.set_data(xt[tail_orig], yt[tail_orig])
                marker.set_data([xt[orig_idx]], [yt[orig_idx]])
                title.set_text(
                    f"Trajectory {traj_idx:02d}  "
                    f"(x0={x0:.1f}, y0={y0:.1f})  "
                    f"t = {t_traj[orig_idx]:.1f} $\\Omega_{{ci}}^{{-1}}$"
                )
                return im, tail_line, marker, title

            anim = FuncAnimation(fig, update, frames=n_frames, interval=50, blit=False)
            plt.close(fig)
            anim.save(str(fname), writer=movie_writer, fps=MOVIE_FPS, dpi=MOVIE_DPI)
            print(f"  [{traj_idx + 1}/{N_total}] Saved: {fname}")

        print("All movies saved.")

    # --- Compute truncated PSDs ---
    print("\nComputing truncated PSDs...")
    COMPONENTS = ["Ex", "Ey", "Ez", "Bx", "By", "Bz"]
    DERIVED = ["Sx", "Sy", "Sz", "uE", "uB"]
    psd_results = {}
    freq_results = {}

    _dt = t_traj[2] - t_traj[1]

    for comp in COMPONENTS:
        print(f"  Computing PSDs for {comp}...")
        sampled = ds_batch[f"{comp}_sampled"].values
        psd_list = []
        freq_list = []
        for i in range(N_total):
            end = end_time_idx[i]
            vals = sampled[i, :end]
            t_valid = t_traj[:end]
            da = xr.DataArray(vals, dims=("time",), coords={"time": t_valid})
            f, Pxx = compute_psd_time(da, dt=_dt, method=PSD_METHOD)
            freq_list.append(f)
            psd_list.append(Pxx)
        psd_results[comp] = psd_list
        freq_results[comp] = freq_list

    # Derived quantities
    Ex_s = ds_batch["Ex_sampled"].values
    Ey_s = ds_batch["Ey_sampled"].values
    Ez_s = ds_batch["Ez_sampled"].values
    Bx_s = ds_batch["Bx_sampled"].values
    By_s = ds_batch["By_sampled"].values
    Bz_s = ds_batch["Bz_sampled"].values

    derived_fields = {
        "Sx": Ey_s * Bz_s - Ez_s * By_s,
        "Sy": Ez_s * Bx_s - Ex_s * Bz_s,
        "Sz": Ex_s * By_s - Ey_s * Bx_s,
        "uE": 0.5 * (Ex_s**2 + Ey_s**2 + Ez_s**2),
        "uB": 0.5 * (Bx_s**2 + By_s**2 + Bz_s**2),
    }

    for comp in DERIVED:
        print(f"  Computing PSDs for {comp}...")
        sampled = derived_fields[comp]
        psd_list = []
        freq_list = []
        for i in range(N_total):
            end = end_time_idx[i]
            vals = sampled[i, :end]
            t_valid = t_traj[:end]
            da = xr.DataArray(vals, dims=("time",), coords={"time": t_valid})
            f, Pxx = compute_psd_time(da, dt=_dt, method=PSD_METHOD)
            freq_list.append(f)
            psd_list.append(Pxx)
        psd_results[comp] = psd_list
        freq_results[comp] = freq_list

    print(f"PSD lengths per trajectory: {[len(p) for p in psd_results[COMPONENTS[0]]]}")

    # --- Save truncated results ---
    print("\nSaving truncated results...")
    ALL_COMPS = COMPONENTS + DERIVED
    max_nfreq = max(len(p) for p in psd_results[COMPONENTS[0]])
    freq_max = freq_results[COMPONENTS[0]][
        np.argmax([len(f) for f in freq_results[COMPONENTS[0]]])
    ]

    def pad_to_length(arr, length):
        padded = np.full(length, np.nan)
        padded[:len(arr)] = arr
        return padded

    ds_out = xr.Dataset(
        {
            "end_time_idx": (["trajectory"], end_time_idx),
            "x0": (["trajectory"], x0s),
            "y0": (["trajectory"], y0s),
            "n_freq": (["trajectory"],
                       np.array([len(p) for p in psd_results[COMPONENTS[0]]])),
        },
        coords={
            "trajectory": np.arange(N_total),
            "frequency": freq_max,
        },
        attrs={
            "source_file": BATCH_FILE,
            "psd_method": PSD_METHOD,
            "dt": dt,
        },
    )

    for comp in ALL_COMPS:
        padded = np.array([pad_to_length(p, max_nfreq) for p in psd_results[comp]])
        ds_out[f"psd_{comp}"] = (["trajectory", "frequency"], padded)

    out_path = SAVE_DIR / f"{SAVE_NAME}.nc"
    ds_out.to_netcdf(out_path)
    print(f"Saved to {out_path}")
    print(ds_out)

    # --- Per-trajectory PSD plots ---
    print("\nGenerating per-trajectory PSD plots...")
    E_COMPS = ["Ex", "Ey", "Ez"]
    B_COMPS = ["Bx", "By", "Bz"]
    S_COMPS = ["Sx", "Sy", "Sz"]
    ENERGY_COMPS = ["uE", "uB"]
    COMP_COLORS = {
        "Ex": "C0", "Ey": "C1", "Ez": "C2",
        "Bx": "C3", "By": "C4", "Bz": "C5",
        "Sx": "C0", "Sy": "C1", "Sz": "C2",
        "uE": "C6", "uB": "C7",
    }

    # Compute global axis limits
    all_freqs_min, all_freqs_max = np.inf, -np.inf
    e_psd_min, e_psd_max = np.inf, -np.inf
    b_psd_min, b_psd_max = np.inf, -np.inf
    d_psd_min, d_psd_max = np.inf, -np.inf

    for traj_idx in range(N_total):
        for comp in COMPONENTS + DERIVED:
            f = freq_results[comp][traj_idx]
            p = psd_results[comp][traj_idx]
            f_pos, p_pos = f[1:], p[1:]
            if len(f_pos) == 0:
                continue
            all_freqs_min = min(all_freqs_min, f_pos[0])
            all_freqs_max = max(all_freqs_max, f_pos[-1])
            p_valid = p_pos[p_pos > 0]
            if len(p_valid) == 0:
                continue
            if comp in E_COMPS:
                e_psd_min = min(e_psd_min, p_valid.min())
                e_psd_max = max(e_psd_max, p_valid.max())
            elif comp in B_COMPS:
                b_psd_min = min(b_psd_min, p_valid.min())
                b_psd_max = max(b_psd_max, p_valid.max())
            else:
                d_psd_min = min(d_psd_min, p_valid.min())
                d_psd_max = max(d_psd_max, p_valid.max())

    FREQ_LIM = (all_freqs_min * 0.8, all_freqs_max * 1.2)
    E_PSD_LIM = (e_psd_min * 0.3, e_psd_max * 3.0)
    B_PSD_LIM = (b_psd_min * 0.3, b_psd_max * 3.0)
    D_PSD_LIM = (d_psd_min * 0.3, d_psd_max * 3.0)

    print(f"Freq limits:     {FREQ_LIM}")
    print(f"E PSD limits:    {E_PSD_LIM}")
    print(f"B PSD limits:    {B_PSD_LIM}")
    print(f"Derived limits:  {D_PSD_LIM}")

    for traj_idx in range(N_total):
        x0 = x0s[traj_idx]
        y0 = y0s[traj_idx]
        end = end_time_idx[traj_idx]
        t_end = t_traj[end - 1] if end > 0 else 0.0

        fig, axes = plt.subplots(1, 3, figsize=(20, 5))

        # Left panel: E-field
        ax = axes[0]
        for comp in E_COMPS:
            f = freq_results[comp][traj_idx]
            p = psd_results[comp][traj_idx]
            ax.loglog(f[1:], p[1:], color=COMP_COLORS[comp], lw=1.2, label=comp)
        ax.set_xlim(FREQ_LIM)
        ax.set_ylim(E_PSD_LIM)
        ax.set_xlabel(r"frequency [$\Omega_{ci}$]")
        ax.set_ylabel("PSD")
        ax.set_title("E-field")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, which="both")

        # Middle panel: B-field
        ax = axes[1]
        for comp in B_COMPS:
            f = freq_results[comp][traj_idx]
            p = psd_results[comp][traj_idx]
            ax.loglog(f[1:], p[1:], color=COMP_COLORS[comp], lw=1.2, label=comp)
        ax.set_xlim(FREQ_LIM)
        ax.set_ylim(B_PSD_LIM)
        ax.set_xlabel(r"frequency [$\Omega_{ci}$]")
        ax.set_ylabel("PSD")
        ax.set_title("B-field")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, which="both")

        # Right panel: Poynting + energy
        ax = axes[2]
        for comp in S_COMPS:
            f = freq_results[comp][traj_idx]
            p = psd_results[comp][traj_idx]
            ax.loglog(f[1:], p[1:], color=COMP_COLORS[comp], lw=1.2, label=comp)
        for comp in ENERGY_COMPS:
            f = freq_results[comp][traj_idx]
            p = psd_results[comp][traj_idx]
            ax.loglog(f[1:], p[1:], color=COMP_COLORS[comp], lw=1.2,
                      linestyle="--", label=comp)
        ax.set_xlim(FREQ_LIM)
        ax.set_ylim(D_PSD_LIM)
        ax.set_xlabel(r"frequency [$\Omega_{ci}$]")
        ax.set_ylabel("PSD")
        ax.set_title("Poynting & Energy")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, which="both")

        fig.suptitle(
            f"Trajectory {traj_idx:02d}  (x0={x0:.1f}, y0={y0:.1f})  "
            f"t_end={t_end:.0f} $\\Omega_{{ci}}^{{-1}}$ ({end} steps)",
            fontsize=12,
        )
        plt.tight_layout()

        fname = PSD_DIR / f"psd_{traj_idx:02d}_x0_{x0:.1f}_y0_{y0:.1f}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved {N_total} PSD plots to {PSD_DIR}/")
    ds_batch.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
