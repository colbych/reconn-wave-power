"""
Lagrangian frame frequency spectra.

Trace trajectories following the local E×B drift velocity and compute
frequency PSDs along those trajectories (plasma rest-frame spectra).

APIs:
- compute_exb_velocity(ds, time_idx=None) -> vx, vy
- trace_trajectory(ds, x0, y0, t0_idx) -> x_traj, y_traj, t_traj
- sample_along_trajectory(ds, field, x_traj, y_traj, t_traj) -> DataArray
- lagrangian_psd(ds, field, x0, y0, t0_idx, dt, ...) -> (f, Pxx, trajectory)
"""
from typing import Dict, Optional, Tuple

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

from .spectrum import compute_psd_time


def compute_exb_velocity(
    ds: xr.Dataset,
    time_idx: Optional[int] = None,
) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Compute the in-plane E×B drift velocity: v_ExB = (E × B) / |B|².

    For a 2D (x, y) simulation with all 6 field components the in-plane
    components are:
        vx = (Ey*Bz - Ez*By) / |B|²
        vy = (Ez*Bx - Ex*Bz) / |B|²

    Parameters
    ----------
    ds : xr.Dataset
        Must contain Ex, Ey, Ez, Bx, By, Bz variables.
    time_idx : int, optional
        If given, select a single timestep by positional index before
        computing.  Otherwise compute for all timesteps.

    Returns
    -------
    vx, vy : xr.DataArray
    """
    for comp in ("Ex", "Ey", "Ez", "Bx", "By", "Bz"):
        if comp not in ds:
            raise KeyError(f"Dataset missing required field component '{comp}'")

    if time_idx is not None:
        d = ds.isel(time=time_idx)
    else:
        d = ds

    B2 = d["Bx"] ** 2 + d["By"] ** 2 + d["Bz"] ** 2
    vx = (d["Ey"] * d["Bz"] - d["Ez"] * d["By"]) / B2
    vy = (d["Ez"] * d["Bx"] - d["Ex"] * d["Bz"]) / B2

    vx.name = "vExB_x"
    vy.name = "vExB_y"
    return vx, vy


def trace_trajectory(
    ds: xr.Dataset,
    x0: float,
    y0: float,
    t0_idx: int,
    dt: Optional[float] = None,
    progress: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Integrate a trajectory through the E×B velocity field.

    Steps forward and backward from (x0, y0) at timestep index *t0_idx*
    using simple Euler stepping.  The step size is derived from the actual
    spacing of the time coordinate (not from *dt*).  At each timestep the
    local E×B velocity is obtained by spatial interpolation (bilinear).
    The trajectory is truncated when it exits the simulation domain.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with all six E/B components and dimensions (time, x, y).
    x0, y0 : float
        Starting position.
    t0_idx : int
        Timestep index of the starting point (index into ds.time).
    dt : float, optional
        Deprecated — ignored by the trajectory integrator.  Kept for
        backward-compatible call signatures; callers like ``lagrangian_psd``
        may still pass *dt* for the PSD computation.
    progress : bool, optional
        If True, display a tqdm progress bar during integration.

    Returns
    -------
    x_traj, y_traj : ndarray
        Spatial coordinates along the trajectory.
    t_traj : ndarray
        Time values corresponding to each trajectory point.
    """
    x_vals = ds["x"].values
    y_vals = ds["y"].values
    times = ds["time"].values
    nt = len(times)

    x_min, x_max = float(x_vals[0]), float(x_vals[-1])
    y_min, y_max = float(y_vals[0]), float(y_vals[-1])

    def _in_domain(x: float, y: float) -> bool:
        return x_min <= x <= x_max and y_min <= y <= y_max

    # Grab references to the underlying data stores — no copy, works for
    # both numpy-backed and dask-backed arrays.  We use .variable.data so
    # that indexing later bypasses xarray coordinate machinery.
    _fields = [ds[c].variable.data for c in ("Ex", "Ey", "Ez", "Bx", "By", "Bz")]
    _is_dask = hasattr(_fields[0], "dask")

    # Grid spacing for O(1) index lookup
    dx = float(x_vals[1] - x_vals[0])
    dy = float(y_vals[1] - y_vals[0])
    x0_grid = float(x_vals[0])
    y0_grid = float(y_vals[0])
    nx = len(x_vals)
    ny = len(y_vals)

    # Batch-load timesteps to minimize dask scheduler overhead.
    # For dask-backed data, one dask.compute() call per batch (instead of
    # per timestep) reduces ~1000 scheduler invocations to ~16.
    _BATCH = 64
    _batch_cache: dict = {}

    def _preload_batch(start: int, end: int) -> None:
        """Load field slices for timesteps [start, end) into _batch_cache."""
        _batch_cache.clear()
        end = min(end, nt)
        if _is_dask:
            import dask
            tasks = []
            for ti in range(start, end):
                for f in _fields:
                    tasks.append(f[ti])
            results = dask.compute(*tasks)
            for idx, ti in enumerate(range(start, end)):
                _batch_cache[ti] = results[idx * 6:(idx + 1) * 6]
        else:
            for ti in range(start, end):
                _batch_cache[ti] = tuple(np.asarray(f[ti]) for f in _fields)

    def _get_fields(ti: int, backward: bool = False) -> tuple:
        if ti not in _batch_cache:
            if backward:
                _preload_batch(max(ti - _BATCH + 1, 0), ti + 1)
            else:
                _preload_batch(ti, ti + _BATCH)
        return _batch_cache[ti]

    def _interpolated_velocity(
        ti: int, x: float, y: float, backward: bool = False,
    ) -> Tuple[float, float]:
        """Compute ExB velocity at (x, y) for timestep ti using only the 4 surrounding cells."""
        fi = (x - x0_grid) / dx
        fj = (y - y0_grid) / dy
        fi = max(0.0, min(fi, nx - 1.0))
        fj = max(0.0, min(fj, ny - 1.0))
        i0 = min(int(fi), nx - 2)
        j0 = min(int(fj), ny - 2)
        wi = fi - i0
        wj = fj - j0

        w00 = (1 - wi) * (1 - wj)
        w10 = wi * (1 - wj)
        w01 = (1 - wi) * wj
        w11 = wi * wj

        ex, ey, ez, bx, by, bz = _get_fields(ti, backward=backward)
        i1, j1 = i0 + 1, j0 + 1
        ex00 = float(ex[i0, j0]); ex10 = float(ex[i1, j0])
        ex01 = float(ex[i0, j1]); ex11 = float(ex[i1, j1])
        ey00 = float(ey[i0, j0]); ey10 = float(ey[i1, j0])
        ey01 = float(ey[i0, j1]); ey11 = float(ey[i1, j1])
        ez00 = float(ez[i0, j0]); ez10 = float(ez[i1, j0])
        ez01 = float(ez[i0, j1]); ez11 = float(ez[i1, j1])
        bx00 = float(bx[i0, j0]); bx10 = float(bx[i1, j0])
        bx01 = float(bx[i0, j1]); bx11 = float(bx[i1, j1])
        by00 = float(by[i0, j0]); by10 = float(by[i1, j0])
        by01 = float(by[i0, j1]); by11 = float(by[i1, j1])
        bz00 = float(bz[i0, j0]); bz10 = float(bz[i1, j0])
        bz01 = float(bz[i0, j1]); bz11 = float(bz[i1, j1])

        # ExB / |B|^2 at each of the 4 corners
        b2_00 = bx00*bx00 + by00*by00 + bz00*bz00
        b2_10 = bx10*bx10 + by10*by10 + bz10*bz10
        b2_01 = bx01*bx01 + by01*by01 + bz01*bz01
        b2_11 = bx11*bx11 + by11*by11 + bz11*bz11

        vx00 = (ey00*bz00 - ez00*by00) / b2_00
        vx10 = (ey10*bz10 - ez10*by10) / b2_10
        vx01 = (ey01*bz01 - ez01*by01) / b2_01
        vx11 = (ey11*bz11 - ez11*by11) / b2_11

        vy00 = (ez00*bx00 - ex00*bz00) / b2_00
        vy10 = (ez10*bx10 - ex10*bz10) / b2_10
        vy01 = (ez01*bx01 - ex01*bz01) / b2_01
        vy11 = (ez11*bx11 - ex11*bz11) / b2_11

        vx = vx00*w00 + vx10*w10 + vx01*w01 + vx11*w11
        vy = vy00*w00 + vy10*w10 + vy01*w01 + vy11*w11
        return vx, vy

    # Optional progress bar
    if progress:
        try:
            from tqdm.auto import tqdm
        except ImportError:
            from tqdm import tqdm
        pbar = tqdm(total=nt - 1, desc="Tracing trajectory", unit="ts")
    else:
        pbar = None

    # --- Forward integration (t0_idx -> end) ---
    # Preload first batch covering the forward direction.
    _preload_batch(t0_idx, t0_idx + _BATCH)
    fwd_x, fwd_y, fwd_t = [x0], [y0], [float(times[t0_idx])]
    cx, cy = x0, y0
    for ti in range(t0_idx, nt - 1):
        actual_dt = float(times[ti + 1] - times[ti])
        vx_loc, vy_loc = _interpolated_velocity(ti, cx, cy)
        cx_new = cx + vx_loc * actual_dt
        cy_new = cy + vy_loc * actual_dt
        if pbar is not None:
            pbar.update(1)
        if not _in_domain(cx_new, cy_new):
            break
        cx, cy = cx_new, cy_new
        fwd_x.append(cx)
        fwd_y.append(cy)
        fwd_t.append(float(times[ti + 1]))

    # --- Backward integration (t0_idx -> 0) ---
    # Preload first batch covering the backward direction.
    _preload_batch(max(t0_idx - _BATCH + 1, 0), t0_idx + 1)
    bwd_x, bwd_y, bwd_t = [], [], []
    cx, cy = x0, y0
    for ti in range(t0_idx, 0, -1):
        actual_dt = float(times[ti] - times[ti - 1])
        vx_loc, vy_loc = _interpolated_velocity(ti, cx, cy, backward=True)
        cx_new = cx - vx_loc * actual_dt
        cy_new = cy - vy_loc * actual_dt
        if pbar is not None:
            pbar.update(1)
        if not _in_domain(cx_new, cy_new):
            break
        cx, cy = cx_new, cy_new
        bwd_x.append(cx)
        bwd_y.append(cy)
        bwd_t.append(float(times[ti - 1]))

    if pbar is not None:
        pbar.close()

    # Reverse backward lists so time is monotonically increasing
    bwd_x.reverse()
    bwd_y.reverse()
    bwd_t.reverse()

    x_traj = np.array(bwd_x + fwd_x)
    y_traj = np.array(bwd_y + fwd_y)
    t_traj = np.array(bwd_t + fwd_t)

    return x_traj, y_traj, t_traj


def sample_along_trajectory(
    ds: xr.Dataset,
    field: str,
    x_traj: np.ndarray,
    y_traj: np.ndarray,
    t_traj: np.ndarray,
) -> xr.DataArray:
    """
    Interpolate a field along a trajectory in (x, y, t).

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing *field*.
    field : str
        Variable name, e.g. "Bz".
    x_traj, y_traj, t_traj : ndarray
        Trajectory coordinates (same length).

    Returns
    -------
    xr.DataArray
        1-D time series with coordinate "time".
    """
    if field not in ds:
        raise KeyError(f"Field '{field}' not in dataset")

    da = ds[field]
    x_vals = da.coords["x"].values
    y_vals = da.coords["y"].values
    t_vals = da.coords["time"].values

    data_3d = da.values  # (time, x, y)
    interp = RegularGridInterpolator(
        (t_vals, x_vals, y_vals), data_3d, method="linear",
        bounds_error=False, fill_value=np.nan,
    )

    points = np.column_stack([t_traj, x_traj, y_traj])
    values = interp(points)

    return xr.DataArray(values, dims=("time",), coords={"time": t_traj})


def lagrangian_psd(
    ds: xr.Dataset,
    field: str,
    x0: float,
    y0: float,
    t0_idx: int,
    dt: Optional[float] = None,
    method: str = "welch",
    **psd_kwargs,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Compute a frequency PSD in the Lagrangian (co-moving E×B) frame.

    Convenience wrapper: trace trajectory → sample field → PSD.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with all six E/B components plus the target *field*.
    field : str
        Variable name to analyse (e.g. "Bz").
    x0, y0 : float
        Starting position for the trajectory.
    t0_idx : int
        Starting timestep index.
    dt : float, optional
        Time spacing used for the PSD computation.  If *None*, inferred
        from the median spacing of the dataset time coordinate.
    method : str
        "welch" or "fft"; passed to ``compute_psd_time``.
    **psd_kwargs
        Extra keyword arguments forwarded to ``compute_psd_time``.

    Returns
    -------
    f : ndarray
        Frequency array.
    Pxx : ndarray
        Power spectral density.
    trajectory : dict
        ``{"x_traj": ..., "y_traj": ..., "t_traj": ...}``
    """
    x_traj, y_traj, t_traj = trace_trajectory(ds, x0, y0, t0_idx)
    ts = sample_along_trajectory(ds, field, x_traj, y_traj, t_traj)
    if dt is None:
        dt = float(np.median(np.diff(ds["time"].values)))
    f, Pxx = compute_psd_time(ts, dt=dt, method=method, **psd_kwargs)
    trajectory = {"x_traj": x_traj, "y_traj": y_traj, "t_traj": t_traj}
    return f, Pxx, trajectory
