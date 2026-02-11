"""
Lagrangian frame frequency spectra.

Trace trajectories following the local E×B drift velocity and compute
frequency PSDs along those trajectories (plasma rest-frame spectra).

APIs:
- compute_exb_velocity(ds, time_idx=None) -> vx, vy
- trace_trajectory(ds, x0, y0, t0_idx) -> x_traj, y_traj, t_traj
- trace_trajectories(ds, x0s, y0s, t0_idx) -> x_traj, y_traj, t_traj, active_mask
- sample_along_trajectory(ds, field, x_traj, y_traj, t_traj) -> DataArray
- sample_along_trajectories(ds, field, x_traj, y_traj, t_traj, active_mask) -> DataArray
- lagrangian_psd(ds, field, x0, y0, t0_idx, dt, ...) -> (f, Pxx, trajectory)
- lagrangian_psds(ds, field, x0s, y0s, t0_idx, dt, ...) -> list of (f, Pxx, trajectory)
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
import xarray as xr
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


def _bilinear_interp_batch(
    field_2d: np.ndarray,
    fx: np.ndarray,
    fy: np.ndarray,
    nx: int,
    ny: int,
) -> np.ndarray:
    """Bilinear interpolation of a 2D field at M arbitrary fractional-index positions.

    Parameters
    ----------
    field_2d : ndarray, shape (nx, ny)
    fx, fy : ndarray, shape (M,)
        Fractional indices (already clipped to valid range).
    nx, ny : int
        Grid dimensions (used for clamping).

    Returns
    -------
    ndarray, shape (M,)
    """
    fx = fx % nx
    fy = fy % ny
    i0 = np.floor(fx).astype(np.intp)
    j0 = np.floor(fy).astype(np.intp)
    wi = fx - i0
    wj = fy - j0
    i1 = (i0 + 1) % nx
    j1 = (j0 + 1) % ny
    return (
        field_2d[i0, j0] * (1 - wi) * (1 - wj)
        + field_2d[i1, j0] * wi * (1 - wj)
        + field_2d[i0, j1] * (1 - wi) * wj
        + field_2d[i1, j1] * wi * wj
    )


def _exb_velocity_batch(
    fields: tuple,
    fx: np.ndarray,
    fy: np.ndarray,
    nx: int,
    ny: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute ExB velocity at M points.

    Matches the scalar implementation: compute ExB/|B|^2 at the 4 surrounding
    grid corners, then bilinearly interpolate the *velocity* (not the fields).
    This matters because the operation is nonlinear.
    """
    fx = fx % nx
    fy = fy % ny
    i0 = np.floor(fx).astype(np.intp)
    j0 = np.floor(fy).astype(np.intp)
    wi = fx - i0
    wj = fy - j0
    i1 = (i0 + 1) % nx
    j1 = (j0 + 1) % ny

    ex, ey, ez, bx, by, bz = fields

    # Compute vx, vy at each of the 4 corners
    vx_out = np.zeros_like(fx)
    vy_out = np.zeros_like(fx)

    for ii, jj, w in [
        (i0, j0, (1 - wi) * (1 - wj)),
        (i1, j0, wi * (1 - wj)),
        (i0, j1, (1 - wi) * wj),
        (i1, j1, wi * wj),
    ]:
        b2 = bx[ii, jj] ** 2 + by[ii, jj] ** 2 + bz[ii, jj] ** 2
        vx_corner = (ey[ii, jj] * bz[ii, jj] - ez[ii, jj] * by[ii, jj]) / b2
        vy_corner = (ez[ii, jj] * bx[ii, jj] - ex[ii, jj] * bz[ii, jj]) / b2
        vx_out += vx_corner * w
        vy_out += vy_corner * w

    return vx_out, vy_out


def trace_trajectories(
    ds: xr.Dataset,
    x0s: np.ndarray,
    y0s: np.ndarray,
    t0_idx: int,
    *,
    progress: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Trace N trajectories simultaneously through the E×B velocity field.

    All trajectories start at timestep *t0_idx* and are integrated forward
    and backward in time.  This is much faster than calling
    ``trace_trajectory`` N times because the inner loop is vectorized over
    trajectories using numpy array operations.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with all six E/B components and dimensions (time, x, y).
    x0s, y0s : array_like, shape (N,)
        Starting positions for each trajectory.
    t0_idx : int
        Timestep index of the starting point.
    progress : bool, optional
        If True, display a tqdm progress bar.

    Returns
    -------
    x_traj : ndarray, shape (N, n_times)
        x-positions along each trajectory (NaN after domain exit).
    y_traj : ndarray, shape (N, n_times)
        y-positions along each trajectory (NaN after domain exit).
    t_traj : ndarray, shape (n_times,)
        Shared time axis covering the full dataset time range.
    active_mask : ndarray, shape (N, n_times), dtype bool
        True where the trajectory is still inside the domain.
    """
    x0s = np.asarray(x0s, dtype=np.float64)
    y0s = np.asarray(y0s, dtype=np.float64)
    N = len(x0s)

    x_vals = ds["x"].values
    y_vals = ds["y"].values
    times = ds["time"].values
    nt = len(times)

    x_min = float(x_vals[0])
    y_min = float(y_vals[0])
    dx_grid = float(x_vals[1] - x_vals[0])
    dy_grid = float(y_vals[1] - y_vals[0])
    x0_grid = x_min
    y0_grid = y_min
    nx = len(x_vals)
    ny = len(y_vals)
    Lx = nx * dx_grid
    Ly = ny * dy_grid

    _fields = [ds[c].variable.data for c in ("Ex", "Ey", "Ez", "Bx", "By", "Bz")]
    _is_dask = hasattr(_fields[0], "dask")

    # Batch loading for dask
    _BATCH = 64
    _batch_cache: dict = {}

    def _preload_batch(start: int, end: int) -> None:
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

    # Allocate output arrays — full time range
    x_out = np.full((N, nt), np.nan)
    y_out = np.full((N, nt), np.nan)
    active_mask = np.zeros((N, nt), dtype=bool)

    # Set starting positions
    x_out[:, t0_idx] = x0s
    y_out[:, t0_idx] = y0s
    active_mask[:, t0_idx] = True

    # Optional progress bar
    if progress:
        try:
            from tqdm.auto import tqdm
        except ImportError:
            from tqdm import tqdm
        pbar = tqdm(total=nt - 1, desc="Tracing trajectories", unit="ts")
    else:
        pbar = None

    # --- Forward integration ---
    _preload_batch(t0_idx, t0_idx + _BATCH)
    cx = x0s.copy()
    cy = y0s.copy()

    for ti in range(t0_idx, nt - 1):
        actual_dt = float(times[ti + 1] - times[ti])
        fields_ti = _get_fields(ti)

        fx = (cx - x0_grid) / dx_grid
        fy = (cy - y0_grid) / dy_grid
        vx, vy = _exb_velocity_batch(fields_ti, fx, fy, nx, ny)

        cx = x_min + (cx + vx * actual_dt - x_min) % Lx
        cy = y_min + (cy + vy * actual_dt - y_min) % Ly

        x_out[:, ti + 1] = cx
        y_out[:, ti + 1] = cy
        active_mask[:, ti + 1] = True

        if pbar is not None:
            pbar.update(1)

    # --- Backward integration ---
    _preload_batch(max(t0_idx - _BATCH + 1, 0), t0_idx + 1)
    cx = x0s.copy()
    cy = y0s.copy()

    for ti in range(t0_idx, 0, -1):
        actual_dt = float(times[ti] - times[ti - 1])
        fields_ti = _get_fields(ti, backward=True)

        fx = (cx - x0_grid) / dx_grid
        fy = (cy - y0_grid) / dy_grid
        vx, vy = _exb_velocity_batch(fields_ti, fx, fy, nx, ny)

        cx = x_min + (cx - vx * actual_dt - x_min) % Lx
        cy = y_min + (cy - vy * actual_dt - y_min) % Ly

        x_out[:, ti - 1] = cx
        y_out[:, ti - 1] = cy
        active_mask[:, ti - 1] = True

        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    t_traj = times.astype(np.float64)
    return x_out, y_out, t_traj, active_mask


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
    x_all, y_all, t_traj, mask = trace_trajectories(
        ds, np.array([x0]), np.array([y0]), t0_idx, progress=progress,
    )
    return x_all[0], y_all[0], t_traj


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
    nx = len(x_vals)
    ny = len(y_vals)
    dx_grid = float(x_vals[1] - x_vals[0])
    dy_grid = float(y_vals[1] - y_vals[0])
    x0_grid = float(x_vals[0])
    y0_grid = float(y_vals[0])

    t_indices = np.searchsorted(t_vals, t_traj)
    t_indices = np.clip(t_indices, 0, len(t_vals) - 1)

    field_data = da.variable.data
    _is_dask = hasattr(field_data, "dask")
    _BATCH = 64
    _batch_cache: dict = {}

    def _preload_field_batch(start: int, end: int) -> None:
        _batch_cache.clear()
        end = min(end, len(t_vals))
        if _is_dask:
            import dask
            results = dask.compute(*[field_data[ti] for ti in range(start, end)])
            for idx, ti in enumerate(range(start, end)):
                _batch_cache[ti] = results[idx]
        else:
            for ti in range(start, end):
                _batch_cache[ti] = np.asarray(field_data[ti])

    values = np.empty(len(t_traj))

    for k in range(len(t_traj)):
        ti = t_indices[k]
        if ti not in _batch_cache:
            _preload_field_batch(ti, ti + _BATCH)
        field_2d = _batch_cache[ti]
        fx = np.array([(x_traj[k] - x0_grid) / dx_grid])
        fy = np.array([(y_traj[k] - y0_grid) / dy_grid])
        values[k] = _bilinear_interp_batch(field_2d, fx, fy, nx, ny)[0]

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


def sample_along_trajectories(
    ds: xr.Dataset,
    field: str,
    x_traj: np.ndarray,
    y_traj: np.ndarray,
    t_traj: np.ndarray,
    active_mask: np.ndarray,
) -> xr.DataArray:
    """Interpolate a field along N trajectories simultaneously.

    Uses vectorized bilinear interpolation at each timestep rather than
    building a full 3-D RegularGridInterpolator.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing *field*.
    field : str
        Variable name, e.g. "Bz".
    x_traj : ndarray, shape (N, n_times)
        x-positions (NaN where inactive).
    y_traj : ndarray, shape (N, n_times)
        y-positions (NaN where inactive).
    t_traj : ndarray, shape (n_times,)
        Shared time axis.
    active_mask : ndarray, shape (N, n_times), dtype bool
        True where trajectory is inside the domain.

    Returns
    -------
    xr.DataArray, shape (N, n_times)
        Sampled values with NaN where inactive. Dims: ("trajectory", "time").
    """
    if field not in ds:
        raise KeyError(f"Field '{field}' not in dataset")

    da = ds[field]
    x_vals = da.coords["x"].values
    y_vals = da.coords["y"].values
    t_vals = da.coords["time"].values
    nx = len(x_vals)
    ny = len(y_vals)
    dx_grid = float(x_vals[1] - x_vals[0])
    dy_grid = float(y_vals[1] - y_vals[0])
    x0_grid = float(x_vals[0])
    y0_grid = float(y_vals[0])

    N, nt = x_traj.shape
    out = np.full((N, nt), np.nan)

    # Map t_traj values to integer time indices
    t_indices = np.searchsorted(t_vals, t_traj)
    t_indices = np.clip(t_indices, 0, len(t_vals) - 1)

    field_data = da.variable.data
    _is_dask = hasattr(field_data, "dask")
    _BATCH = 64

    # Batch-load timesteps to avoid one-at-a-time dask computes
    _batch_cache: dict = {}

    def _preload_field_batch(start: int, end: int) -> None:
        _batch_cache.clear()
        end = min(end, len(t_vals))
        if _is_dask:
            import dask
            results = dask.compute(*[field_data[ti] for ti in range(start, end)])
            for idx, ti in enumerate(range(start, end)):
                _batch_cache[ti] = results[idx]
        else:
            for ti in range(start, end):
                _batch_cache[ti] = np.asarray(field_data[ti])

    for col in range(nt):
        active = active_mask[:, col]
        if not active.any():
            continue
        ti = t_indices[col]
        if ti not in _batch_cache:
            _preload_field_batch(ti, ti + _BATCH)
        field_2d = _batch_cache[ti]
        fx = (x_traj[active, col] - x0_grid) / dx_grid
        fy = (y_traj[active, col] - y0_grid) / dy_grid
        out[active, col] = _bilinear_interp_batch(field_2d, fx, fy, nx, ny)

    return xr.DataArray(
        out,
        dims=("trajectory", "time"),
        coords={"time": t_traj},
    )


def lagrangian_psds(
    ds: xr.Dataset,
    field: str,
    x0s: np.ndarray,
    y0s: np.ndarray,
    t0_idx: int,
    dt: Optional[float] = None,
    method: str = "welch",
    *,
    progress: bool = False,
    **psd_kwargs,
) -> List[Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]]:
    """Compute Lagrangian PSDs for N trajectories in batch.

    Traces all trajectories simultaneously, samples the field along each,
    then computes per-trajectory PSDs.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with all six E/B components plus the target *field*.
    field : str
        Variable name to analyse (e.g. "Bz").
    x0s, y0s : array_like, shape (N,)
        Starting positions for each trajectory.
    t0_idx : int
        Starting timestep index.
    dt : float, optional
        Time spacing for PSD.  If *None*, inferred from dataset.
    method : str
        "welch" or "fft"; passed to ``compute_psd_time``.
    progress : bool
        If True, show progress bar during tracing.
    **psd_kwargs
        Extra keyword arguments forwarded to ``compute_psd_time``.

    Returns
    -------
    list of (f, Pxx, trajectory)
        One entry per trajectory, same format as ``lagrangian_psd``.
    """
    x0s = np.asarray(x0s, dtype=np.float64)
    y0s = np.asarray(y0s, dtype=np.float64)
    N = len(x0s)

    x_all, y_all, t_traj, mask = trace_trajectories(
        ds, x0s, y0s, t0_idx, progress=progress,
    )
    sampled = sample_along_trajectories(ds, field, x_all, y_all, t_traj, mask)

    if dt is None:
        dt = float(np.median(np.diff(ds["time"].values)))

    results = []
    for i in range(N):
        valid = mask[i]
        t_valid = t_traj[valid]
        vals = sampled.values[i, valid]
        ts_da = xr.DataArray(vals, dims=("time",), coords={"time": t_valid})
        f, Pxx = compute_psd_time(ts_da, dt=dt, method=method, **psd_kwargs)
        traj = {
            "x_traj": x_all[i, valid],
            "y_traj": y_all[i, valid],
            "t_traj": t_valid,
        }
        results.append((f, Pxx, traj))

    return results
