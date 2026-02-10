"""
Lagrangian frame frequency spectra.

Trace trajectories following the local E×B drift velocity and compute
frequency PSDs along those trajectories (plasma rest-frame spectra).

APIs:
- compute_exb_velocity(ds, time_idx=None) -> vx, vy
- trace_trajectory(ds, x0, y0, t0_idx, dt) -> x_traj, y_traj, t_traj
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
    dt: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Integrate a trajectory through the E×B velocity field.

    Steps forward and backward from (x0, y0) at timestep index *t0_idx*
    using simple Euler stepping at the simulation cadence *dt*.  At each
    timestep the local E×B velocity is obtained by spatial interpolation
    (scipy RegularGridInterpolator).  The trajectory is truncated when it
    exits the simulation domain.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with all six E/B components and dimensions (time, x, y).
    x0, y0 : float
        Starting position.
    t0_idx : int
        Timestep index of the starting point (index into ds.time).
    dt : float
        Time between consecutive timesteps.

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

    # Extract raw field arrays (no ExB computation over the full grid)
    Ex = ds["Ex"].values
    Ey = ds["Ey"].values
    Ez = ds["Ez"].values
    Bx = ds["Bx"].values
    By = ds["By"].values
    Bz = ds["Bz"].values

    # Grid spacing for O(1) index lookup
    dx = float(x_vals[1] - x_vals[0])
    dy = float(y_vals[1] - y_vals[0])
    x0_grid = float(x_vals[0])
    y0_grid = float(y_vals[0])
    nx = len(x_vals)
    ny = len(y_vals)

    def _interpolated_velocity(ti: int, x: float, y: float) -> Tuple[float, float]:
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

        # Extract the 2x2 patch from each field component
        sl = (ti, slice(i0, i0 + 2), slice(j0, j0 + 2))
        ex = Ex[sl]; ey = Ey[sl]; ez = Ez[sl]
        bx = Bx[sl]; by = By[sl]; bz = Bz[sl]

        # ExB / |B|^2 at the 4 grid points
        b2 = bx**2 + by**2 + bz**2
        vx_patch = (ey * bz - ez * by) / b2
        vy_patch = (ez * bx - ex * bz) / b2

        vx = vx_patch[0, 0] * w00 + vx_patch[1, 0] * w10 + vx_patch[0, 1] * w01 + vx_patch[1, 1] * w11
        vy = vy_patch[0, 0] * w00 + vy_patch[1, 0] * w10 + vy_patch[0, 1] * w01 + vy_patch[1, 1] * w11
        return float(vx), float(vy)

    # --- Forward integration (t0_idx -> end) ---
    fwd_x, fwd_y, fwd_t = [x0], [y0], [float(times[t0_idx])]
    cx, cy = x0, y0
    for ti in range(t0_idx, nt - 1):
        vx_loc, vy_loc = _interpolated_velocity(ti, cx, cy)
        cx_new = cx + vx_loc * dt
        cy_new = cy + vy_loc * dt
        if not _in_domain(cx_new, cy_new):
            break
        cx, cy = cx_new, cy_new
        fwd_x.append(cx)
        fwd_y.append(cy)
        fwd_t.append(float(times[ti + 1]))

    # --- Backward integration (t0_idx -> 0) ---
    bwd_x, bwd_y, bwd_t = [], [], []
    cx, cy = x0, y0
    for ti in range(t0_idx, 0, -1):
        vx_loc, vy_loc = _interpolated_velocity(ti, cx, cy)
        cx_new = cx - vx_loc * dt
        cy_new = cy - vy_loc * dt
        if not _in_domain(cx_new, cy_new):
            break
        cx, cy = cx_new, cy_new
        bwd_x.append(cx)
        bwd_y.append(cy)
        bwd_t.append(float(times[ti - 1]))

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
    dt: float,
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
    dt : float
        Time between consecutive timesteps.
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
    x_traj, y_traj, t_traj = trace_trajectory(ds, x0, y0, t0_idx, dt)
    ts = sample_along_trajectory(ds, field, x_traj, y_traj, t_traj)
    f, Pxx = compute_psd_time(ts, dt=dt, method=method, **psd_kwargs)
    trajectory = {"x_traj": x_traj, "y_traj": y_traj, "t_traj": t_traj}
    return f, Pxx, trajectory
