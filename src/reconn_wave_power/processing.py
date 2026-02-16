"""
Preprocessing utilities: detrend, window, resample, mask, unit conversion.
"""
import numpy as np
import xarray as xr
from scipy import signal


def detrend_inplace(da: xr.DataArray, type: str = "linear") -> xr.DataArray:
    """Detrend along 'time' coordinate by default."""
    axis = da.get_axis_num("time") if "time" in da.dims else -1
    return xr.apply_ufunc(
        signal.detrend, da, kwargs={"type": type}, input_core_dims=[["time"]], output_core_dims=[["time"]]
    )


def compute_flux_function(ds: xr.Dataset, time_idx: int | None = None) -> xr.DataArray:
    """Compute magnetic flux function ψ from Bx and By.

    Uses a two-step integration consistent with Bx = ∂ψ/∂y and
    By = -∂ψ/∂x:

    1. Boundary at x=0: ψ(0, y) = ∫₀ʸ Bx(0, y') dy'
    2. Interior:         ψ(x, y) = ψ(0, y) - ∫₀ˣ By(x', y) dx'

    Parameters
    ----------
    ds : xr.Dataset
        Must contain "Bx" and "By" with dims including "x" and "y".
        Typical shapes are (time, x, y) or (x, y).
    time_idx : int, optional
        If provided, select a single timestep before integrating.

    Returns
    -------
    xr.DataArray
        Flux function with same spatial dims as input, named "psi".
    """
    bx = ds["Bx"]
    by = ds["By"]
    if time_idx is not None:
        bx = bx.isel(time=time_idx)
        by = by.isel(time=time_idx)

    bx_vals = bx.values  # (..., x, y)
    by_vals = by.values
    x = bx.coords["x"].values
    y = bx.coords["y"].values
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    psi = np.zeros_like(bx_vals)
    # Step 1: boundary at x=0 — integrate Bx along y
    psi[..., 0, 1:] = np.cumsum(bx_vals[..., 0, 1:], axis=-1) * dy
    # Step 2: interior — extend using -∂ψ/∂x = By
    psi[..., 1:, :] = psi[..., 0:1, :] - np.cumsum(by_vals[..., 1:, :], axis=-2) * dx

    return xr.DataArray(psi, coords=bx.coords, dims=bx.dims, name="psi")