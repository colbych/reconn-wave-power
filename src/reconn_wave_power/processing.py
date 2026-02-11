"""
Preprocessing utilities: detrend, window, resample, mask, unit conversion.
"""
import numpy as np
import xarray as xr
from scipy import signal
from scipy.integrate import cumulative_trapezoid

def detrend_inplace(da: xr.DataArray, type: str = "linear") -> xr.DataArray:
    """Detrend along 'time' coordinate by default."""
    axis = da.get_axis_num("time") if "time" in da.dims else -1
    return xr.apply_ufunc(
        signal.detrend, da, kwargs={"type": type}, input_core_dims=[["time"]], output_core_dims=[["time"]]
    )


def compute_flux_function(ds: xr.Dataset, time_idx: int | None = None) -> xr.DataArray:
    """Compute magnetic flux function ψ from Bx via ∫ Bx dy.

    Parameters
    ----------
    ds : xr.Dataset
        Must contain "Bx" with dims including "y". Typical shapes are
        (time, x, y) or (x, y).
    time_idx : int, optional
        If provided, select a single timestep before integrating.

    Returns
    -------
    xr.DataArray
        Flux function with same spatial dims as input, named "psi".
    """
    bx = ds["Bx"]
    if time_idx is not None:
        bx = bx.isel(time=time_idx)

    y = bx.coords["y"].values
    y_axis = bx.get_axis_num("y")

    psi_vals = cumulative_trapezoid(bx.values, y, axis=y_axis, initial=0)

    return xr.DataArray(psi_vals, coords=bx.coords, dims=bx.dims, name="psi")