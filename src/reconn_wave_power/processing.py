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