"""
I/O wrappers to convert dhybridrpy simulation outputs into xarray Datasets.

Usage:
  from dhybridrpy import DHybridrpy
  dpy = DHybridrpy(input_file="path", output_folder="Output", lazy=True)
  ds = read_simulation(dpy, timesteps=[1,2], fields=("B","E"))
"""
from typing import Iterable, List, Optional, Sequence, Union, Tuple, Dict
import warnings
import numpy as np
import xarray as xr

# Import dhybridrpy optionally; tests can inject a fake module into sys.modules
try:
    from dhybridrpy import DHybridrpy  # type: ignore
except Exception:
    DHybridrpy = None  # type: ignore


def _component_names_for_group(group: str) -> List[str]:
    g = group.lower()
    if g in ("b", "magnetic", "mag", "B"):
        return ["Bx", "By", "Bz"]
    if g in ("e", "electric", "E"):
        return ["Ex", "Ey", "Ez"]
    # if explicit component names passed, return as-is split on commas/spaces
    if "," in group:
        return [s.strip() for s in group.split(",")]
    return [group]


def _infer_spatial_dims_and_coords(field) -> Tuple[Tuple[str, ...], Dict[str, object]]:
    """Infer dims and coords from Field object (xdata, ydata, zdata)."""
    data = getattr(field, "data", None)
    if data is None:
        raise ValueError("Field object has no .data attribute")

    # Access coordinate properties safely - they may raise IndexError for lower-dim data
    try:
        xd = field.xdata
    except (AttributeError, IndexError):
        xd = None
    try:
        yd = field.ydata
    except (AttributeError, IndexError):
        yd = None
    try:
        zd = field.zdata
    except (AttributeError, IndexError):
        zd = None

    # Determine rank from available coords or data.ndim
    ndim = getattr(data, "ndim", None)
    if ndim is None:
        # fallback heuristics
        if xd is not None and yd is None:
            ndim = 1
        elif xd is not None and yd is not None and zd is None:
            ndim = 2
        elif zd is not None:
            ndim = 3

    coords: Dict[str, object] = {}
    if ndim == 1:
        dims = ("x",)
        if xd is not None:
            coords["x"] = xd
    elif ndim == 2:
        dims = ("x", "y")
        if xd is not None:
            coords["x"] = xd
        if yd is not None:
            coords["y"] = yd
    elif ndim == 3:
        dims = ("x", "y", "z")
        if xd is not None:
            coords["x"] = xd
        if yd is not None:
            coords["y"] = yd
        if zd is not None:
            coords["z"] = zd
    else:
        # fallback to indexed dims
        shape = getattr(data, "shape", ())
        dims = tuple(f"dim_{i}" for i in range(len(shape)))
        for i, s in enumerate(shape):
            coords[dims[i]] = np.arange(s)
        warnings.warn(f"Could not infer standard spatial dims; using {dims}")

    return dims, coords


def _field_to_dataarray(field, time_coord: Optional[float] = None) -> xr.DataArray:
    """Convert a dhybridrpy Field object into an xarray.DataArray."""
    data = field.data
    dims, coords = _infer_spatial_dims_and_coords(field)

    xr_coords = {}
    for d in dims:
        if d in coords:
            xr_coords[d] = coords[d]
        else:
            # integer index
            xr_coords[d] = np.arange(getattr(data, "shape", ())[dims.index(d)])

    if time_coord is not None:
        # add a leading time axis
        arr = xr.DataArray(data[np.newaxis, ...], dims=("time",) + tuple(dims), coords={"time": [time_coord], **xr_coords})
    else:
        arr = xr.DataArray(data, dims=tuple(dims), coords=xr_coords)

    # copy a few metadata attributes if present
    for attr in ("name", "type", "timestep", "time"):
        if hasattr(field, attr):
            arr.attrs[attr] = getattr(field, attr)

    return arr


def _ensure_dhybridrpy_instance(source, input_file: Optional[str], output_folder: Optional[str], lazy: bool):
    """Return a DHybridrpy instance from a source or construct one from file paths."""
    if DHybridrpy is None:
        raise ImportError("dhybridrpy is required for reading simulation data. Install it first.")

    if source is None:
        if input_file is None or output_folder is None:
            raise ValueError("Either pass an initialized DHybridrpy instance as 'source' or provide input_file and output_folder.")
        return DHybridrpy(input_file=input_file, output_folder=output_folder, lazy=lazy)

    if isinstance(source, DHybridrpy):
        return source

    if isinstance(source, (tuple, list)) and len(source) == 2:
        return DHybridrpy(input_file=source[0], output_folder=source[1], lazy=lazy)

    raise ValueError("source must be a DHybridrpy instance or (input_file, output_folder) tuple")


def read_simulation(
    source: Union[None, object, Tuple[str, str]] = None,
    *,
    input_file: Optional[str] = None,
    output_folder: Optional[str] = None,
    timesteps: Optional[Sequence[int]] = None,
    fields: Sequence[str] = ("E", "B"),
    field_type: str = "Total",
    lazy: bool = True,
) -> xr.Dataset:
    """
    Read simulation data (electric and magnetic fields) and return an xarray.Dataset.

    Parameters
    ----------
    source:
        DHybridrpy instance or (input_file, output_folder) tuple. If None, input_file/output_folder must be set.
    timesteps:
        Sequence of integer timesteps to read. If None, read all available timesteps.
    fields:
        Which groups to read: e.g., ("E","B") or explicit components like ("Bx", "By").
    field_type:
        "Total", "External", or "Self"
    lazy:
        Pass-through to constructor if we instantiate DHybridrpy here.
    """
    dpy = _ensure_dhybridrpy_instance(source, input_file, output_folder, lazy)

    available_ts = list(dpy.timesteps())
    if timesteps is None:
        timesteps = available_ts
    else:
        missing = [t for t in timesteps if t not in available_ts]
        if missing:
            raise ValueError(f"Requested timesteps {missing} not in available timesteps {available_ts}")

    # Expand groups to component names
    requested_components: List[str] = []
    for g in fields:
        requested_components.extend(_component_names_for_group(g))
    # deduplicate
    seen = set()
    comps = []
    for c in requested_components:
        if c not in seen:
            comps.append(c)
            seen.add(c)

    var_arrays: Dict[str, xr.DataArray] = {}
    for ts in timesteps:
        ts_obj = dpy.timestep(ts)
        time_val = getattr(ts_obj, "time", None)
        if time_val is None:
            try:
                dt = float(dpy.inputs["time"]["dt"])
                time_val = dt * ts
            except Exception:
                time_val = float(ts)

        for comp in comps:
            fields_api = getattr(ts_obj, "fields")
            getter = getattr(fields_api, comp, None)
            if getter is None:
                try:
                    field = fields_api(comp)  # type: ignore
                except Exception:
                    raise AttributeError(f"Could not access field component '{comp}' on timestep {ts}")
            else:
                try:
                    field = getter(type=field_type)
                except TypeError:
                    field = getter()

            da = _field_to_dataarray(field, time_coord=time_val)
            if comp not in var_arrays:
                var_arrays[comp] = da
            else:
                var_arrays[comp] = xr.concat([var_arrays[comp], da], dim="time")

    ds = xr.Dataset()
    for comp, da in var_arrays.items():
        ds[comp] = da

    # attach sampling attrs if available
    try:
        if "time" in dpy.inputs and "dt" in dpy.inputs["time"]:
            ds.attrs["dt"] = float(dpy.inputs["time"]["dt"])
    except Exception:
        pass
    try:
        grid = dpy.inputs.get("grid", {})
        for k in ("dx", "dy", "dz"):
            if k in grid:
                ds.attrs[k] = float(grid[k])
    except Exception:
        pass

    return ds