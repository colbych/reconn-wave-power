"""
I/O wrappers to convert dhybridrpy simulation outputs into xarray Datasets.

Usage:
  from dhybridrpy import DHybridrpy
  dpy = DHybridrpy(input_file="path", output_folder="Output", lazy=True)
  ds = read_simulation(dpy, timesteps=[1,2], fields=("B","E"))
  ds_sub = read_simulation_subregion(dpy, x_range=(2.0, 5.0), y_range=(1.0, 3.0))
"""
from typing import List, Optional, Sequence, Union, Tuple, Dict
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
    progress: bool = False,
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
    progress:
        If True, display a progress bar while loading timesteps.
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

    # Wrap timesteps with progress bar if requested
    if progress:
        try:
            from tqdm.auto import tqdm
            timestep_iter = tqdm(timesteps, desc="Loading timesteps", unit="ts")
        except ImportError:
            warnings.warn("tqdm not installed; progress bar disabled. Install with: pip install tqdm")
            timestep_iter = timesteps
    else:
        timestep_iter = timesteps

    for ts in timestep_iter:
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

    _attach_sim_attrs(ds, dpy)
    return ds


def _attach_sim_attrs(ds: xr.Dataset, dpy) -> None:
    """Attach dt, ndump, dt_output, and grid spacing attrs from dpy.inputs.

    dt_output = dt * ndump is the time between consecutive output snapshots
    and should be used as the sample spacing when computing FFT/PSD frequency
    axes.  dt is the underlying simulation timestep and is retained for
    reference.
    """
    try:
        dt = float(dpy.inputs["time"]["dt"])
        ds.attrs["dt"] = dt
    except Exception:
        dt = None

    try:
        ndump = int(dpy.inputs["global_output"]["ndump"])
        ds.attrs["ndump"] = ndump
    except Exception:
        ndump = None

    if dt is not None and ndump is not None:
        ds.attrs["dt_output"] = dt * ndump

    try:
        grid = dpy.inputs.get("grid", {})
        for k in ("dx", "dy", "dz"):
            if k in grid:
                ds.attrs[k] = float(grid[k])
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Subregion I/O — reads only a spatial slice from disk via h5py
# ---------------------------------------------------------------------------

def _get_grid_info(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read coordinate arrays from one HDF5 field file.

    Uses the same cell-centred formula as dhybridrpy:
        delta = (lims[1] - lims[0]) / n
        coords = delta * arange(n) + delta/2 + lims[0]

    Parameters
    ----------
    file_path : str
        Path to a dHybridR HDF5 field file.

    Returns
    -------
    x_coords, y_coords : ndarray
        Full 1-D coordinate arrays for the simulation grid.
    """
    import h5py
    with h5py.File(file_path, "r") as f:
        x_lims = f["AXIS"]["X1 AXIS"][:]
        y_lims = f["AXIS"]["X2 AXIS"][:]
        ny, nx = f["DATA"].shape  # on disk: (ny, nx); .T gives (nx, ny)
    dx = (x_lims[1] - x_lims[0]) / nx
    dy = (y_lims[1] - y_lims[0]) / ny
    x_coords = dx * np.arange(nx) + dx / 2.0 + x_lims[0]
    y_coords = dy * np.arange(ny) + dy / 2.0 + y_lims[0]
    return x_coords, y_coords


def _read_hdf5_subregion(
    file_path: str,
    i0: int,
    i1: int,
    j0: int,
    j1: int,
) -> np.ndarray:
    """Read a spatial subregion from one HDF5 field file.

    DATA is stored on disk as (ny, nx); after ``.T`` it becomes (nx, ny).
    x-indices therefore map to axis 1 in the file, y-indices to axis 0.

    Parameters
    ----------
    file_path : str
        Path to a dHybridR HDF5 field file.
    i0, i1 : int
        Start and end indices along the x axis (exclusive end).
    j0, j1 : int
        Start and end indices along the y axis (exclusive end).

    Returns
    -------
    ndarray, shape (i1-i0, j1-j0)
    """
    import h5py
    with h5py.File(file_path, "r") as f:
        return np.array(f["DATA"][j0:j1, i0:i1]).T


def read_simulation_subregion(
    source: Union[None, object, Tuple[str, str]] = None,
    *,
    input_file: Optional[str] = None,
    output_folder: Optional[str] = None,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    timesteps: Optional[Sequence[int]] = None,
    fields: Sequence[str] = ("E", "B"),
    field_type: str = "Total",
    progress: bool = False,
) -> xr.Dataset:
    """Read a spatial subregion of the simulation, loading only that slice from disk.

    Each timestep's HDF5 file is opened and only the requested spatial window is
    read via h5py hyperslab selection, avoiding the cost of loading the full
    spatial domain.  Suitable for large simulations where loading the complete
    field arrays is memory-prohibitive.

    Parameters
    ----------
    source :
        DHybridrpy instance or (input_file, output_folder) tuple.  If None,
        ``input_file`` and ``output_folder`` keyword args must be provided.
    input_file, output_folder : str, optional
        Used only when *source* is None.
    x_range : (float, float)
        Physical x coordinates of the subregion ``(x_min, x_max)``.  All grid
        cells whose centre falls within this range are included.
    y_range : (float, float)
        Physical y coordinates of the subregion ``(y_min, y_max)``.
    timesteps : sequence of int, optional
        Which timesteps to load.  Defaults to all available timesteps.
    fields : sequence of str
        Field groups or component names to load (e.g. ``("E", "B")``).
    field_type : str
        ``"Total"``, ``"External"``, or ``"Self"``.
    progress : bool
        If True, display a tqdm progress bar.

    Returns
    -------
    xr.Dataset
        Dataset with dimensions ``(time, x, y)`` spanning only the requested
        subregion.  Coordinate arrays and ``.attrs`` match those from
        :func:`read_simulation`.
    """
    dpy = _ensure_dhybridrpy_instance(source, input_file, output_folder, lazy=False)

    available_ts = list(dpy.timesteps())
    if timesteps is None:
        timesteps = available_ts
    else:
        missing = [t for t in timesteps if t not in available_ts]
        if missing:
            raise ValueError(
                f"Requested timesteps {missing} not in available timesteps {available_ts}"
            )

    # Expand field groups to component names — identical to read_simulation
    requested_components: List[str] = []
    for g in fields:
        requested_components.extend(_component_names_for_group(g))
    seen: set = set()
    comps: List[str] = []
    for c in requested_components:
        if c not in seen:
            comps.append(c)
            seen.add(c)

    # Helper: retrieve a field object from a timestep object
    def _get_field_obj(ts_obj, comp):
        fapi = getattr(ts_obj, "fields")
        getter = getattr(fapi, comp, None)
        if getter is None:
            try:
                return fapi(comp)
            except Exception:
                raise AttributeError(
                    f"Could not access field component '{comp}'"
                )
        try:
            return getter(type=field_type)
        except TypeError:
            return getter()

    # Read grid coordinates once from the first available field file
    probe_ts_obj = dpy.timestep(timesteps[0])
    probe_field = _get_field_obj(probe_ts_obj, comps[0])
    x_coords, y_coords = _get_grid_info(probe_field.file_path)

    # Validate ranges
    x_min_d = float(x_coords[0])
    x_max_d = float(x_coords[-1])
    y_min_d = float(y_coords[0])
    y_max_d = float(y_coords[-1])

    if x_range[0] >= x_range[1]:
        raise ValueError(
            f"x_range must satisfy x_range[0] < x_range[1], got {x_range}"
        )
    if y_range[0] >= y_range[1]:
        raise ValueError(
            f"y_range must satisfy y_range[0] < y_range[1], got {y_range}"
        )
    if x_range[1] <= x_min_d or x_range[0] >= x_max_d:
        raise ValueError(
            f"x_range {x_range} is entirely outside the domain "
            f"x ∈ [{x_min_d:.4g}, {x_max_d:.4g}]"
        )
    if y_range[1] <= y_min_d or y_range[0] >= y_max_d:
        raise ValueError(
            f"y_range {y_range} is entirely outside the domain "
            f"y ∈ [{y_min_d:.4g}, {y_max_d:.4g}]"
        )

    # Compute integer index bounds.
    # Include all cells whose centre falls within [range[0], range[1]].
    i0 = int(np.searchsorted(x_coords, x_range[0], side="left"))
    i1 = int(np.searchsorted(x_coords, x_range[1], side="right"))
    j0 = int(np.searchsorted(y_coords, y_range[0], side="left"))
    j1 = int(np.searchsorted(y_coords, y_range[1], side="right"))

    i0 = max(0, i0)
    i1 = min(len(x_coords), i1)
    j0 = max(0, j0)
    j1 = min(len(y_coords), j1)

    if i1 <= i0:
        raise ValueError(
            f"x_range {x_range} maps to an empty subregion (x indices {i0}:{i1})"
        )
    if j1 <= j0:
        raise ValueError(
            f"y_range {y_range} maps to an empty subregion (y indices {j0}:{j1})"
        )

    sub_x = x_coords[i0:i1]
    sub_y = y_coords[j0:j1]

    # Progress bar
    if progress:
        try:
            from tqdm.auto import tqdm
            timestep_iter = tqdm(timesteps, desc="Loading subregion", unit="ts")
        except ImportError:
            warnings.warn(
                "tqdm not installed; progress bar disabled. "
                "Install with: pip install tqdm"
            )
            timestep_iter = timesteps
    else:
        timestep_iter = timesteps

    # Main loading loop
    var_arrays: Dict[str, xr.DataArray] = {}

    for ts in timestep_iter:
        ts_obj = dpy.timestep(ts)
        time_val = getattr(ts_obj, "time", None)
        if time_val is None:
            try:
                dt = float(dpy.inputs["time"]["dt"])
                time_val = dt * ts
            except Exception:
                time_val = float(ts)

        for comp in comps:
            field_obj = _get_field_obj(ts_obj, comp)
            data_2d = _read_hdf5_subregion(field_obj.file_path, i0, i1, j0, j1)
            da = xr.DataArray(
                data_2d[np.newaxis, ...],
                dims=("time", "x", "y"),
                coords={"time": [time_val], "x": sub_x, "y": sub_y},
            )
            if comp not in var_arrays:
                var_arrays[comp] = da
            else:
                var_arrays[comp] = xr.concat([var_arrays[comp], da], dim="time")

    ds = xr.Dataset(var_arrays)

    _attach_sim_attrs(ds, dpy)
    return ds