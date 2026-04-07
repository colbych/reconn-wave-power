"""
Tests for read_simulation_subregion (io.py) and compute_psd_map (spectrum.py).

Strategy
--------
* Private helpers (_get_grid_info, _read_hdf5_subregion) are tested directly
  against real HDF5 files created in a tmp_path fixture — no dhybridrpy needed.

* read_simulation_subregion is tested via a MockDHybridrpy that subclasses
  whatever DHybridrpy class io.py resolved at import time (real or fake).
  The mock returns field objects whose .file_path points to real HDF5 files,
  so the full h5py read path is exercised.

* compute_psd_map is tested purely on in-memory DataArrays.
"""
import os
import numpy as np
import xarray as xr
import pytest
import h5py

import reconn_wave_power.io as _rw_io
from reconn_wave_power.io import (
    _get_grid_info,
    _read_hdf5_subregion,
    read_simulation_subregion,
)
from reconn_wave_power.spectrum import compute_psd_map, compute_psd_time

# ---------------------------------------------------------------------------
# Shared grid constants
# ---------------------------------------------------------------------------
NX, NY = 20, 10
N_TS = 5
X_LIMS = [0.0, 10.0]
Y_LIMS = [0.0, 5.0]
DT = 0.1
COMPS = ["Bx", "By", "Bz", "Ex", "Ey", "Ez"]

# Derived grid coords (mirrors dhybridrpy's cell-centred formula)
_dx = (X_LIMS[1] - X_LIMS[0]) / NX
_dy = (Y_LIMS[1] - Y_LIMS[0]) / NY
FULL_X = _dx * np.arange(NX) + _dx / 2 + X_LIMS[0]  # shape (20,)
FULL_Y = _dy * np.arange(NY) + _dy / 2 + Y_LIMS[0]  # shape (10,)


# ---------------------------------------------------------------------------
# HDF5 helpers
# ---------------------------------------------------------------------------

def _write_field_hdf5(path: str, data_nx_ny: np.ndarray) -> None:
    """Write a minimal dHybridR-style HDF5 field file.

    DATA is stored as (ny, nx) — the transpose of the (nx, ny) logical array.
    AXIS groups hold the coordinate limits.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("DATA", data=data_nx_ny.astype(np.float32).T)
        ax = f.create_group("AXIS")
        ax.create_dataset("X1 AXIS", data=np.array(X_LIMS, dtype=np.float64))
        ax.create_dataset("X2 AXIS", data=np.array(Y_LIMS, dtype=np.float64))


# ---------------------------------------------------------------------------
# Mock DHybridrpy for integration tests
# ---------------------------------------------------------------------------

# Subclass whatever class io.py resolved so isinstance() checks pass.
_DHyBase = _rw_io.DHybridrpy


class _MockField:
    """Minimal field object: only .file_path is needed by read_simulation_subregion."""
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path


class _MockFieldsAPI:
    def __init__(self, paths_for_ts: dict) -> None:
        self._paths = paths_for_ts  # {comp: path}

    def __getattr__(self, comp: str):
        def getter(type="Total"):
            if comp not in self._paths:
                raise AttributeError(f"No field '{comp}'")
            return _MockField(self._paths[comp])
        return getter


class _MockTimestep:
    def __init__(self, ts: int, paths_for_ts: dict, dt: float) -> None:
        self.timestep = ts
        self.time = dt * ts
        self.fields = _MockFieldsAPI(paths_for_ts)


class _MockDHybridrpy(_DHyBase):
    """DHybridrpy stand-in backed by real HDF5 files in tmp_path."""

    def __init__(self, all_file_paths: dict, ts_list: list, dt: float = DT) -> None:
        # all_file_paths: {(comp, ts): path_str}
        # Do NOT call super().__init__() — avoids real/fake constructor side-effects.
        self._all_paths = all_file_paths
        self._ts_list = ts_list
        self.inputs = {
            "time": {"dt": dt, "t0": 0.0},
            "grid": {"dx": _dx, "dy": _dy},
        }

    def timesteps(self):
        return self._ts_list

    def timestep(self, ts: int):
        paths = {comp: path for (comp, ts_), path in self._all_paths.items() if ts_ == ts}
        return _MockTimestep(ts, paths, self.inputs["time"]["dt"])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sim_data(tmp_path):
    """Create synthetic HDF5 files; return mock DHybridrpy + ground-truth arrays."""
    rng = np.random.default_rng(42)
    ts_list = list(range(1, N_TS + 1))

    arrays = {}       # {(comp, ts): ndarray(nx, ny)}
    file_paths = {}   # {(comp, ts): str}

    for ts in ts_list:
        for comp in COMPS:
            arr = rng.standard_normal((NX, NY)).astype(np.float32)
            arrays[(comp, ts)] = arr
            fpath = str(tmp_path / f"{comp}_{ts:06d}.h5")
            _write_field_hdf5(fpath, arr)
            file_paths[(comp, ts)] = fpath

    mock_dpy = _MockDHybridrpy(file_paths, ts_list)

    return {
        "mock_dpy": mock_dpy,
        "arrays": arrays,
        "file_paths": file_paths,
        "ts_list": ts_list,
        "first_file": file_paths[(COMPS[0], ts_list[0])],
    }


# ===========================================================================
# Tests for _get_grid_info
# ===========================================================================

def test_get_grid_info_coords(sim_data):
    """Returned coordinates match the cell-centred formula exactly."""
    x, y = _get_grid_info(sim_data["first_file"])
    np.testing.assert_allclose(x, FULL_X, rtol=1e-6)
    np.testing.assert_allclose(y, FULL_Y, rtol=1e-6)


def test_get_grid_info_shape(sim_data):
    """Coordinate arrays have the right length."""
    x, y = _get_grid_info(sim_data["first_file"])
    assert len(x) == NX
    assert len(y) == NY


# ===========================================================================
# Tests for _read_hdf5_subregion
# ===========================================================================

def test_read_hdf5_subregion_full(sim_data):
    """Reading the full domain recovers the original array."""
    fpath = sim_data["first_file"]
    comp, ts = COMPS[0], sim_data["ts_list"][0]
    expected = sim_data["arrays"][(comp, ts)]

    result = _read_hdf5_subregion(fpath, 0, NX, 0, NY)

    assert result.shape == (NX, NY)
    np.testing.assert_allclose(result, expected, atol=1e-6)


def test_read_hdf5_subregion_slice(sim_data):
    """A spatial slice matches the corresponding slice of the full array."""
    fpath = sim_data["first_file"]
    comp, ts = COMPS[0], sim_data["ts_list"][0]
    expected = sim_data["arrays"][(comp, ts)]

    i0, i1, j0, j1 = 5, 12, 2, 8
    result = _read_hdf5_subregion(fpath, i0, i1, j0, j1)

    assert result.shape == (i1 - i0, j1 - j0)
    np.testing.assert_allclose(result, expected[i0:i1, j0:j1], atol=1e-6)


# ===========================================================================
# Tests for read_simulation_subregion
# ===========================================================================

def test_subregion_all_fields_loaded(sim_data):
    """All 6 E/B components are present in the returned Dataset."""
    ds = read_simulation_subregion(
        sim_data["mock_dpy"],
        x_range=(2.0, 7.0),
        y_range=(1.0, 4.0),
    )
    for comp in COMPS:
        assert comp in ds, f"Missing component '{comp}'"


def test_subregion_time_axis(sim_data):
    """Time axis covers all timesteps with correct values."""
    ds = read_simulation_subregion(
        sim_data["mock_dpy"],
        x_range=(2.0, 7.0),
        y_range=(1.0, 4.0),
    )
    expected_times = [DT * ts for ts in sim_data["ts_list"]]
    np.testing.assert_allclose(ds["Bx"].coords["time"].values, expected_times, rtol=1e-6)


def test_subregion_coords_correct(sim_data):
    """Spatial coordinates are the correct subset of the full grid."""
    x_range = (2.0, 7.0)
    y_range = (1.0, 4.0)

    ds = read_simulation_subregion(
        sim_data["mock_dpy"],
        x_range=x_range,
        y_range=y_range,
    )

    # Expected index bounds
    i0 = int(np.searchsorted(FULL_X, x_range[0], side="left"))
    i1 = int(np.searchsorted(FULL_X, x_range[1], side="right"))
    j0 = int(np.searchsorted(FULL_Y, y_range[0], side="left"))
    j1 = int(np.searchsorted(FULL_Y, y_range[1], side="right"))

    np.testing.assert_allclose(ds["Bx"].coords["x"].values, FULL_X[i0:i1], rtol=1e-6)
    np.testing.assert_allclose(ds["Bx"].coords["y"].values, FULL_Y[j0:j1], rtol=1e-6)


def test_subregion_values_match_full_slice(sim_data):
    """Subregion values equal the corresponding slice of the full ground-truth arrays."""
    x_range = (3.0, 6.0)
    y_range = (0.5, 3.5)

    ds = read_simulation_subregion(
        sim_data["mock_dpy"],
        x_range=x_range,
        y_range=y_range,
    )

    i0 = int(np.searchsorted(FULL_X, x_range[0], side="left"))
    i1 = int(np.searchsorted(FULL_X, x_range[1], side="right"))
    j0 = int(np.searchsorted(FULL_Y, y_range[0], side="left"))
    j1 = int(np.searchsorted(FULL_Y, y_range[1], side="right"))

    for t_idx, ts in enumerate(sim_data["ts_list"]):
        expected = sim_data["arrays"][("Bz", ts)][i0:i1, j0:j1]
        result = ds["Bz"].isel(time=t_idx).values
        np.testing.assert_allclose(result, expected, atol=1e-6)


def test_subregion_metadata_attrs(sim_data):
    """Dataset carries dt and grid spacing attrs."""
    ds = read_simulation_subregion(
        sim_data["mock_dpy"],
        x_range=(2.0, 7.0),
        y_range=(1.0, 4.0),
    )
    assert "dt" in ds.attrs
    assert abs(ds.attrs["dt"] - DT) < 1e-9


def test_subregion_out_of_bounds_x_raises(sim_data):
    """x_range entirely outside the domain raises ValueError."""
    with pytest.raises(ValueError, match="x_range"):
        read_simulation_subregion(
            sim_data["mock_dpy"],
            x_range=(50.0, 60.0),
            y_range=(1.0, 4.0),
        )


def test_subregion_out_of_bounds_y_raises(sim_data):
    """y_range entirely outside the domain raises ValueError."""
    with pytest.raises(ValueError, match="y_range"):
        read_simulation_subregion(
            sim_data["mock_dpy"],
            x_range=(2.0, 7.0),
            y_range=(20.0, 30.0),
        )


def test_subregion_inverted_range_raises(sim_data):
    """x_range[0] >= x_range[1] raises ValueError."""
    with pytest.raises(ValueError, match="x_range"):
        read_simulation_subregion(
            sim_data["mock_dpy"],
            x_range=(7.0, 2.0),
            y_range=(1.0, 4.0),
        )


def test_subregion_subset_of_timesteps(sim_data):
    """Loading a subset of timesteps returns only those time points."""
    ts_subset = sim_data["ts_list"][:3]
    ds = read_simulation_subregion(
        sim_data["mock_dpy"],
        x_range=(2.0, 7.0),
        y_range=(1.0, 4.0),
        timesteps=ts_subset,
    )
    assert ds["Bx"].sizes["time"] == 3


# ===========================================================================
# Tests for compute_psd_map
# ===========================================================================

def _make_sinusoid_da(f0: float, dt: float, n_time: int, nx: int, ny: int) -> xr.DataArray:
    """DataArray with a pure sinusoid at every spatial point."""
    t = np.arange(n_time) * dt
    signal = np.sin(2 * np.pi * f0 * t)  # (n_time,)
    # broadcast to (n_time, nx, ny)
    data = np.broadcast_to(signal[:, None, None], (n_time, nx, ny)).copy()
    x = np.arange(nx, dtype=float)
    y = np.arange(ny, dtype=float)
    return xr.DataArray(
        data,
        dims=("time", "x", "y"),
        coords={"time": t, "x": x, "y": y},
    )


def test_psd_map_output_dims():
    """Result has dims (x, y, frequency) with correct sizes."""
    da = _make_sinusoid_da(f0=5.0, dt=0.01, n_time=256, nx=4, ny=3)
    result = compute_psd_map(da, dt=0.01, method="fft")

    assert result.dims == ("x", "y", "frequency")
    assert result.sizes["x"] == 4
    assert result.sizes["y"] == 3
    assert result.sizes["frequency"] == 256 // 2 + 1


def test_psd_map_coords_inherited():
    """x and y coordinates are inherited from the input DataArray."""
    da = _make_sinusoid_da(f0=5.0, dt=0.01, n_time=256, nx=4, ny=3)
    result = compute_psd_map(da, dt=0.01, method="fft")

    np.testing.assert_array_equal(result.coords["x"].values, da.coords["x"].values)
    np.testing.assert_array_equal(result.coords["y"].values, da.coords["y"].values)


def test_psd_map_frequency_axis_matches_psd_time():
    """Frequency axis matches the f returned by compute_psd_time on the same data."""
    dt = 0.01
    da = _make_sinusoid_da(f0=5.0, dt=dt, n_time=256, nx=4, ny=3)
    result = compute_psd_map(da, dt=dt, method="fft")

    # Compute reference frequency axis via compute_psd_time on a 1-D slice
    ts_1d = da.isel(x=0, y=0)
    f_ref, _ = compute_psd_time(ts_1d, dt=dt, method="fft")

    np.testing.assert_allclose(result.coords["frequency"].values, f_ref, rtol=1e-10)


def test_psd_map_values_match_pointwise():
    """PSD at each pixel matches a direct compute_psd_time call on that time series."""
    dt = 0.005
    da = _make_sinusoid_da(f0=10.0, dt=dt, n_time=512, nx=3, ny=4)
    result = compute_psd_map(da, dt=dt, method="fft")

    for ix in range(3):
        for iy in range(4):
            ts = da.isel(x=ix, y=iy)
            _, pxx_ref = compute_psd_time(ts, dt=dt, method="fft")
            np.testing.assert_allclose(
                result.isel(x=ix, y=iy).values,
                pxx_ref,
                rtol=1e-5,
                err_msg=f"Mismatch at pixel (x={ix}, y={iy})",
            )


def test_psd_map_known_frequency_peak():
    """PSD peaks at the correct frequency for every spatial pixel."""
    f0 = 8.0
    dt = 0.002
    da = _make_sinusoid_da(f0=f0, dt=dt, n_time=1024, nx=5, ny=5)
    result = compute_psd_map(da, dt=dt, method="fft")

    freqs = result.coords["frequency"].values
    peak_indices = result.values.argmax(axis=-1)  # (nx, ny)
    peak_freqs = freqs[peak_indices]

    np.testing.assert_allclose(peak_freqs, f0, atol=0.5)


def test_psd_map_welch_runs():
    """Welch method produces output with the same spatial shape."""
    da = _make_sinusoid_da(f0=5.0, dt=0.01, n_time=512, nx=3, ny=3)
    result = compute_psd_map(da, dt=0.01, method="welch", nperseg=128)

    assert result.dims == ("x", "y", "frequency")
    assert result.sizes["x"] == 3
    assert result.sizes["y"] == 3


def test_psd_map_missing_dim_raises():
    """Missing 'time' dimension raises ValueError."""
    da = xr.DataArray(
        np.ones((4, 3)),
        dims=("x", "y"),
        coords={"x": np.arange(4, dtype=float), "y": np.arange(3, dtype=float)},
    )
    with pytest.raises(ValueError, match="time"):
        compute_psd_map(da, dt=0.01)
