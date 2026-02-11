"""Tests for reconn_wave_power.processing."""
import numpy as np
import xarray as xr
import pytest

from reconn_wave_power.processing import compute_flux_function


def _make_bx_dataset(bx_vals, x, y, time=None):
    """Helper to build a Dataset with Bx from raw arrays."""
    if time is not None:
        ds = xr.Dataset(
            {"Bx": (["time", "x", "y"], bx_vals)},
            coords={"time": time, "x": x, "y": y},
        )
    else:
        ds = xr.Dataset(
            {"Bx": (["x", "y"], bx_vals)},
            coords={"x": x, "y": y},
        )
    return ds


class TestComputeFluxFunction:
    """Tests for compute_flux_function."""

    def test_analytic_cos(self):
        """Bx = cos(y) → ψ = sin(y), verified to trapezoidal accuracy."""
        y = np.linspace(0, 2 * np.pi, 200)
        x = np.array([0.0, 1.0])
        # Bx shape (x, y): constant along x, cos(y) along y
        bx_vals = np.cos(y)[np.newaxis, :] * np.ones((len(x), 1))
        ds = _make_bx_dataset(bx_vals, x, y)

        psi = compute_flux_function(ds)

        assert psi.name == "psi"
        assert psi.dims == ("x", "y")
        assert psi.shape == (len(x), len(y))
        # ψ(y) = sin(y) - sin(0) = sin(y)
        expected = np.sin(y)
        np.testing.assert_allclose(psi.values[0], expected, atol=1e-4)
        np.testing.assert_allclose(psi.values[1], expected, atol=1e-4)

    def test_3d_with_time(self):
        """Works on (time, x, y) arrays without time_idx."""
        y = np.linspace(0, np.pi, 100)
        x = np.array([0.0])
        time = np.array([0.0, 1.0])
        bx_vals = np.ones((2, 1, len(y)))  # constant Bx=1 → ψ = y
        ds = _make_bx_dataset(bx_vals, x, y, time=time)

        psi = compute_flux_function(ds)

        assert psi.dims == ("time", "x", "y")
        np.testing.assert_allclose(psi.values[0, 0], y, atol=1e-10)

    def test_time_idx_selects_single_frame(self):
        """time_idx reduces output to (x, y)."""
        y = np.linspace(0, 1, 50)
        x = np.array([0.0])
        time = np.array([0.0, 1.0, 2.0])
        bx_vals = np.ones((3, 1, len(y)))
        ds = _make_bx_dataset(bx_vals, x, y, time=time)

        psi = compute_flux_function(ds, time_idx=1)

        assert psi.dims == ("x", "y")
        assert "time" not in psi.dims

    def test_coordinates_preserved(self):
        """Output coordinates match input spatial coordinates."""
        y = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        x = np.array([10.0, 20.0, 30.0])
        bx_vals = np.zeros((len(x), len(y)))
        ds = _make_bx_dataset(bx_vals, x, y)

        psi = compute_flux_function(ds)

        np.testing.assert_array_equal(psi.coords["x"].values, x)
        np.testing.assert_array_equal(psi.coords["y"].values, y)
