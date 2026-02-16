"""Tests for reconn_wave_power.processing."""
import numpy as np
import xarray as xr
import pytest

from reconn_wave_power.processing import compute_flux_function


def _make_dataset(bx_vals, by_vals, x, y, time=None):
    """Helper to build a Dataset with Bx and By from raw arrays."""
    if time is not None:
        ds = xr.Dataset(
            {
                "Bx": (["time", "x", "y"], bx_vals),
                "By": (["time", "x", "y"], by_vals),
            },
            coords={"time": time, "x": x, "y": y},
        )
    else:
        ds = xr.Dataset(
            {
                "Bx": (["x", "y"], bx_vals),
                "By": (["x", "y"], by_vals),
            },
            coords={"x": x, "y": y},
        )
    return ds


class TestComputeFluxFunction:
    """Tests for compute_flux_function."""

    def test_uniform_bx_zero_by(self):
        """Bx = 1, By = 0 → ψ = y (integrated along y at x=0, constant along x)."""
        x = np.linspace(0, 5, 50)
        y = np.linspace(0, 3, 60)
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        bx_vals = np.ones((len(x), len(y)))
        by_vals = np.zeros((len(x), len(y)))
        ds = _make_dataset(bx_vals, by_vals, x, y)

        psi = compute_flux_function(ds)

        assert psi.name == "psi"
        assert psi.dims == ("x", "y")
        # ψ should equal y everywhere (boundary gives y, By=0 so no x correction)
        expected = y[np.newaxis, :] * np.ones((len(x), 1))
        # cumsum approximation: ψ(0, j) = sum(Bx[0, 1:j+1]) * dy
        expected_boundary = np.zeros(len(y))
        expected_boundary[1:] = np.cumsum(np.ones(len(y) - 1)) * dy
        np.testing.assert_allclose(psi.values[0], expected_boundary, atol=1e-10)
        # All x slices should be the same since By=0
        for ix in range(len(x)):
            np.testing.assert_allclose(psi.values[ix], expected_boundary, atol=1e-10)

    def test_uniform_by_zero_bx(self):
        """Bx = 0, By = 1 → ψ(x, y) = -x (boundary is 0, interior subtracts ∫By dx)."""
        x = np.linspace(0, 5, 100)
        y = np.linspace(0, 3, 60)
        dx = x[1] - x[0]
        bx_vals = np.zeros((len(x), len(y)))
        by_vals = np.ones((len(x), len(y)))
        ds = _make_dataset(bx_vals, by_vals, x, y)

        psi = compute_flux_function(ds)

        # Boundary at x=0: ψ(0, y) = 0 (since Bx=0)
        np.testing.assert_allclose(psi.values[0, :], 0.0, atol=1e-10)
        # Interior: ψ(x, y) = -∫₀ˣ By dx' = -x
        expected_col = np.zeros(len(x))
        expected_col[1:] = -np.cumsum(np.ones(len(x) - 1)) * dx
        for iy in range(len(y)):
            np.testing.assert_allclose(psi.values[:, iy], expected_col, atol=1e-10)

    def test_both_components(self):
        """Bx = -y, By = x gives ψ = -(x² + y²)/2."""
        x = np.linspace(0, 2, 200)
        y = np.linspace(0, 2, 200)
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        X, Y = np.meshgrid(x, y, indexing="ij")  # (nx, ny)
        bx_vals = -Y
        by_vals = X
        ds = _make_dataset(bx_vals, by_vals, x, y)

        psi = compute_flux_function(ds)

        expected = -0.5 * (X**2 + Y**2)
        # Tolerance scales with grid spacing
        np.testing.assert_allclose(psi.values, expected, atol=2 * max(dx, dy))

    def test_3d_with_time(self):
        """Works on (time, x, y) arrays without time_idx."""
        y = np.linspace(0, np.pi, 100)
        x = np.array([0.0, 0.5])
        time = np.array([0.0, 1.0])
        bx_vals = np.ones((2, len(x), len(y)))
        by_vals = np.zeros((2, len(x), len(y)))
        ds = _make_dataset(bx_vals, by_vals, x, y, time=time)

        psi = compute_flux_function(ds)

        assert psi.dims == ("time", "x", "y")
        dy = y[1] - y[0]
        expected_boundary = np.zeros(len(y))
        expected_boundary[1:] = np.cumsum(np.ones(len(y) - 1)) * dy
        np.testing.assert_allclose(psi.values[0, 0], expected_boundary, atol=1e-10)

    def test_time_idx_selects_single_frame(self):
        """time_idx reduces output to (x, y)."""
        y = np.linspace(0, 1, 50)
        x = np.array([0.0, 0.5])
        time = np.array([0.0, 1.0, 2.0])
        bx_vals = np.ones((3, len(x), len(y)))
        by_vals = np.zeros((3, len(x), len(y)))
        ds = _make_dataset(bx_vals, by_vals, x, y, time=time)

        psi = compute_flux_function(ds, time_idx=1)

        assert psi.dims == ("x", "y")
        assert "time" not in psi.dims

    def test_coordinates_preserved(self):
        """Output coordinates match input spatial coordinates."""
        y = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        x = np.array([10.0, 20.0, 30.0])
        bx_vals = np.zeros((len(x), len(y)))
        by_vals = np.zeros((len(x), len(y)))
        ds = _make_dataset(bx_vals, by_vals, x, y)

        psi = compute_flux_function(ds)

        np.testing.assert_array_equal(psi.coords["x"].values, x)
        np.testing.assert_array_equal(psi.coords["y"].values, y)
